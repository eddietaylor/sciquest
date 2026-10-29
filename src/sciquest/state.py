"""Transactional supervisor state. YAML is a human-readable projection, not a lock."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import sqlite3
from typing import Any, Iterator
import uuid

from .io import canonical_json, read_yaml, utc_now, write_yaml
from .readonly import DATABASE_READS_ONLY


def _seed_state(qpath: Path) -> dict[str, Any]:
    seed = read_yaml(qpath / "state.yaml", {})
    if not isinstance(seed, dict):
        raise ValueError("state.yaml must be a mapping")
    if seed.get("schema_version") != 2 and seed.get("last_experiment"):
        seed["legacy_last_experiment"] = seed.get("last_experiment")
        seed["legacy_last_score"] = seed.get("last_score")
        seed["legacy_best_score"] = seed.get("best_score")
        seed.update(
            last_experiment=None,
            last_score=None,
            best_score=None,
            legacy_history_unverified=True,
        )
    return {
        "quest_status": "idle",
        "lock_id": None,
        "failures": [],
        **seed,
        "schema_version": 2,
    }


def authority_dir(qpath: Path) -> Path:
    directory = qpath / ".sciquest"
    if directory.is_symlink():
        raise ValueError("Supervisor storage must not be a symlink")
    if DATABASE_READS_ONLY.get():
        if not directory.is_dir():
            raise ValueError("Read-only audit requires existing supervisor storage")
        return directory
    directory.mkdir(exist_ok=True, mode=0o700)
    return directory


@contextmanager
def transaction(qpath: Path) -> Iterator[sqlite3.Connection]:
    db_path = authority_dir(qpath) / "state.sqlite3"
    if db_path.is_symlink():
        raise ValueError("Supervisor database must not be a symlink")
    if DATABASE_READS_ONLY.get():
        connection = sqlite3.connect(db_path.resolve().as_uri()+"?mode=ro", uri=True, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.close()
        return
    connection = sqlite3.connect(db_path, timeout=10, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, time TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS records (kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(kind,id))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS evidence_usage (hash TEXT NOT NULL, experiment_id TEXT NOT NULL, PRIMARY KEY(hash,experiment_id))"
        )
        if connection.execute("SELECT 1 FROM state WHERE id=1").fetchone() is None:
            seed = _seed_state(qpath)
            connection.execute("INSERT INTO state VALUES(1,?)", (canonical_json(seed),))
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _read(connection: sqlite3.Connection) -> dict[str, Any]:
    return json.loads(
        connection.execute("SELECT payload FROM state WHERE id=1").fetchone()[0]
    )


def _save(connection: sqlite3.Connection, qpath: Path, state: dict[str, Any]) -> None:
    state["last_updated"] = utc_now()
    connection.execute(
        "UPDATE state SET payload=? WHERE id=1", (canonical_json(state),)
    )
    # Serialize the projection inside the transaction so an older writer cannot
    # publish its view after a newer writer. Recovery always reads the database.
    write_yaml(qpath / "state.yaml", state)


def event(connection: sqlite3.Connection, kind: str, payload: dict[str, Any]) -> None:
    connection.execute(
        "INSERT INTO events(time,kind,payload) VALUES(?,?,?)",
        (utc_now(), kind, canonical_json(payload)),
    )


def read_state(qpath: Path) -> dict[str, Any]:
    if not (qpath / ".sciquest/state.sqlite3").is_file():
        return _seed_state(qpath)
    with transaction(qpath) as connection:
        return _read(connection)


def update_state(
    qpath: Path, changes: dict[str, Any], *, owner: str | None = None
) -> dict[str, Any]:
    with transaction(qpath) as connection:
        state = _read(connection)
        if state.get("lock_id") and state["lock_id"] != owner:
            raise RuntimeError("Quest is owned by another run")
        state.update(changes)
        _save(connection, qpath, state)
        return state


def acquire_lock(qpath: Path) -> str:
    with transaction(qpath) as connection:
        state = _read(connection)
        if state.get("lock_id") or state.get("quest_status") == "running":
            raise RuntimeError(f"Quest locked: {state.get('lock_id')}")
        if state.get("quest_status") not in {"idle", "failed"}:
            raise RuntimeError(f"Unknown quest status: {state.get('quest_status')!r}")
        if state.get("quest_status") == "failed":
            raise RuntimeError(
                "Quest failed; use recover before starting another iteration"
            )
        lock = uuid.uuid4().hex
        state.update(
            lock_id=lock,
            quest_status="running",
            owner_pid=os.getpid(),
            owner_host=socket.gethostname(),
            heartbeat=utc_now(),
        )
        _save(connection, qpath, state)
        event(connection, "lock_acquired", {"lock_id": lock, "pid": os.getpid()})
        return lock


def heartbeat(qpath: Path, owner: str) -> None:
    update_state(qpath, {"heartbeat": utc_now()}, owner=owner)


def release_lock(
    qpath: Path,
    lock_id: str,
    status: str = "idle",
    *,
    changes: dict[str, Any] | None = None,
) -> None:
    if status not in {"idle", "failed"}:
        raise ValueError("Released status must be idle or failed")
    with transaction(qpath) as connection:
        state = _read(connection)
        if state.get("lock_id") != lock_id:
            raise RuntimeError("Only the lock owner may finalize this run")
        state.update(changes or {})
        state.update(
            lock_id=None,
            quest_status=status,
            current_experiment=None,
            owner_pid=None,
            owner_host=None,
        )
        _save(connection, qpath, state)
        event(connection, "lock_released", {"lock_id": lock_id, "status": status})


def finalize_run(
    qpath: Path,
    owner: str,
    exp_id: str,
    completion: dict[str, Any],
    changes: dict[str, Any],
) -> None:
    """Commit completion and release ownership in one transaction."""
    with transaction(qpath) as connection:
        state = _read(connection)
        if state.get("lock_id") != owner:
            raise RuntimeError("Only the active owner may finalize a run")
        existing = connection.execute(
            "SELECT payload FROM records WHERE kind='completion' AND id=?", (exp_id,)
        ).fetchone()
        if existing and json.loads(existing[0]) != completion:
            raise ValueError("Conflicting completion record")
        if not existing:
            connection.execute(
                "INSERT INTO records VALUES('completion',?,?)",
                (exp_id, canonical_json(completion)),
            )
            event(connection, "completion", {"id": exp_id, "payload": completion})
        state.update(changes)
        state.update(
            lock_id=None,
            quest_status="idle",
            current_experiment=None,
            owner_pid=None,
            owner_host=None,
        )
        _save(connection, qpath, state)
        event(connection, "lock_released", {"lock_id": owner, "status": "idle"})


def record(
    qpath: Path,
    kind: str,
    record_id: str,
    payload: dict[str, Any],
    *,
    owner: str | None = None,
) -> None:
    """Insert an immutable supervisor record; duplicate IDs fail rather than replace."""
    with transaction(qpath) as connection:
        state = _read(connection)
        if state.get("lock_id") and state["lock_id"] != owner:
            raise RuntimeError("Quest is owned by another run")
        try:
            connection.execute(
                "INSERT INTO records VALUES(?,?,?)",
                (kind, record_id, canonical_json(payload)),
            )
            if kind == "evaluation":
                registration_row = connection.execute(
                    "SELECT payload FROM records WHERE kind='registration' AND id=?",
                    (record_id,),
                ).fetchone()
                if not registration_row:
                    raise ValueError(
                        "An evaluation requires an authoritative registration"
                    )
                registration = json.loads(registration_row[0])
                for name, digest in registration["file_hashes"].items():
                    if (
                        name.startswith("data/")
                        and registration["data_manifest"]["datasets"][
                            int(name.split("/")[1])
                        ].get("role")
                        == "evaluation"
                    ):
                        connection.execute(
                            "INSERT OR IGNORE INTO evidence_usage VALUES(?,?)",
                            (digest, record_id),
                        )
        except sqlite3.IntegrityError as exc:
            raise FileExistsError(f"{kind} {record_id} already exists") from exc
        event(connection, kind, {"id": record_id, "payload": payload})


def get_record(qpath: Path, kind: str, record_id: str) -> dict[str, Any] | None:
    if not (qpath / ".sciquest/state.sqlite3").is_file():
        return None
    with transaction(qpath) as connection:
        row = connection.execute(
            "SELECT payload FROM records WHERE kind=? AND id=?", (kind, record_id)
        ).fetchone()
        return json.loads(row[0]) if row else None


def list_records(
    qpath: Path, kind: str, limit: int = 100
) -> list[tuple[str, dict[str, Any]]]:
    if not 1 <= limit <= 10000:
        raise ValueError("Record retrieval limit must be between 1 and 10000")
    if not (qpath / ".sciquest/state.sqlite3").is_file():
        return []
    with transaction(qpath) as connection:
        rows = connection.execute(
            "SELECT id,payload FROM records WHERE kind=? ORDER BY rowid DESC LIMIT ?",
            (kind, limit),
        ).fetchall()
        return [(row[0], json.loads(row[1])) for row in rows]


def evidence_was_used(qpath: Path, hashes: set[str]) -> bool:
    if not (qpath / ".sciquest/state.sqlite3").is_file():
        return False
    with transaction(qpath) as connection:
        return any(
            connection.execute(
                "SELECT 1 FROM evidence_usage WHERE hash=? LIMIT 1", (digest,)
            ).fetchone()
            for digest in hashes
        )


def recover(qpath: Path, expected_lock: str | None = None) -> dict[str, Any]:
    """Recover a failed run or a confirmed dead local owner; never infer death from age."""
    with transaction(qpath) as connection:
        state = _read(connection)
        lock = state.get("lock_id")
        if lock:
            if expected_lock != lock:
                raise RuntimeError("Recovery requires the current lock ID")
            if state.get("owner_host") != socket.gethostname() or not state.get(
                "owner_pid"
            ):
                raise RuntimeError("Cannot verify that this lock owner is dead")
            try:
                os.kill(int(state["owner_pid"]), 0)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                raise RuntimeError(
                    "Cannot verify that this lock owner is dead"
                ) from exc
            else:
                raise RuntimeError("Owner process is still alive; recovery refused")
        elif state.get("quest_status") != "failed":
            raise RuntimeError("No failed or abandoned run to recover")
        event(connection, "recovery", {"previous_state": state})
        state.update(
            quest_status="idle",
            lock_id=None,
            current_experiment=None,
            owner_pid=None,
            owner_host=None,
        )
        _save(connection, qpath, state)
        return state
