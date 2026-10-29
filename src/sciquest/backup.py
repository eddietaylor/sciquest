"""Consistent archive of an idle quest, including the authoritative SQLite store."""

from __future__ import annotations

from pathlib import Path
import os
import sqlite3
import tarfile
import tempfile
import json

from .io import file_hash, utc_now
from .state import transaction, _read


def backup_quest(qpath: Path, output: Path) -> dict:
    if output.resolve().is_relative_to(qpath.resolve()):
        raise ValueError("Write backups outside the quest directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Complete any initial legacy import before reading from a backup connection.
    with transaction(qpath):
        pass
    with transaction(qpath) as connection:
        if _read(connection).get("lock_id"):
            raise RuntimeError("Cannot archive an active quest")
        with tempfile.TemporaryDirectory(prefix="sciquest-backup-") as temp:
            staging = Path(temp)
            db = staging / "state.sqlite3"
            source = sqlite3.connect(qpath / ".sciquest/state.sqlite3")
            destination = sqlite3.connect(db)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()
            archive = staging / "quest.tar.gz"
            hashes = {}
            with tarfile.open(archive, "w:gz") as tar:
                for path in sorted(qpath.rglob("*")):
                    if path.is_symlink():
                        raise ValueError(
                            "Backups require regular files, not symlinked data"
                        )
                    if not path.is_file():
                        continue
                    relative = str(path.relative_to(qpath))
                    if relative.startswith(".sciquest/state.sqlite3"):
                        continue
                    before = file_hash(path)
                    tar.add(path, arcname=relative, recursive=False)
                    if file_hash(path) != before:
                        raise RuntimeError("Quest file changed during backup")
                    hashes[relative] = before
                tar.add(db, arcname=".sciquest/state.sqlite3")
                hashes[".sciquest/state.sqlite3"] = file_hash(db)
                manifest = staging / "backup_manifest.json"
                manifest.write_text(
                    json.dumps(
                        {"schema_version": 1, "created_at": utc_now(), "files": hashes},
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n"
                )
                tar.add(manifest, arcname="backup_manifest.json")
            # Atomic publication cannot overwrite an earlier archive.
            fd, published = tempfile.mkstemp(
                prefix=".sciquest-backup-", dir=output.parent
            )
            try:
                with os.fdopen(fd, "wb") as stream, archive.open("rb") as incoming:
                    import shutil

                    shutil.copyfileobj(incoming, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(published, output)
            finally:
                os.unlink(published)
    return {"archive": str(output), "sha256": file_hash(output), "files": len(hashes)}
