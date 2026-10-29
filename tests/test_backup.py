import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from sciquest.backup import backup_quest
from sciquest.core import create_quest
from sciquest.state import acquire_lock, get_record, record, release_lock


def test_backup_contains_consistent_database_and_verifiable_files(tmp_path):
    q = create_quest(tmp_path, {"hero_statement": "Preserve science"}, slug="backup")
    record(q, "evidence", "e1", {"outcome": "negative"})
    output = tmp_path / "backup.tar.gz"
    result = backup_quest(q, output)
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    restored = tmp_path / "restored"
    restored.mkdir()
    with tarfile.open(output, "r:gz") as archive:
        # The package creates this archive from checked regular relative files.
        for member in archive.getmembers():
            assert (
                not Path(member.name).is_absolute()
                and ".." not in Path(member.name).parts
            )
        archive.extractall(restored, filter="data")
    manifest = json.loads((restored / "backup_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((restored / name).read_bytes()).hexdigest() == expected
    assert get_record(restored, "evidence", "e1") == {"outcome": "negative"}
    with pytest.raises(FileExistsError):
        backup_quest(q, output)


def test_backup_refuses_active_runs_and_recursive_destination(tmp_path):
    q = create_quest(tmp_path, {"hero_statement": "Preserve science"}, slug="backup")
    with pytest.raises(ValueError, match="outside"):
        backup_quest(q, q / "backup.tar.gz")
    owner = acquire_lock(q)
    with pytest.raises(RuntimeError, match="active"):
        backup_quest(q, tmp_path / "active.tar.gz")
    release_lock(q, owner)
