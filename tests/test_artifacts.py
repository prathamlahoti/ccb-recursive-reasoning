from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ccb.artifacts import archive_directory


def test_archive_directory_writes_external_zip_atomically(tmp_path: Path) -> None:
    source = tmp_path / "records"
    source.mkdir()
    (source / "result.json").write_text('{"ok": true}\n', encoding="utf-8")

    archive = archive_directory(source, tmp_path / "records.zip")

    assert archive.exists()
    with zipfile.ZipFile(archive) as bundle:
        assert "records/result.json" in bundle.namelist()


def test_archive_directory_rejects_self_inclusion(tmp_path: Path) -> None:
    source = tmp_path / "records"
    source.mkdir()

    with pytest.raises(ValueError, match="outside"):
        archive_directory(source, source / "latest.zip")
