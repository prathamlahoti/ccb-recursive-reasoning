"""Safe, durable artifact helpers."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path


def archive_directory(source_directory: Path, archive_path: Path) -> Path:
    """Atomically write a ZIP of a directory to a sibling or external path.

    An archive located inside its source tree can recursively include a previous
    copy of itself. This can exhaust notebook disk space. The output path is
    rejected when it is inside the source directory.
    """

    source = source_directory.resolve()
    destination = archive_path.resolve()
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")
    if destination.suffix != ".zip":
        raise ValueError("archive path must end in .zip")
    if destination.is_relative_to(source):
        raise ValueError(
            "archive path must be outside its source directory to prevent "
            f"recursive self-inclusion: {destination}"
        )

    temporary_base = destination.with_name(
        f".{destination.stem}-{uuid.uuid4().hex}.tmp"
    )
    temporary_zip = Path(
        shutil.make_archive(
            str(temporary_base),
            "zip",
            root_dir=source.parent,
            base_dir=source.name,
        )
    )
    os.replace(temporary_zip, destination)
    return destination
