"""Provenance stamping for generated tables.

Every artifact this project publishes records the command, the git commit, the
UTC timestamp and the SHA-256 of its input, so a number quoted in the README
can be traced to the code and data that produced it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

TABLES_DIR = Path("reports/tables")


def git_commit() -> str:
    """Return the short git SHA of the working tree, marked dirty if uncommitted."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*, or an empty string if absent."""
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(
    frame: pd.DataFrame,
    name: str,
    description: str,
    command: str,
    inputs: list[Path] | None = None,
    directory: Path = TABLES_DIR,
) -> Path:
    """Write *frame* as CSV with a sibling ``.meta.json`` provenance record.

    Args:
        frame: Table to write.
        name: Base filename, without the extension.
        description: One line saying what the table is.
        command: The command a reader can run to regenerate it.
        inputs: Files the table was derived from; each is hashed.
        directory: Destination directory.

    Returns:
        The CSV path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    frame.to_csv(path, index=False)

    (directory / f"{name}.meta.json").write_text(
        json.dumps(
            {
                "description": description,
                "command": command,
                "git_commit": git_commit(),
                "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "inputs": [{"path": str(p), "sha256": sha256(p)} for p in (inputs or [])],
                "input_sha256": sha256(inputs[0]) if inputs else "",
                "rows": len(frame),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
