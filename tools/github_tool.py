
from __future__ import annotations

import re
import shutil
from pathlib import Path

from git import Repo

from utils.storage_paths import make_clone_parent_dir


def _sanitize_name(url: str) -> str:
    m = re.search(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", url.strip())
    if m:
        return f"{m.group(1)}__{m.group(2)}"
    return "repo"


def clone_public_repo(repo_url: str, target_parent: Path | None = None) -> Path:
    """Clone a public HTTPS or SSH GitHub URL under ASSISTANT_DATA_DIR/clones (or target_parent)."""
    parent = target_parent or make_clone_parent_dir()
    parent.mkdir(parents=True, exist_ok=True)
    name = _sanitize_name(repo_url)
    dest = parent / name
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    Repo.clone_from(repo_url.strip(), str(dest), depth=1, single_branch=True)
    return dest
