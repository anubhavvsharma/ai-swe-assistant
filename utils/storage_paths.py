import os
import tempfile
from pathlib import Path


_DEFAULT_ROOT = Path("F:/ai_se_assistant_data")


def get_data_root() -> Path:
    override = (os.environ.get("ASSISTANT_DATA_DIR") or "").strip()
    root = Path(override) if override else _DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def make_upload_staging_dir() -> Path:
    """Unique empty directory for one ZIP extract (under data root / uploads)."""
    uploads = get_data_root() / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="upload_", dir=str(uploads)))


def make_clone_parent_dir() -> Path:
    """Unique empty parent directory for one ``git clone`` (under data root / clones)."""
    clones = get_data_root() / "clones"
    clones.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="gitclone_", dir=str(clones)))
