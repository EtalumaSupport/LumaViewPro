"""Path helpers for the standalone stitching/blending prototype."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
DATA_DIR = PROJECT_ROOT / 'data'
SYNTHETIC_DATA_DIR = DATA_DIR / 'synthetic'
PUBLIC_DATA_DIR = DATA_DIR / 'public'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'


def ensure_project_dirs() -> dict[str, Path]:
    """Create the prototype data/output directories and return their paths."""
    dirs = {
        'root': PROJECT_ROOT,
        'data': DATA_DIR,
        'synthetic': SYNTHETIC_DATA_DIR,
        'public': PUBLIC_DATA_DIR,
        'outputs': OUTPUTS_DIR,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_from_root(path: str | Path) -> Path:
    """Resolve a relative path against the prototype root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate

