"""Helpers for resolving install-side resources vs. user data paths."""

from __future__ import annotations

import os
import pathlib

import modules.app_context as _app_ctx


def get_script_root() -> pathlib.Path:
    """Return the application install/source root."""
    return pathlib.Path(__file__).resolve().parent.parent


def _read_version(script_root: pathlib.Path) -> str:
    version_file = script_root / "version.txt"
    try:
        with open(version_file, "r") as f:
            return f.readline().strip()
    except OSError:
        return ""


def get_source_root(
    source_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Return the writable user data root for the current app session."""
    if source_path is not None:
        return pathlib.Path(source_path)

    ctx = _app_ctx.ctx
    if ctx is not None and getattr(ctx, "source_path", None):
        return pathlib.Path(ctx.source_path)

    script_root = get_script_root()
    if os.name != "nt" or not (script_root / "marker.lvpinstalled").exists():
        return script_root

    version = _read_version(script_root)
    if not version:
        return script_root

    import userpaths

    documents_dir = pathlib.Path(userpaths.get_my_documents())
    return documents_dir / f"LumaViewPro {version}"


def resolve_data_file(
    *parts: str,
    source_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve a file under the writable data/ directory."""
    return get_source_root(source_path).joinpath("data", *parts)


def resolve_script_file(*parts: str) -> pathlib.Path:
    """Resolve a file under the install/source root."""
    return get_script_root().joinpath(*parts)
