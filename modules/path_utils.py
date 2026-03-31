"""Helpers for resolving install-side resources vs. user data paths."""

from __future__ import annotations

import os
import pathlib

import modules.app_context as _app_ctx


def get_script_root() -> pathlib.Path:
    """Return the application install/source root."""
    return pathlib.Path(__file__).resolve().parent.parent


def read_version(script_root: pathlib.Path | None = None) -> tuple[str, str]:
    """Read version and build timestamp from version.txt.

    Returns (version, build_timestamp). Either may be empty string on error.
    Line 1 = version string (path-safe, e.g., "4.0.0-beta2")
    Line 2 = build timestamp (display only, e.g., "2026-03-27 18:52")
    """
    if script_root is None:
        script_root = get_script_root()
    version_file = script_root / "version.txt"
    try:
        lines = version_file.read_text().splitlines()
        version = lines[0].strip() if len(lines) > 0 else ""
        build_timestamp = lines[1].strip() if len(lines) > 1 else ""
        return version, build_timestamp
    except FileNotFoundError:
        return "", ""
    except OSError:
        return "", ""


def _read_version(script_root: pathlib.Path) -> str:
    """Legacy wrapper — returns version string only."""
    version, _ = read_version(script_root)
    return version


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
