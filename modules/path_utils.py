"""Helpers for resolving install-side resources vs. user data paths."""

from __future__ import annotations

import os
import pathlib


MAX_COLLISION_SUFFIX = 999


class CaptureLocationError(Exception):
    """The capture location cannot hold a new output directory."""


def allocate_directory(desired: pathlib.Path) -> pathlib.Path:
    """Reserve a new directory at ``desired``, or the next free name after it.

    Output directory names are derived from second-resolution timestamps, so
    two captures started inside the same second ask for the same name. The
    reservation IS the creation: ``mkdir(exist_ok=False)`` fails if the name is
    taken, so two racing callers cannot both win one. Checking first and
    creating after would let the loser silently join -- and a joined directory
    mixes two captures' files under one manifest, with per-capture indices that
    restart, which reads back as a single scrambled capture.

    The plain name is kept when it is free, so ordinary output names never
    carry a suffix.

    Args:
        desired: The directory to create, suffix-free.

    Returns:
        The directory actually created -- ``desired``, or ``desired`` with a
        numeric suffix.

    Raises:
        CaptureLocationError: if the parent does not exist or is not writable,
            or if every candidate name is taken.
    """
    desired = pathlib.Path(desired)
    candidates = [desired] + [
        desired.with_name(f'{desired.name}_{i:03d}') for i in range(1, MAX_COLLISION_SUFFIX + 1)
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        except (FileNotFoundError, NotADirectoryError) as exc:
            # A missing parent means the configured capture location is wrong --
            # an unplugged drive, a stale path from another machine. Creating it
            # would put the capture in a new empty directory on whatever volume
            # happens to be mounted there, where the user will not look for it.
            raise CaptureLocationError(
                f'{desired.parent} is not an accessible capture location. '
                'Check that the save location exists and any external drive is '
                'connected, then try again.'
            ) from exc
        except OSError as exc:
            raise CaptureLocationError(
                f'Could not create {candidate} in the capture location: {exc}'
            ) from exc
        return candidate
    raise CaptureLocationError(
        f'{desired.name} and its first {MAX_COLLISION_SUFFIX} numbered variants '
        'all exist in the capture location. Move or remove some captures, or '
        'choose a different save location.'
    )


def capture_location_problem(parent_dir: pathlib.Path) -> str | None:
    """Say why ``parent_dir`` could not hold a new output directory, or None.

    Answers WITHOUT writing anything, because the caller is a run's
    preparation gate: it promises a refused request leaves nothing on
    disk, so a probe file is not available to it even as a temporary.

    The predicate is ``os.access`` on the nearest EXISTING ancestor,
    because the directory itself is normally absent -- ``ProtocolData``,
    ``Manual/Z-Stacks`` and their siblings are created by the first run
    that needs them, so testing the leaf alone would refuse every run
    into a fresh save location.

    Deliberately weaker than a write: it passes on a full disk, and on
    mounts that report permissions they do not honour. Those reach
    ``allocate_directory`` at commit time and are reported from there,
    as the failed run they are.

    Args:
        parent_dir: The directory the run would create its output in.

    Returns:
        A phrase naming the problem, for a caller to put in a sentence,
        or None when nothing about the location says it cannot be used.
    """
    existing = pathlib.Path(parent_dir)
    while not existing.exists():
        parent = existing.parent
        if parent == existing:
            # Walked the whole path without finding anything that exists:
            # an unmounted drive, or a path saved on another machine.
            return f'{existing} does not exist'
        existing = parent
    if not existing.is_dir():
        return f'{existing} is a file, not a folder'
    # Entering a directory and creating in it are separate permissions,
    # and a location missing either one cannot take the output.
    if not os.access(existing, os.W_OK | os.X_OK):
        return f'{existing} is not writable'
    return None


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
    version_file = script_root / 'version.txt'
    try:
        lines = version_file.read_text().splitlines()
        version = lines[0].strip() if len(lines) > 0 else ''
        build_timestamp = lines[1].strip() if len(lines) > 1 else ''
        return version, build_timestamp
    except FileNotFoundError:
        return '', ''
    except OSError:
        return '', ''


def _read_version(script_root: pathlib.Path) -> str:
    """Legacy wrapper -- returns version string only."""
    version, _ = read_version(script_root)
    return version


def get_source_root(
    source_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Return the writable user data root for the current app session."""
    if source_path is not None:
        return pathlib.Path(source_path)

    script_root = get_script_root()
    if os.name != 'nt' or not (script_root / 'marker.lvpinstalled').exists():
        return script_root

    version = _read_version(script_root)
    if not version:
        return script_root

    import platformdirs

    documents_dir = pathlib.Path(platformdirs.user_documents_dir())
    return documents_dir / f'LumaViewPro {version}'


def resolve_data_file(
    *parts: str,
    source_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve a file under the writable data/ directory."""
    return get_source_root(source_path).joinpath('data', *parts)


def resolve_script_file(*parts: str) -> pathlib.Path:
    """Resolve a file under the install/source root."""
    return get_script_root().joinpath(*parts)
