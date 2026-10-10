"""Helpers for resolving install-side resources vs. user data paths."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from enum import StrEnum

from modules.exceptions import InstallationFileError


MAX_COLLISION_SUFFIX = 999


class CaptureLocationError(Exception):
    """The capture location cannot hold a new output directory."""


def _inaccessible(location: pathlib.Path) -> CaptureLocationError:
    """The one sentence for a capture location that is not there."""
    return CaptureLocationError(
        f'{location} is not an accessible capture location. '
        'Check that the save location exists and any external drive is '
        'connected, then try again.'
    )


def require_capture_location(live_folder: str | pathlib.Path) -> pathlib.Path:
    """The live folder as a path, if it is there to capture into.

    A writer creates only its own levels inside the live folder, never the
    folder itself: the settings owner brings it up at startup, so a live
    folder that is missing now is an unplugged drive or a stale path, and
    creating it would put the capture in a new empty directory on whatever
    is mounted there, where the user will not look for it.

    Raises:
        CaptureLocationError: the folder does not exist or is not a folder.
    """
    folder = pathlib.Path(live_folder)
    if not folder.is_dir():
        raise _inaccessible(folder)
    return folder


def resolves_inside(root: str | pathlib.Path, path: str | pathlib.Path) -> bool:
    """Whether ``path`` is ``root`` or under it, once every link and ``..`` in both is followed.

    A name built from data -- a record's protocol path, an output named from
    a protocol's steps -- can climb out of the folder it is joined to; the
    text alone cannot tell, so both sides are resolved first.
    """
    return pathlib.Path(path).resolve().is_relative_to(pathlib.Path(root).resolve())


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
            raise _inaccessible(desired.parent) from exc
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


def move_aside(path: pathlib.Path, suffix: str) -> pathlib.Path:
    """Rename ``path`` to its name plus ``suffix``, or the next free numbered name.

    A name already taken is never replaced, so every copy moved aside before
    is kept.

    Returns:
        Where the file now is.

    Raises:
        OSError: the rename failed, or the name and its first
            MAX_COLLISION_SUFFIX numbered variants all exist.
    """
    candidates = [path.with_name(path.name + suffix)] + [
        path.with_name(f'{path.name}{suffix}_{i:03d}') for i in range(1, MAX_COLLISION_SUFFIX + 1)
    ]
    for candidate in candidates:
        if not candidate.exists():
            os.replace(path, candidate)
            return candidate
    raise FileExistsError(
        f'{path.name}{suffix} and its first {MAX_COLLISION_SUFFIX} numbered variants all exist'
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


# The installer drops this file beside the exe; nothing else writes it.
INSTALLED_MARKER = 'marker.lvpinstalled'


class AppRuntime(StrEnum):
    """How this process was launched: the one answer every reader takes.

    ``SOURCE``: a Python interpreter running a checkout or pip install.
    ``BUNDLE``: a PyInstaller build the installer did not install (a build
    run from ``dist``). ``INSTALLED``: a PyInstaller build in the folder the
    installer wrote, its marker beside the exe.
    """

    SOURCE = 'source'
    BUNDLE = 'bundle'
    INSTALLED = 'installed'

    @property
    def frozen(self) -> bool:
        """Whether the process is a PyInstaller build, installed or not."""
        return self is not AppRuntime.SOURCE


def launch_root() -> pathlib.Path:
    """The folder this process was launched from, links not followed.

    The exe's folder for a PyInstaller build, where the installer drops its
    marker; the checkout as the interpreter found it otherwise. Links are
    not followed: a simulator run from a scratch folder of links into a
    clone keeps its logs and settings in that folder, not the clone's.
    ``get_script_root`` follows them, for the build's own files and git.
    """
    if getattr(sys, 'frozen', False):
        return pathlib.Path(os.path.abspath(sys.executable)).parent
    return pathlib.Path(os.path.abspath(__file__)).parent.parent


def app_runtime() -> AppRuntime:
    """How this process was launched, read afresh on each call.

    A marker beside a source checkout does not make it installed: only a
    frozen build is ever installed.
    """
    if not getattr(sys, 'frozen', False):
        return AppRuntime.SOURCE
    if (launch_root() / INSTALLED_MARKER).exists():
        return AppRuntime.INSTALLED
    return AppRuntime.BUNDLE


def read_version(script_root: pathlib.Path | None = None) -> tuple[str, str]:
    """Read version and build timestamp from version.txt -- the one reader of it.

    Returns (version, build_timestamp). Either may be empty string on error.
    Line 1 = version string (path-safe, e.g., "4.0.0-beta2")
    Line 2 = build timestamp (display only, e.g., "2026-03-27 18:52")

    Read as utf-8-sig: a byte-order mark is not whitespace, so ``strip``
    would leave it on line 1. The version names the per-user data folder
    and is stamped into the TIFF Software tag of every saved image, where
    a non-ASCII character raises "TIFF strings must be 7-bit ASCII" and no
    image can be saved at all.
    """
    if script_root is None:
        script_root = get_script_root()
    version_file = pathlib.Path(script_root) / 'version.txt'
    try:
        lines = version_file.read_text(encoding='utf-8-sig').splitlines()
        version = lines[0].strip() if len(lines) > 0 else ''
        build_timestamp = lines[1].strip() if len(lines) > 1 else ''
        return version, build_timestamp
    except FileNotFoundError:
        return '', ''
    except (OSError, UnicodeDecodeError) as e:
        # Present but unreadable is not the source-tree case above: say so.
        # This runs while the logger module is still being imported, before
        # any handler exists, so the record reaches stderr through logging's
        # last-resort handler.
        logging.getLogger('LVP.modules.path_utils').warning(
            f'version.txt at {version_file} could not be read: {e}'
        )
        return '', ''


def data_folder_name(version: str) -> str:
    """The name of an installed build's per-user data folder, in Documents.

    Named here and placed by ``get_source_root``, which every reader of
    the folder takes it from: the logger, which writes the logs there, the
    GUI's environment, which copies the shipped data into it on first
    launch, and every reader of an installation file.
    """
    return f'LumaViewPro {version}'


def get_source_root(
    source_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Return the writable user data root for the current app session.

    The one derivation of it: an installed build keeps its data in
    Documents, in a folder named for its version; any other run keeps it in
    the folder it was launched from.

    Raises:
        InstallationFileError: an installed build whose ``version.txt`` names
            no version, so its data folder has no name.
    """
    if source_path is not None:
        return pathlib.Path(source_path)

    if app_runtime() is not AppRuntime.INSTALLED:
        return launch_root()

    version, _build_timestamp = read_version()
    if not version:
        raise InstallationFileError(get_script_root() / 'version.txt', 'names no version')

    import platformdirs

    documents_dir = pathlib.Path(platformdirs.user_documents_dir())
    return documents_dir / data_folder_name(version)


def desktop_folder() -> pathlib.Path:
    """The user's Desktop, where a person expects a support ZIP; the home folder when there is none.

    Through platformdirs, so a localized Desktop ("Schreibtisch" on German
    Windows) is the one found. The one Desktop for every caller that names
    it: the GUI's support report and logs zip, and the command-line report.
    """
    import platformdirs

    desktop = pathlib.Path(platformdirs.user_desktop_dir())
    return desktop if desktop.is_dir() else pathlib.Path.home()


def resolve_data_file(
    *parts: str,
    source_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve a file under the writable data/ directory."""
    return get_source_root(source_path).joinpath('data', *parts)


def read_installation_file(path: str | pathlib.Path) -> dict:
    """The JSON object in a file the installation ships, or a refusal naming the file.

    The one reader for these files, so every one of them fails the same
    way: the installation is at fault, not the user's settings, and a
    caller that answered a settings error by falling back to the shipped
    template would replace good settings and still fail on the same file.

    Raises:
        InstallationFileError: the file is missing, unreadable, not JSON,
            or holds something other than a JSON object.
    """
    try:
        with open(path, encoding='utf-8') as read_file:
            contents = json.load(read_file)
    except FileNotFoundError as e:
        raise InstallationFileError(path, 'is missing') from e
    except json.JSONDecodeError as e:
        raise InstallationFileError(path, f'is not valid JSON ({e})') from e
    except (OSError, UnicodeDecodeError) as e:
        raise InstallationFileError(path, f'cannot be read ({e})') from e
    if not isinstance(contents, dict):
        raise InstallationFileError(path, f'holds a {type(contents).__name__}, not a JSON object')
    return contents


def resolve_script_file(*parts: str) -> pathlib.Path:
    """Resolve a file under the install/source root."""
    return get_script_root().joinpath(*parts)
