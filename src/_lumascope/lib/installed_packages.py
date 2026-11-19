# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Whether a source run's installed packages are the ones ``requirements.txt`` pins.

``requirements.txt`` is the one statement of what a source run needs. A
package at another version imports and runs until some call no longer
answers the same way, and then fails wherever that call is, worded for that
site's guess (a USB cable); a missing one fails at its first import,
wherever that is. So the package's own import judges the environment against
the file before anything third-party loads, and refuses the run on a row
(``PackagesNotAsPinnedError``). An installed build is not judged: its
environment was built to the pins, and the bundle carries no
``requirements.txt``.

Standard library only at import: this runs before any third-party package
has been seen to be there. ``packaging`` parses the file and evaluates each
line's environment marker, and is imported inside the judge, so its own
absence is a row like any other.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import pathlib
import sys
import sysconfig
from dataclasses import dataclass
from typing import ClassVar

from _lumascope.lib.build_files import build_root
from _lumascope.lib.lumascope_error import Fault, RefusalCause

PINS = 'requirements.txt'


@dataclass(frozen=True)
class PinRow:
    """One package installed otherwise than its pin.

    Attributes:
        name: The package as ``requirements.txt`` names it.
        installed: The version its metadata states, or None when it is not
            installed.
        pinned: The pin's version clause as the file states it (``==26.10``).
    """

    name: str
    installed: str | None
    pinned: str


@dataclass(frozen=True)
class RunningPython:
    """The facts about the Python that ran which decide how its packages can be fixed.

    Attributes:
        executable: The interpreter's path (``sys.executable``).
        in_venv: Whether it is a virtual environment's.
        has_pip: Whether pip is importable by it.
        externally_managed: Whether its standard library carries the
            ``EXTERNALLY-MANAGED`` marker (PEP 668), by which a system or
            Homebrew Python refuses a pip install outside a virtual
            environment.
    """

    executable: str
    in_venv: bool
    has_pip: bool
    externally_managed: bool

    @property
    def can_install(self) -> bool:
        """Whether ``<executable> -m pip install`` can install into this Python."""
        return self.has_pip and (self.in_venv or not self.externally_managed)


def running_python() -> RunningPython:
    """The facts about the Python running this process."""
    return RunningPython(
        executable=sys.executable,
        in_venv=sys.prefix != sys.base_prefix,
        has_pip=importlib.util.find_spec('pip') is not None,
        externally_managed=(
            pathlib.Path(sysconfig.get_path('stdlib')) / 'EXTERNALLY-MANAGED'
        ).is_file(),
    )


def packages_unlike_the_pins(pins: pathlib.Path | None = None) -> list[PinRow]:
    """Every package that applies to this host and is not installed as ``pins`` pins it.

    Args:
        pins: The requirements file; the build's own ``requirements.txt``
            when None.

    Returns:
        The rows, in the file's order; empty when every pin that applies is
        met. A line whose environment marker does not hold here is not
        judged. With ``packaging`` itself missing, the file cannot be judged
        and its one row is ``packaging``'s.

    Raises:
        PackagesNotAsPinnedError: ``pins_unread``: the file cannot be read,
            or a line in it is not a requirement.
    """
    pins = build_root() / PINS if pins is None else pins
    try:
        lines = pins.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeDecodeError) as e:
        raise PackagesNotAsPinnedError.pins_unread(pins, str(e)) from e
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line and not line.startswith('#')]
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        # The one pin that cannot be judged by parsing: its own parser.
        pinned = next(
            (line.partition('==')[2] for line in lines if line.partition('==')[0] == 'packaging'),
            '',
        )
        return [PinRow('packaging', None, f'=={pinned}' if pinned else 'pinned')]
    rows = []
    for line in lines:
        try:
            requirement = Requirement(line)
        except InvalidRequirement as e:
            raise PackagesNotAsPinnedError.pins_unread(
                pins, f'{line!r} is not a requirement: {e}'
            ) from e
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is None or not requirement.specifier.contains(installed, prereleases=True):
            rows.append(PinRow(requirement.name, installed, str(requirement.specifier)))
    return rows


def _fix_scripts(python: RunningPython) -> tuple[str, str, str]:
    """This host's install script, the run script it writes, and the Python it installs into."""
    root = build_root()
    if sys.platform == 'win32':
        script = str(root / 'scripts' / 'install_windows.bat')
        # A venv's Python is told to update its venv, not moved to the system Python.
        if python.in_venv:
            script += ' --venv'
        run = root / 'run.bat'
        return script, str(run), f'the one {run} names'
    name = 'install_mac.sh' if sys.platform == 'darwin' else 'install_linux.sh'
    return str(root / 'scripts' / name), str(root / 'run.sh'), str(root / 'venv' / 'bin' / 'python')


class PackagesNotAsPinnedError(Fault, Exception):
    """A source run's installed packages are not the ones ``requirements.txt`` pins.

    Raised at the API package's import, before anything third-party loads,
    on every host: the GUI shows it as a dialog, and the REST server and any
    other program show its words as the uncaught exception's message.
    No ``except`` can name it, since the import that would bind the name
    is the one that raises. The words name the Python that ran and the fix,
    each half for the launch it fixes: the install script with the run
    script it writes, to start LumaViewPro; the pip line naming this Python
    and the file by absolute path, to run another program with this Python,
    shown only when this Python can install into itself. The install script
    can choose another Python than the one that ran, so the halves are
    never alternatives for one launch.

    Attributes:
        reason: ``'packages_not_as_pinned'`` -- a package that applies to
            this host is missing or at another version than its pin.
            ``'pins_unread'`` -- ``requirements.txt`` cannot be read, or a
            line in it is not a requirement, so nothing can be judged.
        rows: The packages not as pinned; empty for ``pins_unread``.
        python: The Python that ran.
    """

    causes: ClassVar[dict[str, RefusalCause | None]] = {
        'packages_not_as_pinned': None,
        'pins_unread': None,
    }
    words_for_person = True
    title = 'Python Packages Not as Pinned'

    def __init__(self, rows: list[PinRow], python: RunningPython, words: str, reason: str):
        super().__init__(words)
        self.reason = reason
        self.rows = rows
        self.python = python

    @classmethod
    def not_as_pinned(cls, rows: list[PinRow], python: RunningPython) -> PackagesNotAsPinnedError:
        """The refusal for ``rows``, its fix worded from what ``python`` can do."""
        script, run, script_python = _fix_scripts(python)
        listed = '\n'.join(
            f'  {row.name}: '
            f'{"not installed" if row.installed is None else f"{row.installed} installed"}, '
            f'{row.pinned} pinned'
            for row in rows
        )
        start = f'To start LumaViewPro, run {script}, then start it with {run}.'
        if python.can_install:
            own = (
                'To run your own program or the REST server with this Python, run:\n'
                f'  "{python.executable}" -m pip install -r "{build_root() / PINS}"'
            )
        else:
            why = 'it has no pip' if not python.has_pip else 'it is externally managed'
            own = (
                f'This Python cannot install packages into itself ({why}): run your own '
                f'program or the REST server with the Python the install script installs '
                f'into, {script_python}.'
            )
        words = (
            f'LumaViewPro cannot start: the Python packages installed for '
            f'{python.executable} are not the ones {PINS} pins.\n{listed}\n\n{start}\n{own}'
        )
        return cls(rows, python, words, 'packages_not_as_pinned')

    @classmethod
    def pins_unread(cls, pins: pathlib.Path, problem: str) -> PackagesNotAsPinnedError:
        """The refusal for a requirements file that cannot be judged against."""
        words = (
            f'LumaViewPro cannot start: {pins} cannot be read ({problem}), so the installed '
            'Python packages cannot be checked against it. Get the LumaViewPro source again.'
        )
        return cls([], running_python(), words, 'pins_unread')
