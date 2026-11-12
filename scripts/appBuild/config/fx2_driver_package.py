"""The FX2 driver package's ``DriverVer`` (LS460, LS560, LS620, LS720).

The INF lives in the repo. ``write_driver_ver`` writes its ``DriverVer`` when
``build.ps1 -SignClassicDriver`` signs the package, in the build's own
clone: the signing day's date and the driver's own version, the committed
INF's with its last part raised by one, since the driver step finds our
current package by its ``DriverVer``. The version is the driver's, not the
app's: a signing waits on no app version. Until the signed pair is
committed, the INF holds a placeholder and a signed build stamps a copy
(``stamp_driver_package``) with the installer version.

Run by ``build.ps1``::

    python fx2_driver_package.py <source.inf> <output dir> <installer version>
    python fx2_driver_package.py sign <the clone's inf>

Each prints what it wrote: the stamped INF's path, or the new ``DriverVer``.
"""

from __future__ import annotations

import datetime
import os
import re
import sys

PLACEHOLDER = '@DRIVER_VER@'
# The driver's first signed version: 2.0.0.0, apart from the installer's
# 4.0.0.x that a build stamps, so no signing can write a DriverVer a build
# stamped.
FIRST_DRIVER_VERSION = '2.0.0.0'
_INSTALLER_VERSION = re.compile(r'^\d+\.\d+\.\d+\.\d+$')
# The value runs to the end of its line, the line's ending left as it is.
_DRIVER_VER_LINE = re.compile(r'^(DriverVer[ \t]*=[ \t]*)([^\r\n]*)', re.MULTILINE)


def stamp_driver_package(
    source_inf: str, output_dir: str, installer_version: str, build_date: datetime.date
) -> str:
    """Write the INF into ``output_dir`` with its ``DriverVer`` stamped.

    Args:
        source_inf: The repo's INF, holding the placeholder once.
        output_dir: Where the stamped copy is written, under the same name.
        installer_version: The installer's four-part version (``4.0.0.38``).
        build_date: The build's date (UTC).

    Returns:
        The stamped INF's path.

    Raises:
        SystemExit: The version is not four numeric parts, or the source does
            not hold the placeholder exactly once.
    """
    if not _INSTALLER_VERSION.match(installer_version):
        raise SystemExit(
            f'FATAL: FX2 driver package: DriverVer needs a four-part numeric version, got {installer_version!r}'
        )
    with open(source_inf, encoding='ascii') as f:
        text = f.read()
    count = text.count(PLACEHOLDER)
    if count != 1:
        raise SystemExit(
            f'FATAL: FX2 driver package: {source_inf} holds {PLACEHOLDER} {count} times, not once'
        )
    stamp = f'{build_date:%m/%d/%Y},{installer_version}'
    os.makedirs(output_dir, exist_ok=True)
    stamped = os.path.join(output_dir, os.path.basename(source_inf))
    with open(stamped, 'w', encoding='ascii', newline='\r\n') as f:
        f.write(text.replace(PLACEHOLDER, stamp))
    return stamped


def write_driver_ver(inf_path: str, sign_date: datetime.date) -> str:
    """Write the INF's ``DriverVer`` in place, for the catalog signed over it.

    Only the value changes: the catalog is made from the INF's bytes as
    committed. The driver step keeps the package with our current
    ``DriverVer`` and deletes our others, so each signing writes a version
    no signing wrote before: the first ``FIRST_DRIVER_VERSION``, each after
    it the committed one with its last part raised by one.

    Args:
        inf_path: The INF in the build's clone.
        sign_date: The signing day (UTC).

    Returns:
        The ``DriverVer`` written, date then version.

    Raises:
        SystemExit: The INF does not hold one ``DriverVer`` line.
    """
    with open(inf_path, 'rb') as f:
        text = f.read().decode('ascii')
    lines = _DRIVER_VER_LINE.findall(text)
    if len(lines) != 1:
        raise SystemExit(
            f'FATAL: FX2 driver package: {inf_path} holds {len(lines)} DriverVer lines, not one'
        )
    current = lines[0][1].strip()
    if current == PLACEHOLDER:
        version = FIRST_DRIVER_VERSION
    else:
        *head, last = current.partition(',')[2].strip().split('.')
        version = '.'.join([*head, str(int(last) + 1)])
    written = f'{sign_date:%m/%d/%Y},{version}'
    text = _DRIVER_VER_LINE.sub(lambda m: m.group(1) + written, text, count=1)
    with open(inf_path, 'wb') as f:
        f.write(text.encode('ascii'))
    return written


if __name__ == '__main__':
    today = datetime.datetime.now(datetime.UTC).date()
    if len(sys.argv) == 3 and sys.argv[1] == 'sign':
        print(write_driver_ver(sys.argv[2], today))
    elif len(sys.argv) == 4:
        print(stamp_driver_package(sys.argv[1], sys.argv[2], sys.argv[3], today))
    else:
        raise SystemExit(
            'usage: fx2_driver_package.py <source.inf> <output dir> <installer version>, '
            "or fx2_driver_package.py sign <the clone's inf>"
        )
