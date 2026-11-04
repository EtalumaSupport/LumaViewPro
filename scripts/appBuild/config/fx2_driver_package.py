"""The FX2 driver package the Windows installer carries (LS560, LS620, LS720).

The INF lives in the repo with its ``DriverVer`` a placeholder, and the build
stamps it with the build's date and the installer version. Windows ranks
packages of equal signature for one hardware ID by that date, then by the
version, so a package stamped at build outranks a Zadig binding made before
the build, where a fixed date would lose to every one.

Run by ``build.ps1``::

    python fx2_driver_package.py <source.inf> <output dir> <installer version>

It prints the stamped INF's path.
"""

from __future__ import annotations

import datetime
import os
import re
import sys

PLACEHOLDER = '@DRIVER_VER@'
_INSTALLER_VERSION = re.compile(r'^\d+\.\d+\.\d+\.\d+$')


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


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(
            'usage: fx2_driver_package.py <source.inf> <output dir> <installer version>'
        )
    print(
        stamp_driver_package(
            sys.argv[1], sys.argv[2], sys.argv[3], datetime.datetime.now(datetime.UTC).date()
        )
    )
