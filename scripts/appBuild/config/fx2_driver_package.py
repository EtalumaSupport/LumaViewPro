"""The Lumascope Classic driver package (LS460, LS560, LS620, LS720).

The signed pair, ``Lumascope_Classic.inf`` and its catalog
``Lumascope_Classic.cat``, is committed in
``scripts/appBuild/build_exe/fx2_driver/``, and every build carries it as
committed. Two jobs:

- ``write_driver_ver``, when ``build.ps1 -SignClassicDriver`` signs a new
  pair in the build's own clone: the INF's ``DriverVer`` becomes the signing
  day's date and the driver's own version, the committed one with its last
  part raised by one, since the driver step finds our current package by its
  ``DriverVer``. The version is the driver's, not the app's: a signing waits
  on no app version.
- ``check_driver_package``, run by every build before the MSI and by the
  suite: the INF is the one its catalog was signed over. The catalog holds
  the INF's SHA-256, so an INF edited, its comments or its line endings
  included, and not signed again is refused on any platform, at its own
  commit. The signer's chain and timestamp are checked once, by the signing
  run's ``signtool verify``, since nothing off Windows can read them and
  nothing after that run changes them.

Run by ``build.ps1``::

    python fx2_driver_package.py sign <the clone's inf>
    python fx2_driver_package.py check <the package's folder>

Each prints the ``DriverVer``: the one written, or the one checked.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import re
import sys

INF_NAME = 'Lumascope_Classic.inf'
# Each value runs to the end of its line, the line's ending left as it is.
_DRIVER_VER_LINE = re.compile(r'^(DriverVer[ \t]*=[ \t]*)([^\r\n]*)', re.MULTILINE)
_CATALOG_FILE_LINE = re.compile(r'^CatalogFile[ \t]*=[ \t]*([^\r\n]*)', re.MULTILINE)
_DRIVER_VER = re.compile(r'^(\d{2}/\d{2}/\d{4}),(\d+\.\d+\.\d+\.\d+)$')


def _refuse(message: str) -> SystemExit:
    return SystemExit(f'FATAL: Lumascope Classic driver package: {message}')


def _driver_ver(inf_path: str, text: str) -> tuple[str, str]:
    """The INF's one ``DriverVer``, as its date and its version.

    Raises:
        SystemExit: The INF holds no ``DriverVer`` line or several, or its
            value is not a real date and a four-part numeric version.
    """
    lines = _DRIVER_VER_LINE.findall(text)
    if len(lines) != 1:
        raise _refuse(f'{inf_path} holds {len(lines)} DriverVer lines, not one')
    value = lines[0][1].strip()
    match = _DRIVER_VER.match(value)
    if match is None:
        raise _refuse(f'{inf_path} DriverVer {value!r} is not mm/dd/yyyy,a.b.c.d')
    try:
        datetime.datetime.strptime(match.group(1), '%m/%d/%Y')
    except ValueError:
        raise _refuse(f'{inf_path} DriverVer {value!r} names no real date') from None
    return match.group(1), match.group(2)


def write_driver_ver(inf_path: str, sign_date: datetime.date) -> str:
    """Write the INF's ``DriverVer`` in place, for the catalog signed over it.

    Only the value changes: the catalog is made from the INF's bytes as
    committed. The driver step keeps the package with our current
    ``DriverVer`` and deletes our others, so each signing writes a version
    no signing wrote before: the committed one with its last part raised
    by one.

    Args:
        inf_path: The INF in the build's clone.
        sign_date: The signing day (UTC).

    Returns:
        The ``DriverVer`` written, date then version.

    Raises:
        SystemExit: The committed ``DriverVer`` is not one real date and
            four-part version.
    """
    with open(inf_path, 'rb') as f:
        text = f.read().decode('ascii')
    _, version = _driver_ver(inf_path, text)
    *head, last = version.split('.')
    written = f'{sign_date:%m/%d/%Y},{".".join([*head, str(int(last) + 1)])}'
    text = _DRIVER_VER_LINE.sub(lambda m: m.group(1) + written, text, count=1)
    with open(inf_path, 'wb') as f:
        f.write(text.encode('ascii'))
    return written


def check_driver_package(package_dir: str) -> str:
    """Refuse a package whose INF is not the one its catalog was signed over.

    Three facts, each readable on any platform: the INF's ``DriverVer`` is a
    real date and a four-part version; the catalog is the file the INF's
    ``CatalogFile`` names, by its exact name (Windows reads it by that name,
    and macOS's file system would match it in another case); and the INF's
    SHA-256 is in the catalog's bytes, where the signing put it, so the INF
    is byte for byte the one signed.

    Args:
        package_dir: The folder holding the INF and its catalog.

    Returns:
        The INF's ``DriverVer``.

    Raises:
        SystemExit: Naming the fact that fails.
    """
    inf_path = os.path.join(package_dir, INF_NAME)
    with open(inf_path, 'rb') as f:
        inf = f.read()
    text = inf.decode('ascii')
    date, version = _driver_ver(inf_path, text)
    catalogs = _CATALOG_FILE_LINE.findall(text)
    if len(catalogs) != 1:
        raise _refuse(f'{inf_path} holds {len(catalogs)} CatalogFile lines, not one')
    catalog_name = catalogs[0].strip().strip('"')
    if catalog_name not in os.listdir(package_dir):
        raise _refuse(f'{package_dir} holds no file named {catalog_name}, its CatalogFile')
    with open(os.path.join(package_dir, catalog_name), 'rb') as f:
        catalog = f.read()
    if hashlib.sha256(inf).digest() not in catalog:
        raise _refuse(
            f'{inf_path} is not the INF {catalog_name} was signed over: an edited INF '
            'needs a new signing (build.ps1 -SignClassicDriver)'
        )
    return f'{date},{version}'


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'sign':
        print(write_driver_ver(sys.argv[2], datetime.datetime.now(datetime.UTC).date()))
    elif len(sys.argv) == 3 and sys.argv[1] == 'check':
        print(check_driver_package(sys.argv[2]))
    else:
        raise SystemExit(
            "usage: fx2_driver_package.py sign <the clone's inf>, "
            "or fx2_driver_package.py check <the package's folder>"
        )
