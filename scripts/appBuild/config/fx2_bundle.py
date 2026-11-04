"""What the Windows release build carries for the FX2 (LS560, LS620, LS720).

Three kinds of file, each where the frozen app or the installer looks for it:

- the firmware hex the driver uploads at every connect, under ``firmware/``
  (``_FX2Connection.find_firmware_path`` reads ``<_MEIPASS>/firmware/``);
- libusb-package's own library inside the ``libusb_package`` folder, since
  ``libusb_package.get_library_path()`` looks only inside its package, and
  the driver runs on that file and no other;
- the driver step's executable at the root, which the installer runs to
  bind the FX2's driver (``modules/fx2_driver_step.py``).

The release spec imports this module, so the build is refused, naming what
is missing, before PyInstaller runs and again on the tree it produced.
"""

from __future__ import annotations

import os

FX2_HEX_FILES = ('LumascopeClassic.hex', 'Lumascope600.hex')
LIBUSB_PACKAGE_DIR = 'libusb_package'
DRIVER_STEP_EXE = 'lvp_fx2_driver_step.exe'


def fx2_bundle_entries(source_root: str, library_path) -> tuple[list, list]:
    """The spec's ``datas`` and ``binaries`` entries for the FX2.

    Args:
        source_root: The clone the build runs from.
        library_path: ``libusb_package.get_library_path()`` on the build
            host: None when the installed wheel holds no library.

    Raises:
        SystemExit: Naming each file the build would ship without.
    """
    datas = [(os.path.join(source_root, 'firmware', name), 'firmware') for name in FX2_HEX_FILES]
    missing = [source for source, _ in datas if not os.path.isfile(source)]
    if library_path is None:
        missing.append('libusb-package holds no library file on this build host')
    elif not os.path.isfile(library_path):
        missing.append(str(library_path))
    if missing:
        raise SystemExit(
            'FATAL: FX2 gate: the build would ship without what the FX2 needs: '
            + '; '.join(missing)
        )
    return datas, [(str(library_path), LIBUSB_PACKAGE_DIR)]


def check_fx2_dist(dist_root: str, library_name: str) -> None:
    """Refuse a built tree that lacks a file the FX2 needs where the app looks.

    Raises:
        SystemExit: Naming each file missing from ``dist_root``.
    """
    expected = [os.path.join('firmware', name) for name in FX2_HEX_FILES]
    expected.append(os.path.join(LIBUSB_PACKAGE_DIR, library_name))
    expected.append(DRIVER_STEP_EXE)
    missing = [rel for rel in expected if not os.path.isfile(os.path.join(dist_root, rel))]
    if missing:
        raise SystemExit(
            'FATAL: FX2 gate: missing from the dist tree where the app looks: ' + ', '.join(missing)
        )
