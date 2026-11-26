"""What the Windows release build carries for the FX2 (LS460, LS560, LS620, LS720).

Four kinds of file, each where the frozen app or the installer looks for it:

- the firmware hex the driver uploads at every connect, under ``firmware/``
  (``_FX2Connection.find_firmware_path`` reads ``<build root>/firmware/``, the
  bundle's folder in the frozen build);
- the driver's Windows libusb beside its module, in
  ``_lumascope/drivers/libusb_windows/``, the one place the frozen driver
  looks for it (``libusb_windows.library_path()``), with the patches it was
  built with in ``docs/licenses/`` beside the LGPL text;
- the driver step's executable at the root, which the installer runs to
  bind the FX2's driver (``src/_lumascope/modules/fx2_driver_step.py``).

The release spec imports this module, so the build is refused, naming what
is missing, before PyInstaller runs and again on the tree it produced.
"""

from __future__ import annotations

import glob
import importlib.util
import os

FX2_HEX_FILES = ('LumascopeClassic.hex', 'Lumascope600.hex')
DRIVER_STEP_EXE = 'lvp_fx2_driver_step.exe'
LICENCES_DIR = os.path.join('docs', 'licenses')
# The driver's module that names its Windows libusb, from the clone's root.
_LIBUSB_WINDOWS_MODULE = os.path.join(
    'src', '_lumascope', 'drivers', 'libusb_windows', '__init__.py'
)


def _windows_libusb(source_root: str) -> tuple[str, str, list[str]]:
    """The driver's Windows libusb: its file, its folder in the frozen tree, its patches.

    The path is the driver's own (``libusb_windows.library_path()``), loaded
    from the clone by file so the build host need not import LumaViewPro;
    the frozen tree keeps the folder's place under ``src/``, where the
    frozen module looks beside itself.
    """
    module_file = os.path.join(source_root, _LIBUSB_WINDOWS_MODULE)
    spec = importlib.util.spec_from_file_location('fx2_bundle_libusb_windows', module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = str(module.library_path())
    folder = os.path.dirname(library)
    dest = os.path.relpath(folder, os.path.realpath(os.path.join(source_root, 'src')))
    return library, dest, sorted(glob.glob(os.path.join(folder, '*.patch')))


def fx2_bundle_entries(source_root: str) -> tuple[list, list]:
    """The spec's ``datas`` and ``binaries`` entries for the FX2.

    Args:
        source_root: The clone the build runs from.

    Raises:
        SystemExit: Naming each file the build would ship without.
    """
    datas = [(os.path.join(source_root, 'firmware', name), 'firmware') for name in FX2_HEX_FILES]
    missing = [source for source, _ in datas if not os.path.isfile(source)]
    library, dest, patches = _windows_libusb(source_root)
    if not os.path.isfile(library):
        missing.append(library)
    if not patches:
        missing.append(f'the patches beside {library}')
    if missing:
        raise SystemExit(
            'FATAL: FX2 gate: the build would ship without what the FX2 needs: '
            + '; '.join(missing)
        )
    datas.extend((patch, LICENCES_DIR) for patch in patches)
    return datas, [(library, dest)]


def check_fx2_dist(dist_root: str, source_root: str) -> None:
    """Refuse a built tree that lacks a file the FX2 needs where the app looks.

    Args:
        dist_root: The tree PyInstaller built.
        source_root: The clone it was built from, which names the libusb's place.

    Raises:
        SystemExit: Naming each file missing from ``dist_root``.
    """
    library, dest, patches = _windows_libusb(source_root)
    expected = [os.path.join('firmware', name) for name in FX2_HEX_FILES]
    expected.append(os.path.join(dest, os.path.basename(library)))
    expected.extend(os.path.join(LICENCES_DIR, os.path.basename(patch)) for patch in patches)
    expected.append(DRIVER_STEP_EXE)
    missing = [rel for rel in expected if not os.path.isfile(os.path.join(dist_root, rel))]
    if missing:
        raise SystemExit(
            'FATAL: FX2 gate: missing from the dist tree where the app looks: ' + ', '.join(missing)
        )
