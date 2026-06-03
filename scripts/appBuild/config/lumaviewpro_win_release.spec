# -*- mode: python ; coding: utf-8 -*-

from kivy_deps import sdl2, glew
from PyInstaller.utils.hooks import copy_metadata, collect_submodules

app_name = 'lumaviewpro'
datas = [
    ('data', 'data'),
    ('ui', 'ui'),
    ('modules', 'modules'),
    ('drivers', 'drivers'),
    ('docs/licenses', 'docs/licenses'),
    ('docs/LICENSE', 'docs'),
    ('version.txt', '.'),
    ('lvp_logger.py', '.'),
]

for pkg in ('numpy',):
    try:
        datas.extend(copy_metadata(pkg))
    except Exception:
        pass  # Package may not be installed on all build machines


# imagecodecs uses lazy submodule imports for each compression
# algorithm (lzw_encode, zlib_encode, etc.). PyInstaller can't follow
# them statically, so the previous {_imcd, _shared}-only hidden-import
# list missed lzw_encode and the bundle crashed on every TIFF write
# with `compression='lzw'`. collect_submodules picks up the full set;
# install-size cost is minor and future tifffile compression options
# just work.
hiddenimports = [
    *collect_submodules('imagecodecs'),
    'skimage.measure',
    'win32timezone',
    # FX2 (LVC) USB driver imports — pyusb + libusb1 dynamic submodules
    # that PyInstaller can't follow statically. Listed unconditionally
    # because drivers/fx2driver.py guards their import with try/except
    # (_FX2_AVAILABLE flag), so absence of the libraries at install
    # time degrades to "FX2 not available" rather than a crash.
    'usb',
    'usb.core',
    'usb.util',
    'usb.backend.libusb1',
    'usb1',
]

# Optional libusb-1.0.dll (FX2 USB I/O on Windows). Sourced by
# build.ps1 from dependencies\fx2\libusb-1.0.dll and passed via the
# FX2_LIBUSB_DLL env var. When unset/missing, FX2 cameras silently
# stay unsupported on the installed app — pyusb can't load its
# libusb1 backend without the native DLL on Windows. Bundling at the
# install root puts it in PyInstaller's _MEIPASS search path so
# ctypes.cdll.LoadLibrary("libusb-1.0.dll") inside pyusb finds it.
import os as _os
binaries = []
_fx2_dll = _os.environ.get('FX2_LIBUSB_DLL', '').strip()
if _fx2_dll and _os.path.exists(_fx2_dll):
    binaries.append((_fx2_dll, '.'))

a = Analysis(
    ['lumaviewpro.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Kivy optional deps LVP doesn't use. Listing them here suppresses
        # the build-time WARNING noise from PyInstaller's Kivy hooks and
        # keeps the bundle slightly smaller.
        # NOTE: 'enchant' is intentionally NOT excluded. Kivy's
        # kivy.core.spelling probes for it at import time; if absent, Kivy
        # logs CRITICAL during PyInstaller analysis and at runtime startup.
        # We install pyenchant via requirements.txt so the import succeeds
        # and the log stays clean.
        'kivy.lib.gstplayer',   # Kivy GStreamer video provider — LVP uses Pylon/IDS
    ],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

splash = Splash(
    'data/icons/etaluma_splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(10, 50),
    text_size=12,
    minify_script=True,
    always_on_top=True,
    text_color='black',
    text_default='',
)

exe = EXE(
    pyz,
    splash,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    contents_directory='.',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # Windowed build: no PyInstaller bootloader console window. Without
    # this, every .exe launch opened a black terminal alongside the
    # Kivy SDL2 window, and the lock-loser path's os._exit(1) (after
    # writing "ERROR: ... Exiting." to stderr) left that terminal
    # orphaned showing the last stderr line -- the "extra terminal
    # windows that say 'exiting'" symptom Chris reported on #559. LVP
    # log output is file-only (KIVY_NO_CONSOLELOG=1 at lumaviewpro.py
    # :115), so a windowed build doesn't lose any production logging.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['data\\icons\\icon.ico'],
)
coll = COLLECT(
    exe,
    splash.binaries,
    a.binaries,
    a.zipfiles,
    a.datas,
    *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)
