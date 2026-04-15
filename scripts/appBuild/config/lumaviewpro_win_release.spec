# -*- mode: python ; coding: utf-8 -*-

from kivy_deps import sdl2, glew
from PyInstaller.utils.hooks import copy_metadata

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
    # FX2 firmware hex files — uploaded to the Cypress FX2 chip on
    # every LS620 / LS560 / LS720 connect (the FX2 has no flash ROM).
    # _find_firmware_path() in drivers/fx2driver.py searches
    # sys._MEIPASS/firmware/ under PyInstaller, so this copies the
    # whole firmware directory into the bundle root. Preferred:
    # LumascopeClassic.hex (patched "LS Classic" product string for
    # detection). Fallback: Lumascope600.hex.
    ('firmware', 'firmware'),
]

for pkg in ('numpy', 'scyjava', 'imglyb', 'pyimagej'):
    try:
        datas.extend(copy_metadata(pkg))
    except Exception:
        pass  # Package may not be installed on all build machines


hiddenimports = [
    'imagecodecs._imcd',
    'imagecodecs._shared',
    'skimage.measure',
    'win32timezone',
    # FX2 driver (drivers/fx2driver.py) uses pyusb + python-libusb1
    # for USB control transfers and ISO streaming. PyInstaller's
    # static analysis misses backend submodules because pyusb resolves
    # them dynamically at runtime via usb.backend.
    'usb',
    'usb.core',
    'usb.util',
    'usb.backend',
    'usb.backend.libusb1',
    'usb1',
]

# FX2 driver requires libusb-1.0.dll at runtime on Windows for ISO
# streaming via python-libusb1 (drivers.winusb_iso handles the Windows
# native WinUSB path separately). The DLL must be vendored into the
# repo before a Windows build will work end-to-end — drop the file at:
#     drivers/third_party/windows/libusb-1.0.dll
# and add:
#     binaries = [
#         ('drivers/third_party/windows/libusb-1.0.dll', '.'),
#     ]
# below. The DLL is LGPL 2.1+ (https://libusb.info/) — verify the
# version and SHA256 before committing it. macOS/Linux builds don't
# need this entry; they rely on the native libusb from Homebrew / apt.
# TRACKED: Stage 3 of the LVC port; see `project_lumaviewclassic_repo.md`
# in auto-memory and the comment in drivers/fx2driver.py module docstring.
binaries = []

a = Analysis(
    ['lumaviewpro.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    console=True,
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
