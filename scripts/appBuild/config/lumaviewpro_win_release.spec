# -*- mode: python ; coding: utf-8 -*-

from kivy_deps import sdl2, glew
from PyInstaller.utils.hooks import copy_metadata, collect_all, collect_submodules

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

# Every ids_peak* package is collected WHOLESALE, never left to inference:
# afl/icv are reached only through runtime probes that build the import
# path from strings (drivers/idscamera.py), so static analysis never sees
# them at all; and for the graph-visible base packages, binary-dependency
# inference is NON-DETERMINISTIC for secondary DLLs -- one build included
# ids_peak's log4cpp DLL and the next silently dropped it, which on a
# client machine surfaces as a bare "DLL load failed" with no build-time
# signal. collect_all mirrors each wheel's package dir exactly; the
# content census in build.ps1 verifies the result name-by-name.
for _ids_pkg in ('ids_peak', 'ids_peak_ipl', 'ids_peak_afl', 'ids_peak_icv'):
    _ids_datas, _ids_binaries, _ids_hidden = collect_all(_ids_pkg)
    datas += _ids_datas
    binaries += _ids_binaries
    hiddenimports += _ids_hidden

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
        # Drops the module from the dependency graph, which keeps the
        # bundle smaller. It does NOT quiet the build log: PyInstaller's
        # Kivy hook logs while it runs, before graph exclusion applies.
        #
        # Expect CRITICAL "Spelling: Unable to find any valuable Spelling
        # provider" lines during the build. They are Kivy's, they concern
        # a spell-check provider LVP never imports, and nothing keys on
        # them. Considered silencing them two ways and rejected both:
        # installing pyenchant costs ~2-4 MB shipped to buy one clean log
        # line, and excluding 'enchant' here was tried and had no effect
        # (the hook logs before this list is consulted). Revisit only if a
        # build gate ever keys on CRITICAL lines.
        'kivy.lib.gstplayer',   # Kivy GStreamer video provider — LVP uses Pylon/IDS
    ],
    noarchive=False,
)

# Camera-SDK completeness gate. The dist-folder census in build.ps1 sees
# only the BINARY halves of these packages; the pure-Python wrapper
# modules live in the frozen archive and are invisible on disk -- a
# bundle can carry every DLL yet fail `from ids_peak import ids_peak` at
# runtime. Assert the wrappers made it into the analysis, so a
# camera-blind exe fails HERE instead of on a client machine.
_required_pure_modules = [
    'pypylon.pylon',
    'ids_peak.ids_peak',
    'ids_peak.ids_peak_ipl_extension',
    'ids_peak_ipl',
]
_analyzed = {entry[0] for entry in a.pure}
_missing_pure = [m for m in _required_pure_modules if m not in _analyzed]
if _missing_pure:
    raise SystemExit(
        f'FATAL: camera-SDK modules missing from the frozen bundle: {_missing_pure}. '
        f'The exe would import-fail these bindings at runtime and silently lose '
        f'that camera class. Check the build venv wheels and PyInstaller hooks.'
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
    # windows that say 'exiting'" symptom researchers see on startup. LVP
    # log output is file-only (KIVY_NO_CONSOLELOG=1 at lumaviewpro.py
    # :115), so a windowed build doesn't lose any production logging.
    console=False,
    # Suppress the PyInstaller bootloader's windowed-traceback dialog. On a
    # hard crash (an exception escaping to the bootloader) PyInstaller pops a
    # Windows message box containing a raw Python traceback -- a researcher
    # must never see that. The crash is still captured: custom_except_hook
    # logs uncaught exceptions to the file logs, and notifications.critical
    # surfaces a plain-language popup at the app layer.
    disable_windowed_traceback=True,
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

# CRT policy gate. PyInstaller collects a VC++ runtime from the build
# environment by design, with no version control: one build shipped an
# app-root msvcp140.dll a decade older than its own companions, and
# because an app-local DLL shadows System32 for the whole process, the
# IDS SDK's native module failed DLL init on client machines -- while
# the same machine's System32 copy worked (hardware-isolated: removing
# only that file restored full camera acquisition). Policy: the app
# ships NO msvcp140/concrt140 of its own; the installer chains the
# official VC++ Redistributable so the process resolves them from
# System32 exactly like a source checkout does. vcruntime140*.dll stays
# app-local -- python3xx.dll imports it, the exe cannot start without
# it. This runs post-COLLECT so every collection channel (Analysis,
# Splash, Tree) is covered, and it fails the build outright if a
# forbidden copy survives -- a bad runtime must die here, not on a
# client machine.
_dist_root = _os.path.join(DISTPATH, app_name)
_forbidden_crt = ('msvcp140.dll', 'concrt140.dll')

for _crt_name in _forbidden_crt:
    _crt_path = _os.path.join(_dist_root, _crt_name)
    if _os.path.exists(_crt_path):
        _os.remove(_crt_path)
        print(f'CRT policy: removed app-root {_crt_name}')

_crt_leftover = [
    _n for _n in _os.listdir(_dist_root) if _n.lower() in _forbidden_crt
]
if _crt_leftover:
    raise SystemExit(
        f'FATAL: CRT policy: forbidden runtime DLLs still at the app root '
        f'after removal: {_crt_leftover}'
    )

if not _os.path.exists(_os.path.join(_dist_root, 'vcruntime140.dll')):
    raise SystemExit(
        'FATAL: CRT policy: vcruntime140.dll missing at the app root -- '
        'python3xx.dll imports it; the frozen exe cannot start. Check the '
        'build environment before shipping.'
    )

# Inventory every VC-family DLL that ships, so the build transcript
# records the exact runtime set and build.ps1 can enforce that the
# chained redistributable is at least as new as anything bundled.
import re as _re
_vc_family = _re.compile(
    r'^(msvcp140|vcruntime140|concrt140)[a-z0-9_\-]*\.dll$', _re.IGNORECASE
)
print('CRT inventory (dist tree):')
for _dirpath, _dirnames, _filenames in _os.walk(_dist_root):
    for _fname in _filenames:
        if _vc_family.match(_fname):
            print('  ' + _os.path.relpath(
                _os.path.join(_dirpath, _fname), _dist_root))
