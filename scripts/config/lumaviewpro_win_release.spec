# -*- mode: python ; coding: utf-8 -*-

from kivy_deps import sdl2, glew

block_cipher = None
app_name = 'lumaviewpro'

a = Analysis(
    ['lumaviewpro.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('.','.')
    ],
    hiddenimports=[
        'plyer.platforms.win.notification',
        'plyer.platforms.win.filechooser',
        'skimage.measure',
        'win32timezone',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

splash = Splash(
    'data/icons/etaluma_splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(10,50),
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
