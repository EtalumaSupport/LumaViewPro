"""Clean-state camera-SDK initialization guards.

The IDS image-processing DLL initializes reliably only in a near-empty
process: on-machine loader probes passed in every import order and
environment variant in bare processes, while every in-app import (after
pylon/Kivy/numpy/cv2 were resident) failed its DLL initialization routine.
The defense is structural ordering: the IDS stack is imported at the
earliest point of startup, before any heavy import, and always
ipl-package-first (its __init__ registers the DLL directory the other
pieces resolve against). These scans keep that ordering from silently
regressing; the app modules import Kivy/hardware SDKs, so source-level
assertions are the testable surface.
"""

from __future__ import annotations

import ast
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding='utf-8')


def test_preload_exists_and_stages_are_ipl_first():
    source = _read('modules/app_environment.py')
    tree = ast.parse(source)
    preload = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == 'preload_camera_sdks'
        ),
        None,
    )
    assert preload is not None, 'app_environment must own the camera-SDK preload'
    segment = ast.get_source_segment(source, preload)
    ipl = re.search(r'import ids_peak_ipl(?!\w)', segment)
    core = re.search(r'from ids_peak import ids_peak(?!\w)', segment)
    assert ipl is not None and core is not None, (
        'preload must import both the ipl package and the core binding'
    )
    assert ipl.start() < core.start(), 'preload must import ids_peak_ipl before the core binding'


def test_startup_preloads_before_any_heavy_import():
    source = _read('lumaviewpro.py')
    call = source.find('preload_camera_sdks()')
    heavy = source.find('import matplotlib')
    assert call != -1, 'lumaviewpro.py must invoke the camera-SDK preload'
    assert heavy != -1
    assert call < heavy, (
        'the camera-SDK preload must run before matplotlib (the first heavy '
        'import); a later preload meets an already-crowded process'
    )


def test_ids_driver_imports_ipl_package_first():
    source = _read('drivers/idscamera.py')
    bare_ipl = re.search(r'^import ids_peak_ipl$', source, re.MULTILINE)
    core = re.search(r'^from ids_peak import ids_peak$', source, re.MULTILINE)
    assert bare_ipl is not None and core is not None
    assert bare_ipl.start() < core.start(), (
        'idscamera must import ids_peak_ipl before the core binding -- the '
        'ipl package __init__ registers the DLL directory the core and the '
        'extension bridge resolve against'
    )
