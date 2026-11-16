# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Basler scope states its pylon SDK version as the dotted version.

``ScopeCapabilities.camera_sdk_version`` is the camera driver's
``get_sdk_info()['version']``: the version of the SDK the camera runs on,
which the capabilities publish over REST and the support snapshot carries.
pypylon's ``GetPylonVersion()`` answers a list, so a driver that takes its
text publishes ``'[12, 5, 2, 1565]'`` where the SDK's version is
``12.5.2.1565``; the binding's version is ``pylon.__version__``, the
``pypylon`` package having none. The environment banner and the support
report state the same versions the driver publishes.

Run against the real pypylon in a child process: the claim is what the SDK
answers, which the test process's stub (``tests/pypylon_stub.py``) cannot
show. Skipped where pypylon is not installed.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import pytest

_CHILD = """
import json, sys
sys.path.insert(0, 'src')
try:
    from pypylon import pylon
except ImportError:
    print(json.dumps({'sdk': False})); sys.exit(0)
from _lumascope.drivers.pyloncamera import PylonCamera
from _lumascope.modules.app_environment import camera_sdk_probe
from _lumascope.modules.tech_support_report import _collect_system_info
print(json.dumps({'sdk': True, 'info': PylonCamera().get_sdk_info(),
                  'binding': pylon.__version__,
                  'banner': camera_sdk_probe()[0],
                  'report': _collect_system_info().get('pylon_version')}))
"""


def test_the_sdk_version_a_basler_camera_publishes_is_dotted():
    repo = pathlib.Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, '-c', _CHILD],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    said = json.loads(done.stdout.strip().splitlines()[-1])
    if not said['sdk']:
        pytest.skip('pypylon is not installed')

    info = said['info']
    assert re.fullmatch(r'\d+(\.\d+)+', info['version'] or ''), (
        f'the SDK version is not the dotted version: {info["version"]!r}'
    )
    assert info['pypylon_version'] == said['binding'], (
        f"the binding version is not pypylon's: {info['pypylon_version']!r}"
    )
    # The environment banner and the support report state the same two.
    assert said['banner'] == (
        f'pypylon binding: {said["binding"]} / Pylon SDK: {info["version"]}'
    ), f'the banner states other versions: {said["banner"]!r}'
    assert said['report'] == info['version'], (
        f'the support report states another SDK version: {said["report"]!r}'
    )
