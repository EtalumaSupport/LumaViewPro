# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The plugin tutorial's hello-world plugin, and the tests it teaches, run as written.

``docs/plugin-tutorial.md`` is the plugin contract's one worked example: a
plugin author copies its package and its test file. Both are taken from the
document into a folder of their own, and the test file is run there by
pytest in a fresh interpreter, as an author runs it: its imports, its own
fixture and its tests are the tutorial's, with nothing from this checkout's
``tests`` on the path, so an example the contract no longer takes fails here,
not in an author's hands.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

_TUTORIAL = pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'plugin-tutorial.md'


def _block_after(text: str, marker: str) -> str:
    """The first fenced Python block after *marker* in the tutorial."""
    found = re.search(re.escape(marker) + r'.*?^```python\n(.*?)^```$', text, re.M | re.S)
    assert found, f'the tutorial has no Python block after {marker!r}'
    return found.group(1)


def test_the_tutorials_plugin_passes_the_tutorials_tests(tmp_path):
    text = _TUTORIAL.read_text(encoding='utf-8')
    package = tmp_path / 'hello_postproc'
    package.mkdir()
    (package / '__init__.py').write_text(
        _block_after(text, '### `hello_postproc/__init__.py`'), encoding='utf-8'
    )
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / 'test_hello_postproc.py').write_text(
        _block_after(text, 'Place this in `tests/test_hello_postproc.py`'), encoding='utf-8'
    )
    # The author's folder holds the plugin; lumascope comes from the
    # PYTHONPATH the suite exports, as the .pth gives it to an author.
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(tmp_path), os.environ['PYTHONPATH']])}
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '-p',
            'no:cacheprovider',
            '-v',
            '-rA',
            'tests/test_hello_postproc.py',
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    passed = sorted(re.findall(r'^PASSED tests/test_hello_postproc\.py::(\w+)', proc.stdout, re.M))
    assert passed == [
        'test_processor_writes_summary',
        'test_registers_in_post_processing_namespace',
        'test_unregister_does_not_raise',
    ], proc.stdout[-3000:]
