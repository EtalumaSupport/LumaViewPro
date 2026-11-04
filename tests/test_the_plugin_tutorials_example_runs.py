# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The plugin tutorial's hello-world plugin, and the tests it teaches, run as written.

``docs/plugin-tutorial.md`` is the plugin contract's one worked example: a
plugin author copies its package and its test file. Both are taken from the
document here, the package imported as a plugin author would install it, and
each of the tutorial's tests called with the fixtures it asks for, so an
example the contract no longer takes fails here, not in an author's hands.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import re
import sys

from tests.plugin_test_harness import _make_ctx

_TUTORIAL = pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'plugin-tutorial.md'


def _block_after(text: str, marker: str) -> str:
    """The first fenced Python block after *marker* in the tutorial."""
    found = re.search(re.escape(marker) + r'.*?^```python\n(.*?)^```$', text, re.M | re.S)
    assert found, f'the tutorial has no Python block after {marker!r}'
    return found.group(1)


def test_the_tutorials_plugin_passes_the_tutorials_tests(tmp_path, monkeypatch):
    text = _TUTORIAL.read_text(encoding='utf-8')
    package = tmp_path / 'src' / 'hello_postproc'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text(
        _block_after(text, '### `hello_postproc/__init__.py`'), encoding='utf-8'
    )
    (tmp_path / 'src' / 'test_hello_postproc.py').write_text(
        _block_after(text, 'Place this in `tests/test_hello_postproc.py`'), encoding='utf-8'
    )
    monkeypatch.syspath_prepend(str(tmp_path / 'src'))
    for name in ('hello_postproc', 'test_hello_postproc'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    tests = importlib.import_module('test_hello_postproc')

    ran = []
    for name, test in inspect.getmembers(tests, inspect.isfunction):
        if not name.startswith('test_'):
            continue
        fixtures = {'harness_ctx': _make_ctx(), 'tmp_path': tmp_path / name}
        (tmp_path / name).mkdir()
        test(**{arg: fixtures[arg] for arg in inspect.signature(test).parameters})
        ran.append(name)

    assert ran == [
        'test_processor_writes_summary',
        'test_registers_in_post_processing_namespace',
        'test_unregister_does_not_raise',
    ]
