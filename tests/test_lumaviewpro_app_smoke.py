# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""LVP-A-19 / Issue 'startup NameError' -- App-build undefined-name regression test.

Catches the class of failure that broke production startup on
2026-05-04: a `lumaviewpro.py` mechanical refactor (LVP-A-3 partial,
commit `0f52812`) deleted module-level globals that were referenced
inside `LumaViewProApp.build()` but never assigned there. Production
hit `NameError: name 'live_histo_setting' is not defined` on first
launch.

Smoke tests ALL passed (1713 / 1713) because every test constructs
``Lumascope`` directly and never runs ``App.build()``. This test
fills that gap with a lightweight static analysis: pyflakes, which
flags undefined-name references at the function-scope level.

Why pyflakes instead of running the actual ``App.build()``:

- pyflakes is fast (<100 ms on a 1k-line file) and pure-Python.
- ``LumaViewProApp().build()`` requires a working Kivy widget tree,
  which is heavy to mock in headless CI. (Tried; `kivy.uix.*` import
  paths multiply faster than the test value justifies.)
- The bug pattern this guards against IS exactly what pyflakes is
  good at: undefined names in nested scopes. Verified against the
  pre-fix broken state (commit `0f52812`): pyflakes reported
  ``undefined name 'live_histo_setting'`` at line 816 -- the exact
  failure that hit production.

If a future bug needs deeper coverage (KeyError on dict access,
import-side-effect failures, etc.) layer a second test on top of
this one. For now, undefined-name protection is sufficient because
that's the failure pattern this test was created to prevent.
"""

import os
import shutil
import subprocess
import sys

import pytest


# Files to scan for undefined names. Currently just lumaviewpro.py
# because that's where the production startup-fail occurred. Add more
# entry-point-style files here as further mechanical refactors land
# in code paths not covered by direct unit tests.
_SCAN_FILES = [
    'lumaviewpro.py',
]


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyflakes_available():
    """Return True if pyflakes is importable in this environment."""
    try:
        import pyflakes  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _pyflakes_available(), reason='pyflakes not installed; install with `pip install pyflakes`'
)
@pytest.mark.parametrize('relpath', _SCAN_FILES)
def test_no_undefined_names(relpath):
    """Fail if pyflakes finds any ``undefined name`` reference.

    Other pyflakes warnings (unused imports, redefined names, etc.)
    are tolerated -- they don't break runtime. The check is narrow on
    purpose: the production failure mode this test guards against is
    exactly the undefined-name class.
    """
    target = os.path.join(_project_root(), relpath)
    assert os.path.exists(target), f'target file missing: {target}'

    # Run pyflakes as a subprocess so its output is exactly what the
    # operator would see when running it manually. Capture stdout
    # (warnings) AND stderr (pyflakes itself); pyflakes exits non-zero
    # when warnings exist, but we only care about the undefined-name
    # subset.
    proc = subprocess.run(
        [sys.executable, '-m', 'pyflakes', target],
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr

    undefined_lines = [
        line
        for line in output.splitlines()
        if 'undefined name' in line or 'may be undefined' in line
    ]
    assert not undefined_lines, (
        f'pyflakes found undefined names in {relpath}:\n'
        + '\n'.join(f'  {line}' for line in undefined_lines)
        + '\n\nThis catches the exact class of bug that broke '
        'production startup on 2026-05-04 (LVP-A-3 partial). A '
        'module-level identifier was deleted on the assumption it '
        'was build-only, but the AppContext kwarg pass-through '
        'still referenced it -- and no smoke test ran App.build() '
        'to catch the resulting NameError.'
    )
