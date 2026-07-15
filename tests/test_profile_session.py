# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for the profiling harness's py-spy resolution (tools/profiling/profile_session.py).

Regression: a bare ``py-spy`` invocation fails under sudo (needed to attach on
macOS) because sudo sanitizes PATH. The harness must resolve py-spy's absolute
path from the interpreter's own bin dir so the same command works elevated.
"""

import pytest

from tools.profiling.profile_session import _pyspy_path


class TestPyspyPath:
    def test_resolves_next_to_interpreter(self, tmp_path):
        # py-spy is installed alongside the interpreter; that path wins even when
        # PATH is empty (the sudo case).
        (tmp_path / 'py-spy').write_text('')
        assert _pyspy_path(interpreter_dir=tmp_path) == str(tmp_path / 'py-spy')

    def test_falls_back_to_path(self, tmp_path, monkeypatch):
        # Not next to the interpreter -> use PATH.
        monkeypatch.setattr(
            'tools.profiling.profile_session.shutil.which', lambda name: '/usr/local/bin/py-spy'
        )
        assert _pyspy_path(interpreter_dir=tmp_path) == '/usr/local/bin/py-spy'

    def test_raises_when_missing_everywhere(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            'tools.profiling.profile_session.shutil.which', lambda name: None
        )
        with pytest.raises(FileNotFoundError, match='py-spy not found'):
            _pyspy_path(interpreter_dir=tmp_path)
