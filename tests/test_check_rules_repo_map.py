"""Tests for tools/check_rules.py repo-map dispatch.

The unified check_rules.py is byte-identical in the LumaViewPro and
Firmware repos; _REPO_CHECK_MAP preserves each repo's pre-unification
behavior exactly. These tests pin that contract: the family split per
repo key, the live resolution in THIS repo, and the loud failure when
the repo identity cannot be determined (a gate that cannot know which
checks apply must not silently pass).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.check_rules import _REPO_CHECK_MAP, _repo_config


class TestRepoCheckMap:
    def test_exactly_two_repo_keys(self):
        assert sorted(_REPO_CHECK_MAP) == ['etaluma-firmware', 'lumaviewpro']

    def test_lvp_families_match_pre_unification_behavior(self):
        assert _REPO_CHECK_MAP['lumaviewpro'] == {
            'kv_ascii': True,
            'cv2_channel': True,
            'doc_status': False,
        }

    def test_firmware_families_match_pre_unification_behavior(self):
        assert _REPO_CHECK_MAP['etaluma-firmware'] == {
            'kv_ascii': False,
            'cv2_channel': False,
            'doc_status': True,
        }

    def test_live_resolution_in_this_repo(self, monkeypatch):
        repo_root = Path(__file__).resolve().parent.parent
        monkeypatch.chdir(repo_root)
        key = repo_root.name
        config = _repo_config()
        # Which repo this test runs in depends on which working tree the
        # file sits in (the test file itself is LVP-only today, but the
        # contract is written repo-agnostically on purpose).
        assert config in _REPO_CHECK_MAP.values(), (key, config)

    def test_missing_key_fails_loud(self, tmp_path, monkeypatch):
        subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True, capture_output=True)
        (tmp_path / 'pyproject.toml').write_text('[tool.other]\nx = 1\n')
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            _repo_config()
        assert exc.value.code == 2

    def test_missing_pyproject_fails_loud(self, tmp_path, monkeypatch):
        subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True, capture_output=True)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            _repo_config()
        assert exc.value.code == 2
