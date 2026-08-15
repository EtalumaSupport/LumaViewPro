# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""cProfile is gated on its own flag, decoupled from debug_mode, and writes
to logs/cprofile/.

debug_mode previously double-dutied: turning it on (e.g. to capture [PERF])
silently started a whole-app cProfile run that dumped into logs/profile/ --
the same directory the profile_trace CSVs use, mixing the two artifact sets.
cProfile now has its own opt-in cprofile_enabled flag and its own output dir.

lumaviewpro.py is locked by source parse (the app module cannot import
headless); profiling_utils.py and the settings default are checked directly.
"""

from __future__ import annotations

import json
import pathlib

# pin-justified (settings read): the shipped default in data/settings.json
# is the contract a fresh install receives.
REPO = pathlib.Path(__file__).resolve().parent.parent
LVP = (REPO / 'lumaviewpro.py').read_text()
SETTINGS = json.loads((REPO / 'data' / 'settings.json').read_text())


def test_cprofile_gated_on_own_flag_not_debug_mode():
    assert "settings.get('cprofile_enabled'" in LVP
    # The cProfile start must no longer be triggered by debug_mode.
    assert 'cProfile enabled (cprofile_enabled=true)' in LVP
    assert 'cProfile enabled (debug_mode=true)' not in LVP


def test_cprofile_default_dir_is_cprofile_namespace(tmp_path, monkeypatch):
    # Constructing ProfilingHelper with no save_path must land artifacts in
    # logs/cprofile/, NOT logs/profile/ (the profile_trace CSV namespace).
    #
    # Anchored on the data directory rather than the working directory: the
    # default used to be CWD-relative, which an installed build cannot write
    # to. The namespace separation is what this test is for; where the root
    # comes from is not.
    import lvp_logger

    monkeypatch.setattr(lvp_logger, 'lvp_appdata', str(tmp_path))
    from modules.profiling_utils import ProfilingHelper

    helper = ProfilingHelper()
    artifact_dir = helper._profile_artifact_path
    assert artifact_dir.parent == (tmp_path / 'logs' / 'cprofile').resolve()
    # 'cprofile' as its own path part -- not the profile_trace CSV dir.
    assert 'profile' not in artifact_dir.parts
    assert artifact_dir.is_dir()  # constructor created the namespace


def test_cprofile_enabled_seeded_off():
    assert SETTINGS.get('cprofile_enabled') is False
