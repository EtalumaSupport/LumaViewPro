# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""cProfile is gated on its own flag, decoupled from debug_mode, and writes
to logs/cprofile/.

debug_mode previously double-dutied: turning it on (e.g. to capture [PERF])
silently started a whole-app cProfile run that dumped into logs/profile/ --
the same directory the profile_trace CSVs use, mixing the two artifact sets.
cProfile now has its own opt-in cprofile_enabled flag and its own output dir.

lumaviewpro.py and profiling_utils.py are locked by source parse (the app
module cannot import headless); the settings default is checked directly.
"""

from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
LVP = (REPO / 'lumaviewpro.py').read_text()
PUTIL = (REPO / 'modules' / 'profiling_utils.py').read_text()
SETTINGS = json.loads((REPO / 'data' / 'settings.json').read_text())


def test_cprofile_gated_on_own_flag_not_debug_mode():
    assert "settings.get('cprofile_enabled'" in LVP
    # The cProfile start must no longer be triggered by debug_mode.
    assert 'cProfile enabled (cprofile_enabled=true)' in LVP
    assert 'cProfile enabled (debug_mode=true)' not in LVP


def test_cprofile_default_dir_is_cprofile_namespace():
    assert "'./logs/cprofile/" in PUTIL
    # The cProfile default path must not reuse the profile_trace CSV dir.
    assert "'./logs/profile/" not in PUTIL


def test_cprofile_enabled_seeded_off():
    assert SETTINGS.get('cprofile_enabled') is False
