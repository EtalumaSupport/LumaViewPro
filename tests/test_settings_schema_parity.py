# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Schema parity between data/settings.json and data/current.json.

settings.json holds user defaults + overrides; current.json is the derived
runtime state the app writes back at clean shutdown. The two must carry the
same schema (same top-level keys, same per-layer keys) except for keys that
exist only at runtime -- those are enumerated explicitly below, each with the
reason it is runtime-only.

current.json is gitignored and machine-local, so only the checks against the
committed settings.json run unconditionally. The full two-file parity checks
run where a current.json exists (any machine the app has run on) and skip
elsewhere (fresh clone, CI); they compare only in the direction that is valid
for an on-disk snapshot of any age -- keys the runtime persisted that have no
default and no exemption. The other direction (a settings.json key absent
from current.json) is expected of a stale snapshot: the loader merges new
defaults into current.json in memory, and the file catches up on the next
clean shutdown, so asserting it against the on-disk file would fail spuriously.
"""

import json
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SETTINGS_PATH = REPO_ROOT / 'data' / 'settings.json'
CURRENT_PATH = REPO_ROOT / 'data' / 'current.json'

# Top-level keys that exist ONLY in current.json, on purpose:
# - turret_position: the live turret position the app records as it moves;
#   derived runtime state with no meaningful user default.
RUNTIME_ONLY_TOP_LEVEL = {'turret_position'}

# Per-layer (nested dict) keys that exist ONLY in current.json, on purpose,
# keyed by the top-level entry they live under:
# - frame.native_width / frame.native_height: the connected camera's native
#   sensor dimensions, discovered from hardware at runtime; settings.json
#   cannot know them ahead of the camera.
RUNTIME_ONLY_PER_LAYER = {
    'frame': {'native_width', 'native_height'},
}

requires_current_json = pytest.mark.skipif(
    not CURRENT_PATH.exists(),
    reason='data/current.json is runtime-written and gitignored; parity '
    'against it can only run on a machine the app has run on',
)


def _load_settings():
    return json.loads(SETTINGS_PATH.read_text())


def _load_current():
    return json.loads(CURRENT_PATH.read_text())


def test_settings_defaults_carry_no_runtime_only_keys():
    """The committed defaults file must stay free of runtime state.

    Deterministic (reads only the committed settings.json). If a key on the
    runtime-only lists gains a default in settings.json, the exemption is no
    longer honest -- either the key stopped being runtime-only (retire the
    exemption) or runtime state leaked into the defaults file (remove the key).
    """
    settings = _load_settings()
    leaked_top = RUNTIME_ONLY_TOP_LEVEL & set(settings)
    assert not leaked_top, (
        f'runtime-only keys leaked into settings.json: {sorted(leaked_top)} -- '
        'remove them from the defaults file or retire their '
        'RUNTIME_ONLY_TOP_LEVEL exemption'
    )
    leaked_layer = [
        f'{key}.{sub}'
        for key, subkeys in RUNTIME_ONLY_PER_LAYER.items()
        if isinstance(settings.get(key), dict)
        for sub in subkeys
        if sub in settings[key]
    ]
    assert not leaked_layer, (
        f'runtime-only per-layer keys leaked into settings.json: {leaked_layer} -- '
        'remove them from the defaults file or retire their '
        'RUNTIME_ONLY_PER_LAYER exemption'
    )


@requires_current_json
def test_top_level_key_parity():
    settings, current = _load_settings(), _load_current()
    current_only = set(current) - set(settings) - RUNTIME_ONLY_TOP_LEVEL
    assert not current_only, (
        f'current.json has top-level keys with no settings.json default and '
        f'no runtime-only exemption: {sorted(current_only)} -- add a default '
        'or, if genuinely runtime-only, add the key to RUNTIME_ONLY_TOP_LEVEL '
        'with the reason'
    )


@requires_current_json
def test_per_layer_key_parity():
    settings, current = _load_settings(), _load_current()
    drift = []
    for key in set(settings) & set(current):
        s_val, c_val = settings[key], current[key]
        if not (isinstance(s_val, dict) and isinstance(c_val, dict)):
            if isinstance(s_val, dict) != isinstance(c_val, dict):
                drift.append(f'{key}: dict in one file, {type(c_val).__name__} in the other')
            continue
        allowed = RUNTIME_ONLY_PER_LAYER.get(key, set())
        current_only = set(c_val) - set(s_val) - allowed
        if current_only:
            drift.append(f'{key}: current-only sub-keys {sorted(current_only)}')
    assert not drift, (
        'per-layer schema drift: the runtime persisted keys with no '
        f'settings.json default and no RUNTIME_ONLY_PER_LAYER exemption: {drift}'
    )
