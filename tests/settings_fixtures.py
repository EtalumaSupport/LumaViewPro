# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Complete settings for a factory-built session.

A session factory configures the scope it builds from the settings it is
handed, and refuses a dict that cannot configure one (no ``frame``, no
``objective_id``). File-sourced settings are always complete -- the
pipeline validates them by name and merges the shipped template in --
but a test that hands the factory a hand-built dict has to complete it
the same way, and say which keys it means to override.

Ten test modules install a ``MagicMock`` as ``modules.settings_init`` at
import time. Importing that name here would return the mock (and the
merge would silently return the bare overrides) or, imported first,
would replace the mock for those modules. So the real module is loaded
from its file at call time and never registered in ``sys.modules``.
"""

import copy
import importlib.util
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TEMPLATE = _REPO_ROOT / 'data' / 'settings.json'


def _real_settings_init():
    spec = importlib.util.spec_from_file_location(
        'modules.settings_init', _REPO_ROOT / 'modules' / 'settings_init.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_settings(**overrides) -> dict:
    """The shipped template with ``overrides`` laid over it.

    Overrides win at every depth; keys the overrides lack come from the
    template. Turret slot keys are normalized to ints on both sides
    BEFORE the merge -- merged afterwards, the template's string keys
    would sit beside an int-keyed override and the later normalization
    would let the template's ``None`` win.
    """
    si = _real_settings_init()
    merged = copy.deepcopy(overrides)
    template = si.read_settings_json(str(_TEMPLATE), None)
    si._normalize_turret_slot_keys(merged)
    si._normalize_turret_slot_keys(template)
    si._deep_merge_defaults(merged, template)
    return merged


def complete_settings_without(*keys, **overrides) -> dict:
    """``complete_settings`` with the named top-level keys removed -- the
    shape a factory must refuse."""
    settings = complete_settings(**overrides)
    for key in keys:
        settings.pop(key, None)
    return settings
