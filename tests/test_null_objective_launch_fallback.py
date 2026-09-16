# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A stored null objective must fall back to the template, not fail launch.

`lumaviewpro` composes a session inside `except ConfigError` and answers that
error by republishing the shipped template. That recovery is the designed
response to "stored settings cannot configure a scope", and it only fires for
`ConfigError`: the handler one level out logs and re-raises, and nothing above
`LumaViewProApp().run()` catches, so an untyped escape from the objective
lookup fails app launch outright.

`objective_id` is a legitimate JSON null -- the settings shape gate documents
that "null is a legitimate 'unset'", no load-time repair replaces it, and the
key-presence checks at `_validate_settings` and `ScopeInitConfig.from_settings`
both pass it through. So the null reaches `ObjectiveLoader.get_objective_info`,
where it collides with the `None` that means "argument not supplied".

These use the real loader over the shipped catalogue rather than a mock: the
defect IS the loader's choice of exception type, and a mock would assert the
choice instead of exercising it.
"""

import pytest

from modules.exceptions import ConfigError
from modules.objectives_loader import ObjectiveLoader


@pytest.fixture
def loader():
    return ObjectiveLoader()


def test_null_objective_id_raises_config_error_not_bare_exception(loader):
    """The regression: a stored null must be catchable at the launch boundary."""
    with pytest.raises(ConfigError):
        loader.get_objective_info(objective_id=None)


@pytest.mark.parametrize('unusable', [4, {'a': 1}, 0, False, 12.5])
def test_non_string_objective_ids_keep_raising_config_error(loader, unusable):
    """These already work; this pins them against a fix that would break them.

    An earlier revision of this fix proposed returning None for anything that
    is "not a string", which would have converted these working typed failures
    into a silent None landing on callers that dereference it -- `protocol.py`
    at its focal_length read, `autofocus_runner` in `_calculate_params`, and
    `config_helpers.get_current_objective_info`, which propagates its result to
    thirteen sites. The refusal is the contract.
    """
    with pytest.raises(ConfigError):
        loader.get_objective_info(objective_id=unusable)


def test_unknown_but_valid_string_still_returns_none(loader):
    """The OTHER contract, unchanged: an unknown id resolves to no objective.

    `RuntimeState.set_objective` turns this None into its documented
    ConfigError. Converting this return into a raise would be a different
    change with a different blast radius, so it is pinned here as-is.
    """
    assert loader.get_objective_info(objective_id='zzz-no-such-objective') is None
