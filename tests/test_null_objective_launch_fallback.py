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


def test_an_empty_id_resolves_to_no_objective_not_the_first_one(loader):
    """An empty id used to answer with the smallest lens in the catalogue.

    The lookup falls back to prefix matching when an id is not an exact hit,
    and the empty string is a prefix of every key -- so `''` matched whatever
    happens to be first in objectives.json and returned it as a confident
    answer. On the shipped catalogue that is 1.25x Oly, a real objective with a
    real focal length, so the caller got a plausible lens rather than a
    refusal, and every scale derived from it was wrong by the ratio of the two
    magnifications.

    An empty string is an unknown-but-valid string, so it answers the way every
    other unresolvable string already does.
    """
    assert loader.get_objective_info(objective_id='') is None


def test_a_whitespace_id_resolves_to_no_objective(loader):
    """The same hole with the same shape: a whitespace-only id is not an
    identifier either, and no catalogue key begins with a space, so before the
    prefix guard it fell through to the error return by accident rather than by
    contract. Pinned so the emptiness test is about identifiers, not about the
    particular string ''."""
    assert loader.get_objective_info(objective_id='   ') is None


def test_a_real_partial_id_still_resolves(loader):
    """The prefix fallback is deliberate and stays: a genuine partial id
    resolves to the objective it names. The fix narrows what counts as a
    prefix, not whether prefixes work."""
    info = loader.get_objective_info(objective_id='4x Oly')
    assert info is not None
    assert info['short_name'] == '4xOly'
