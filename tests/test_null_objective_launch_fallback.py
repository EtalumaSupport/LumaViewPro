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


def test_an_unknown_string_is_refused_by_name(loader):
    """An id the catalogue does not hold is refused with the same type as null.

    It used to answer None. Of the twelve production readers of this lookup,
    nine subscript the answer, so the None reached the user as a bare
    "'NoneType' object is not subscriptable" naming nothing they could act on;
    the three that guarded it did so three different ways. One refusal, one
    type, and the launch path already recovers from it.
    """
    with pytest.raises(ConfigError, match="unknown objective 'zzz-no-such-objective'"):
        loader.get_objective_info(objective_id='zzz-no-such-objective')


@pytest.mark.parametrize('not_an_identifier', ['', '   '])
def test_an_empty_or_whitespace_id_is_refused_not_matched(loader, not_an_identifier):
    """An empty id used to answer with the smallest lens in the catalogue.

    The lookup fell back to prefix matching when an id was not an exact hit,
    and the empty string is a prefix of every key -- so `''` matched whatever
    happens to be first in objectives.json and returned it as a confident
    answer. On the shipped catalogue that is 1.25x Oly, a real objective with a
    real focal length, so the caller got a plausible lens rather than a
    refusal, and every scale derived from it was wrong by the ratio of the two
    magnifications. Neither is an identifier, so both are refused.
    """
    with pytest.raises(ConfigError):
        loader.get_objective_info(objective_id=not_an_identifier)


def test_a_partial_id_is_refused_not_guessed(loader):
    """The prefix fallback is gone: a near miss names two lenses, not one.

    The shipped catalogue holds both '10x Oly' and '10x Phase', so an id of
    '10x' bound to whichever came first in the file and answered with its focal
    length as if the match were exact. A partial id is refused so the file
    that names it gets corrected; the exact key still resolves.
    """
    assert '10x Oly' in loader.get_objectives_list()
    assert '10x Phase' in loader.get_objectives_list()
    with pytest.raises(ConfigError, match="unknown objective '10x'"):
        loader.get_objective_info(objective_id='10x')
    assert loader.get_objective_info(objective_id='10x Oly')['short_name'] == '10xOly'


def test_the_catalogue_key_is_the_only_way_to_name_an_objective(loader):
    """The lookup takes one identifier, the catalogue key.

    It used to take a short name as well, resolved through a reverse lookup
    that no production code called: a second way to name an objective with
    no consumer, kept only by habit. The short name is a filename token
    derived from the key, not an identity (Eric, 2026-09-21: one way of
    doing things).
    """
    import inspect

    parameters = list(inspect.signature(loader.get_objective_info).parameters)
    assert parameters == ['objective_id']
    assert not hasattr(loader, 'find_objective_id_from_short_name')
    assert loader.get_objective_info(objective_id='4x Oly')['short_name'] == '4xOly'
