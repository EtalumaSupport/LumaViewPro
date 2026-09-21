# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The objective-metadata boundary refuses by name, and declares it.

`ObjectiveLoader.get_objective_info` refuses an id the catalogue does not hold
with a `ConfigError`, the type the launch path already recovers from. The API
method in front of it forwards the answer unchanged, so its signature promises
the dict it returns and its docstring names the refusal -- that pair is the
contract REST and the SDK inherit. An L2 caller reading `-> dict` may
subscript without guarding, because there is no None to guard against.

An annotation is not executable, so the signature is read statically; the
behavioural half, that the refusal crosses the boundary untouched, is
asserted beside it.
"""

import inspect
import typing

import pytest

from modules.exceptions import ConfigError
from modules.lumascope_api.runtime_state import RuntimeState
from modules.objectives_loader import ObjectiveLoader


def _return_annotation(method):
    return typing.get_type_hints(method).get('return')


def test_the_boundary_promises_a_dict_and_nothing_else():
    """The signature admits no None, because the call cannot produce one."""
    assert _return_annotation(RuntimeState.get_objective_info) is dict


def test_the_loader_it_forwards_promises_the_same():
    """The premise above, asserted rather than assumed: the two move together."""
    assert _return_annotation(ObjectiveLoader.get_objective_info) is dict


def test_an_unknown_objective_id_is_refused_through_the_boundary():
    """The refusal crosses the boundary as itself, not wrapped or swallowed."""
    api = RuntimeState.__new__(RuntimeState)
    api._objectives_loader = ObjectiveLoader()
    with pytest.raises(ConfigError, match="unknown objective 'not-a-real-objective'"):
        api.get_objective_info(objective_id='not-a-real-objective')


def test_selecting_an_unknown_objective_is_refused_and_leaves_state_untouched():
    """The selection member relied on the loader answering None and refused on
    its behalf; with the loader refusing, that second check was a duplicate and
    is gone. What it protected still holds: a bad id never tears the pair."""
    api = RuntimeState.__new__(RuntimeState)
    api._objectives_loader = ObjectiveLoader()
    api._objective_id = '4x Oly'
    api._objective = api._objectives_loader.get_objective_info(objective_id='4x Oly')
    with pytest.raises(ConfigError):
        api.set_objective('not-a-real-objective')
    assert api._objective_id == '4x Oly'
    assert api._objective['short_name'] == '4xOly'


def test_the_docstring_names_the_refusal():
    """A docstring silent about the raise is the same gap in prose."""
    doc = inspect.getdoc(RuntimeState.get_objective_info) or ''
    assert 'ConfigError' in doc
    assert 'None' not in doc.split('Raises:')[0], 'the Returns section must not promise a None'
