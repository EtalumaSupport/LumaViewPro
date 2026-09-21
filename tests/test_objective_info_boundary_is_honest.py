# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The objective-metadata boundary declares the None it forwards.

`ObjectiveLoader.get_objective_info` answers None for an id the catalogue
does not hold, deliberately: an untyped raise there escapes the launch path's
recovery and takes app start down. The API method in front of it forwards that
value unchanged, so it has to declare it -- an L2 or REST caller reading a
`-> dict` promise has no reason to guard, and gets an attribute error on a
dict that is None.

The annotation is the contract REST and the SDK inherit, so it is asserted
here rather than left to review. An annotation is not executable, so this is
a static read of the signature: the behavioural half (that None is what comes
back) is asserted beside it.
"""

import inspect
import typing

from modules.lumascope_api.runtime_state import RuntimeState


def _return_annotation(method):
    return typing.get_type_hints(method).get('return')


def test_the_boundary_declares_the_none_it_can_return():
    """The signature admits None, because the call can produce it."""
    annotation = _return_annotation(RuntimeState.get_objective_info)
    args = typing.get_args(annotation)
    assert type(None) in args, (
        f'get_objective_info is annotated {annotation!r}, which forbids None, '
        'but it forwards ObjectiveLoader.get_objective_info unchanged and that '
        'answers None for an unknown or null objective id. A caller reading this '
        'signature has no reason to guard.'
    )


def test_the_loader_it_forwards_still_answers_none():
    """The premise above, asserted rather than assumed.

    If the loader is ever changed to raise, this fails and the annotation
    should be narrowed back -- the two move together.
    """
    from modules.objectives_loader import ObjectiveLoader

    annotation = _return_annotation(ObjectiveLoader.get_objective_info)
    assert type(None) in typing.get_args(annotation), (
        'The loader no longer declares None; get_objective_info above was '
        'widened to match it, so narrow both or neither.'
    )


def test_an_unknown_objective_id_answers_none_through_the_boundary():
    """The behaviour the annotation now describes."""

    class _Loader:
        def get_objective_info(self, objective_id=None, short_name=None):
            return None

    api = RuntimeState.__new__(RuntimeState)
    api._objectives_loader = _Loader()
    assert api.get_objective_info(objective_id='not-a-real-objective') is None


def test_the_signature_and_the_docstring_agree():
    """A docstring promising a dict is the same lie in prose."""
    doc = inspect.getdoc(RuntimeState.get_objective_info) or ''
    assert 'None' in doc, 'the docstring must say when None comes back'
