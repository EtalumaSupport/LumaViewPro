# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An outcome's type publishes only fields its instances hold, under names a problem does not use.

A REST problem carries each field an exception's type publishes
(``@api_fields``) beside its own members. A published name the type's
``__init__`` never sets would answer a refusal with a fault, and one that
is also a problem member would overwrite it.
"""

from __future__ import annotations

import ast
import inspect

import modules.exceptions
from modules.api_surface import fields_of
from rest import problems

# The members every problem has, which no published field may replace.
PROBLEM_MEMBERS = frozenset(
    problems._problem(
        type_='',
        title='',
        detail='',
        status=0,
        request_id='',
        kind=problems.OutcomeKind.REFUSAL,
        reason=None,
        remedy=None,
    )
)


def _published_exceptions() -> list[type[BaseException]]:
    return [
        cls
        for _, cls in inspect.getmembers(modules.exceptions, inspect.isclass)
        if issubclass(cls, BaseException) and fields_of(cls)
    ]


def _set_in_init(cls: type) -> set[str]:
    """The attributes ``self.<name> = ...`` assigns in the ``__init__`` of *cls* or a base."""
    found: set[str] = set()
    for klass in cls.__mro__:
        init = vars(klass).get('__init__')
        if init is None or not hasattr(init, '__code__'):
            continue
        tree = ast.parse(inspect.cleandoc('\n' + inspect.getsource(init)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name)
                and node.value.id == 'self'
            ):
                found.add(node.attr)
    return found


def test_some_outcome_publishes_its_fields():
    assert modules.exceptions.ArgumentRefusedError in _published_exceptions()


def test_every_published_field_is_set_by_its_init_and_is_no_problem_member():
    for cls in _published_exceptions():
        published = set(fields_of(cls))
        assert published <= _set_in_init(cls), cls.__name__
        assert not published & PROBLEM_MEMBERS, cls.__name__
