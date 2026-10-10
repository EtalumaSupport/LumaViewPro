# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The ``@api`` door: an argument is refused unless it is of the type its member declares.

``api_surface.api`` wraps every marked member that takes an argument, and
the wrapper asks this module, at the member's first call, for the member's
door: its annotations resolved over every class and ``type`` alias under
``modules/``, each turned into one check. A value of the declared type
passes; a number is taken by what it means, so an ``int`` passes where a
``float`` is declared and a numpy scalar passes as the Python number it
holds, and either reaches the body converted. A ``bool`` is never a
number. Anything else is ``ArgumentTypeRefusedError``, before the body runs.

The door judges the type and nothing more: NaN is a float, so finiteness
stays ``finite_number``'s; a name's membership and a dictionary's
contents stay the member's own.

Imported by the wrapper at a member's first call, not at import: resolving
an annotation needs every class under ``modules/`` loaded.
"""

from __future__ import annotations

import datetime
import inspect
import os
import types
import typing
from collections.abc import Callable, Iterable, Mapping

import numpy as np

from modules import wire_encoding
from modules.exceptions import ArgumentTypeRefusedError

# What a check answers for a value its type does not admit. A value of its
# own, since None is one an argument can be.
_REFUSED = object()

type _Check = Callable[[object], object]
type _Door = Callable[[tuple, dict], tuple[tuple, dict]]

_namespace: dict[str, object] | None = None


def _names() -> dict[str, object]:
    """The names an annotation may use: every project class and alias, and the outside ones.

    ``SimulatedStall`` is the one class from ``drivers/`` an annotation
    names; ``datetime`` and ``np`` are named by module.
    """
    global _namespace
    if _namespace is None:
        from drivers.simulated_camera import SimulatedStall

        _namespace = {
            **wire_encoding.project_classes(),
            **wire_encoding.project_type_aliases(),
            'SimulatedStall': SimulatedStall,
            'datetime': datetime,
            'np': np,
        }
    return _namespace


def _float(value: object) -> object:
    if type(value) is float:
        return value
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    return _REFUSED


def _int(value: object) -> object:
    if type(value) is int:
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return _REFUSED


def _bool(value: object) -> object:
    if type(value) is bool:
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    return _REFUSED


def _str(value: object) -> object:
    if type(value) is str:
        return value
    if isinstance(value, str):
        return str(value)
    return _REFUSED


def _none(value: object) -> object:
    return value if value is None else _REFUSED


def _callable(value: object) -> object:
    return value if callable(value) else _REFUSED


def _instance_of(cls: type) -> _Check:
    def check(value: object) -> object:
        return value if isinstance(value, cls) else _REFUSED

    return check


def _any_of(arms: list[_Check]) -> _Check:
    # The first arm that admits the value converts it, so the arms keep the
    # annotation's order: in ``int | float`` an int stays an int.
    def check(value: object) -> object:
        for arm in arms:
            admitted = arm(value)
            if admitted is not _REFUSED:
                return admitted
        return _REFUSED

    return check


_SCALARS: dict[object, _Check] = {
    float: _float,
    int: _int,
    bool: _bool,
    str: _str,
    type(None): _none,
}
# A container is admitted by its kind; what it holds is its member's.
_KINDS: dict[object, type] = {
    dict: Mapping,
    Mapping: Mapping,
    list: list,
    tuple: tuple,
    Iterable: Iterable,
    os.PathLike: os.PathLike,
}


def _check(hint: object) -> _Check:
    """The check for one resolved annotation.

    Raises:
        TypeError: the annotation is a form no check is written for. The
            guard over every member's door makes this a build failure, never
            a call's.
    """
    if isinstance(hint, typing.TypeAliasType):
        return _check(hint.__value__)
    if hint in _SCALARS:
        return _SCALARS[hint]
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        return _any_of([_check(arm) for arm in typing.get_args(hint)])
    if hint is Callable or origin is Callable:
        return _callable
    kind = _KINDS.get(origin if origin is not None else hint)
    if kind is not None:
        return _instance_of(kind)
    if origin is None and isinstance(hint, type):
        return _instance_of(hint)
    raise TypeError(f'the @api door has no check for the annotation {hint!r}')


def _declared(hint: object) -> str:
    """The annotation as a caller reads it: an alias by its name."""
    if isinstance(hint, typing.TypeAliasType):
        return hint.__name__
    if typing.get_origin(hint) in (types.UnionType, typing.Union):
        return ' | '.join(_declared(arm) for arm in typing.get_args(hint))
    if hint is type(None):
        return 'None'
    if isinstance(hint, type) and typing.get_origin(hint) is None:
        return hint.__name__
    return repr(hint).replace('typing.', '').replace('collections.abc.', '')


def door_for(function: Callable, *, bound: bool) -> _Door:
    """The door of one member: what admits, converts or refuses each argument of a call.

    ``bound`` says the first parameter is ``self`` or ``cls``, which is not
    checked.

    Raises:
        TypeError: an annotation is a form no check is written for.
        NameError: an annotation names something no module under
            ``modules/`` defines.
    """
    hints = typing.get_type_hints(function, localns=_names())
    member = function.__qualname__
    parameters = list(inspect.signature(function).parameters.values())
    positional: list[tuple[str, _Check, str] | None] = []
    by_name: dict[str, tuple[str, _Check, str]] = {}
    for index, parameter in enumerate(parameters):
        if bound and index == 0:
            positional.append(None)
            continue
        hint = hints[parameter.name]
        entry = (parameter.name, _check(hint), _declared(hint))
        by_name[parameter.name] = entry
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            positional.append(entry)

    def admitted(entry: tuple[str, _Check, str], value: object) -> object:
        name, check, declared = entry
        converted = check(value)
        if converted is _REFUSED:
            raise ArgumentTypeRefusedError(
                member=member, argument=name, declared=declared, value=value
            )
        return converted

    def admit(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        # More positional arguments than parameters, or a keyword the
        # member lacks, is left for the call itself to refuse as Python does.
        if len(args) > (1 if bound else 0):
            args = tuple(
                admitted(positional[i], value)
                if i < len(positional) and positional[i] is not None
                else value
                for i, value in enumerate(args)
            )
        for name, value in kwargs.items():
            entry = by_name.get(name)
            if entry is not None:
                kwargs[name] = admitted(entry, value)
        return args, kwargs

    return admit
