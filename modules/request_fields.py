# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The reads of a client's dictionary argument, refused as the request's when they fail.

A member that takes a dictionary reads its keys; a key the client left out,
or a value of the wrong kind, used to fail wherever the value was first
used -- a ``KeyError`` from a subscript, a ``TypeError`` from ``round()`` --
often after something was built or commanded. Each read here refuses with
``ArgumentRefusedError`` naming the key's path as the client wrote it
(``input_config['layer_configs']['BF']['gain_db']``), before the member acts.
A number is asked ``refuse_unless_finite_number``. A key the member does not
read is not judged: callers pass settings blocks that carry more.

A leaf, like ``finite_number``: it imports nothing of the project but the
refusal and that predicate.
"""

import math
import numbers
from collections.abc import Mapping

import numpy as np

from modules.exceptions import ArgumentRefusedError
from modules.finite_number import refuse_unless_finite_number

_KINDS = {
    'bool': ('True or False', lambda v: isinstance(v, (bool, np.bool_))),
    'str': ('text', lambda v: isinstance(v, str)),
    'dict': ('a dictionary', lambda v: isinstance(v, Mapping)),
    'list': ('a list', lambda v: isinstance(v, (list, tuple))),
    # The settings store's count rule: 2.0 is a whole number, 2.5 is not.
    'whole': (
        'a whole number',
        lambda v: (
            isinstance(v, numbers.Real)
            and not isinstance(v, (bool, np.bool_))
            and math.isfinite(v)
            and float(v).is_integer()
        ),
    ),
}


def key_path(argument: str, key: object) -> str:
    """The path of ``key`` inside ``argument``, as the client wrote it."""
    return f'{argument}[{key!r}]'


def required(mapping: Mapping, key: str, argument: str) -> object:
    """The value of ``key`` in ``mapping``, the dictionary named ``argument``.

    Raises:
        ArgumentRefusedError: ``'missing_key'``, naming the key's path.
    """
    if key not in mapping:
        raise ArgumentRefusedError('missing_key', argument=key_path(argument, key), value=None)
    return mapping[key]


def refuse_unless_kind(value: object, kind: str, argument: str, *, nullable: bool = False) -> None:
    """Refuse ``value`` unless it is of ``kind``: one of ``'number'``,
    ``'whole'``, ``'bool'``, ``'str'``, ``'dict'`` or ``'list'``. A
    ``'number'`` is a finite number; ``nullable`` admits None.

    Raises:
        ArgumentRefusedError: ``'not_a_number'`` for a ``'number'``,
            ``'wrong_kind'`` for the rest, naming ``argument``.
    """
    if value is None and nullable:
        return
    if kind == 'number':
        refuse_unless_finite_number(value, argument)
        return
    words, test = _KINDS[kind]
    if not test(value):
        raise ArgumentRefusedError('wrong_kind', argument=argument, value=value, kind=words)
