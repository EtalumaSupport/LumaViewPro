# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Numpy's scalars made Python's, and a closed set's member its string, where a value enters the settings store, a protocol or a file.

A numpy scalar reads as a number everywhere until a JSON writer meets it:
``json`` refuses ``np.int64`` and ``np.float32`` (``np.float64`` passes only
because it subclasses ``float``), and a Kivy ``NumericProperty`` refuses
every one of them. Each owner that takes a caller's value into what it
stores converts it here, once, so nothing downstream needs an encoder.

A ``StrEnum`` member is made the plain string it holds: the settings
store plain data, and their kind check reads exact types
(``settings_paths._kind``), so a stored member would be judged by its
class name, not as the string it is. The API makes the member again where
it hands the value out.

Only numpy scalars and string-enum members change. Everything else stays
as it is: a ``timedelta`` the auto-gain settings carry, an array a caller
owns, a mapping's keys, NaN. Nothing is refused; the owner's own checks
decide what is admitted.
"""

import enum

import numpy as np


def python_values(value: object) -> object:
    """``value`` with every numpy scalar and ``StrEnum`` member in it, at any depth, made the Python value it holds.

    A list or mapping is converted in place and returned, so an owner whose
    store other holders share (the settings) keeps the one object. A numpy
    scalar is returned as its Python value and a ``StrEnum`` member as its
    ``str``; anything else, an ``np.ndarray``
    or a tuple among them, is returned as it is.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, enum.StrEnum):
        return str(value)
    if isinstance(value, list):
        for i, item in enumerate(value):
            converted = python_values(item)
            if converted is not item:
                value[i] = converted
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            converted = python_values(item)
            if converted is not item:
                value[key] = converted
        return value
    return value
