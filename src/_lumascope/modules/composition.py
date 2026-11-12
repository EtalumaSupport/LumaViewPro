# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The one way a session and its scope are made: ``ScopeSession.create``.

A leaf module, importing nothing of the project's, so the session and the
scope both check the argument that only ``create`` holds without importing
each other.
"""


class _Composition:
    """The argument ``ScopeSession.create`` hands the constructors it calls."""

    def __repr__(self) -> str:
        return 'COMPOSED_BY_CREATE'


#: Held by ``ScopeSession.create`` alone. A ``ScopeSession`` or ``Lumascope``
#: built without it would skip what ``create`` does around the
#: construction -- the settings held to their ranges, the executor topology,
#: the scope's configuration, the camera's start -- and fail later, far from
#: the call that made it.
COMPOSED_BY_CREATE = _Composition()


def refuse_unless_composed(composition: object, made: str) -> None:
    """Refuse a construction of *made* that ``ScopeSession.create`` did not ask for.

    Raises:
        TypeError: *composition* is not ``COMPOSED_BY_CREATE``; the words
            name ``ScopeSession.create`` as the way in.
    """
    if composition is not COMPOSED_BY_CREATE:
        raise TypeError(
            f'{made} is made by ScopeSession.create, never constructed directly; '
            f'a session reaches its scope as session.scope'
        )
