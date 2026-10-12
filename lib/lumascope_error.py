# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The root every exception the scope raises descends from, and the fault kind.

Here, not in ``modules/exceptions.py``, because drivers raise faults and may
not import ``modules/``; the other three kinds (``Refusal``, ``Notice``,
``Quiet``) live in ``modules/exceptions.py`` and subclass the root, since no
driver raises them. Standard library only, so every layer may import it.

Neither class is an ``Exception``: each is mixed into a concrete class that
keeps the built-in base it has (``Exception``, ``ValueError``, ...), so
joining the root changes what no ``except`` catches.
"""

from typing import ClassVar


class LumascopeError(BaseException):
    """The root of every exception the scope raises: catch it to catch them all.

    Its kind -- a refusal, a notice, a quiet outcome or a fault -- is the
    concrete class's, stated by one kind mixin among its bases; the
    reporter reads it from there (``notification_center.outcome_of``).
    """


class Fault(LumascopeError):  # noqa: N818 -- a kind of outcome, not an error
    """Something was attempted and failed.

    A fault's words are a developer's unless its type says they are written
    for the person: the reporter then shows them, and otherwise shows a
    generic sentence and keeps the words in the log.

    Attributes:
        words_for_person: True when the class writes its message for the
            person who will read it. Declared, never defaulted here: a
            default on this mixin would sit ahead of a value a fault
            inherits from its subject's class, and hide it. Unset reads as
            False.
    """

    words_for_person: ClassVar[bool]
