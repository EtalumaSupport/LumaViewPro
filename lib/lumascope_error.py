# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The root every exception the scope raises descends from, the fault kind,
and the umbrella mixin.

Here, not in ``modules/exceptions.py``, because drivers raise faults and
name a subject (``HardwareError``), and may not import ``modules/``; the
other three kinds (``Refusal``, ``Notice``, ``Quiet``) live in
``modules/exceptions.py`` and subclass the root, since no driver raises
them. Standard library only, so every layer may import it.

None of these classes is an ``Exception``: each is mixed into a concrete
class that keeps the built-in base it has (``Exception``, ``ValueError``,
...), so joining the root changes what no ``except`` catches.
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


class _Umbrella(LumascopeError):  # noqa: N818 -- a mixin on the root, not an error
    """A subject several outcomes share, caught by name and never raised itself.

    ``except ConfigError`` catches every configuration outcome, whatever its
    kind; a raise of the subject alone would have no kind, reason or title
    to answer with, so constructing an umbrella itself is refused. Its
    children -- each a refusal, a notice, a quiet outcome or a fault -- are
    constructed as before.

    Refused in ``__init__``, not ``__new__``: a ``__new__`` here would hand a
    ``TimeoutError`` child's arguments to ``OSError``, which reads two of
    them as an errno and its words in place of the child's message.
    """

    def __init__(self, *args, **kwargs):
        self._refuse_if_bare()
        super().__init__(*args, **kwargs)

    def _refuse_if_bare(self) -> None:
        """Refuse the umbrella itself; asked first by an umbrella whose words read a child's table."""
        if _Umbrella in type(self).__bases__:
            raise TypeError(
                f'{type(self).__name__} names a subject and is never raised: raise the child '
                'that says what happened'
            )
