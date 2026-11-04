# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The root every exception the scope raises descends from, the fault kind,
the umbrella mixin, and the causes an outcome declares.

Here, not in ``modules/exceptions.py``, because drivers raise faults and
name a subject (``HardwareError``), and may not import ``modules/``; the
other three kinds (``Refusal``, ``Notice``, ``Quiet``) live in
``modules/exceptions.py`` and subclass the root, since no driver raises
them. Standard library only, so every layer may import it.

None of these classes is an ``Exception``: each is mixed into a concrete
class that keeps the built-in base it has (``Exception``, ``ValueError``,
...), so joining the root changes what no ``except`` catches.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import ClassVar


class RefusalCause(StrEnum):
    """Whether a refused request could succeed later, and what must change for it to.

    Declared on the refusal's type, per reason where one type mixes them, so
    a server answers from the declaration (REST: 409 for ``STATE``, 422 for
    ``REQUEST``, 503 for ``RESTART``) and never from the exception's base: a
    ``ValueError`` can be refused for the scope's state, as a live folder on
    an unplugged drive is. A string enum, so the value crosses a wire as
    itself.

    Attributes:
        STATE: The scope's state refused it -- a run is live, an axis is not
            homed, a drive is unplugged -- and the same request may succeed
            later.
        REQUEST: As sent, the request cannot succeed on this scope: a bad
            argument, a value outside a fixed rule or this hardware's range,
            hardware this scope does not have, a handle or a control session
            that no longer names anything live, a file already at the name.
            The caller changes the request.
        RESTART: Nothing the caller changes makes it succeed while this
            process runs -- the scope is disconnected, the motion monitor
            has ended, the session or the server is closing. Do not retry:
            the application must be restarted.
    """

    STATE = 'state'
    REQUEST = 'request'
    RESTART = 'restart'


class _Reason:
    """An outcome's reason: set where it is raised, or the one its class declares.

    A data descriptor, so every write passes the class's ``causes``: a
    reason the class does not declare fails where the outcome is built,
    whichever class builds it. Read off the class itself, a class with one
    reason answers it.
    """

    def __get__(self, instance, owner):
        if instance is not None and 'reason' in instance.__dict__:
            return instance.__dict__['reason']
        causes = getattr(owner, 'causes', None)
        if causes is not None and len(causes) == 1:
            return next(iter(causes))
        raise AttributeError(
            f'{owner.__name__} carries one of several reasons, set where it is raised'
        )

    def __set__(self, instance, value):
        if value not in getattr(type(instance), 'causes', ()):
            raise TypeError(f'{type(instance).__name__} declares no reason {value!r}')
        instance.__dict__['reason'] = value


class _Cause:
    """An outcome's cause: its reason's row in its class's ``causes``, never set apart from it."""

    def __get__(self, instance, owner):
        return owner.causes[(owner if instance is None else instance).reason]

    def __set__(self, instance, value):
        raise AttributeError("an outcome's cause is its reason's, declared in its class's causes")


class LumascopeError(BaseException):
    """The root of every exception the scope raises: catch it to catch them all.

    Its kind -- a refusal, a notice, a quiet outcome or a fault -- is the
    concrete class's, stated by one kind mixin among its bases; the
    reporter reads it from there (``notification_center.outcome_of``).

    Every outcome class -- each class below exactly one kind -- declares
    its reasons as ``causes``, checked where the class is defined
    (``__init_subclass__``) against every outcome class Python holds: a
    string has one kind and one cause wherever it is carried, and a class
    that carries a string another class carries names that class in
    ``same_fact_as``. A subclass carries its parent's reasons as the same
    fact, restated or not, and never with another cause. A class on the
    root is a kind or the umbrella mixin, and a subject below the umbrella
    with no kind carries no reasons. No index is kept: the classes are
    ``__subclasses__()``, so a class nothing holds is gone once collected,
    and one refused where it is defined is never counted.

    Attributes:
        reason: The machine-readable code a client branches on: one of
            ``causes``, set where the outcome is raised, or the class's
            one reason.
        cause: Whether the same request could succeed later, read from
            ``causes`` by ``reason``; None for a fault and a notice that is
            only reported.
        causes: Each reason the class carries, mapped to its
            ``RefusalCause``, or to None where the kind has no cause.
        same_fact_as: The classes whose reasons this class carries as the
            same fact.
    """

    reason = _Reason()
    cause = _Cause()
    causes: ClassVar[Mapping[str, RefusalCause | None]]
    same_fact_as: ClassVar[tuple[type['LumascopeError'], ...]] = ()
    # Whether a kind's reasons map to a cause: True for every reason, False
    # for none, None for either; read from the outcome's kind mixin.
    _needs_cause: ClassVar[bool | None]
    # Set in a class's own body once its check has passed, never inherited:
    # a class refused where it is defined lives until it is collected, and
    # unmarked it is no carrier and no outcome class meanwhile.
    _defined: ClassVar[bool]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Fault and the umbrella mixin, defined beside the check, are the root's own.
        if cls.__module__ != __name__:
            _check_the_declaration(cls)
        cls._defined = True


def _kinds_of(cls: type) -> list[type]:
    """The kinds ``cls`` is, itself included: the root's children that say whether a reason takes a cause."""
    return [k for k in cls.__mro__ if LumascopeError in k.__bases__ and '_needs_cause' in vars(k)]


def _check_the_declaration(cls: type[LumascopeError]) -> None:
    """Refuse, where it is defined, an outcome class whose reasons do not mean one thing.

    A kind itself is not an outcome, and a subject below the umbrella that
    has no kind names what its children are about.

    Raises:
        TypeError: the class has no kind, or two; is a subject that declares
            ``causes``; states ``reason`` or ``cause`` in its body; declares
            no ``causes``; maps a reason to a cause its kind does not take;
            restates a parent's reason with another cause; or carries a
            string another outcome class carries with another kind or cause,
            or without naming one of its carriers in ``same_fact_as``.
    """
    kinds = _kinds_of(cls)
    if kinds == [cls]:
        return
    if not kinds:
        if not issubclass(cls, _Umbrella):
            raise TypeError(f'{cls.__name__} has no kind: mix in Refusal, Notice, Quiet or Fault')
        if 'causes' in vars(cls):
            raise TypeError(
                f'{cls.__name__} is a subject and is never raised: its children declare the '
                'reasons, each under its kind'
            )
        return
    if len(kinds) > 1:
        raise TypeError(
            f'{cls.__name__} is of two kinds, {" and ".join(k.__name__ for k in kinds)}: '
            'an outcome is one of them'
        )
    kind = kinds[0]
    stated = [name for name in ('reason', 'cause') if name in vars(cls)]
    if stated:
        raise TypeError(
            f'{cls.__name__} states {" and ".join(stated)} in its body: declare its reasons '
            'as causes, each mapped to its cause, and set the reason where it is raised'
        )
    causes = getattr(cls, 'causes', None)
    if not isinstance(causes, Mapping) or not causes:
        raise TypeError(
            f'{cls.__name__} declares no reasons: state causes, each reason mapped to its '
            'RefusalCause, or to None where its kind has none'
        )
    needed = kind._needs_cause
    for reason, cause in causes.items():
        takes = (
            isinstance(cause, RefusalCause)
            if needed
            else cause is None
            if needed is False
            else cause is None or isinstance(cause, RefusalCause)
        )
        if not isinstance(reason, str) or not takes:
            raise TypeError(
                f'{cls.__name__} maps {reason!r} to {cause!r}, which a {kind.__name__} '
                'does not take'
            )
    own = vars(cls).get('causes', {})
    for reason in own:
        parent = next((a for a in cls.__mro__[1:] if reason in vars(a).get('causes', {})), None)
        if parent is not None:
            # The parent's row was checked where the parent was defined.
            if vars(parent)['causes'][reason] != own[reason]:
                raise TypeError(
                    f'{cls.__name__} carries {reason!r} as {own[reason]}, and its parent '
                    f'{parent.__name__} as {vars(parent)["causes"][reason]}: a subclass carries '
                    "its parent's reasons as they are -- name the fact apart"
                )
            continue
        carriers = _carriers(reason, cls)
        for other in carriers:
            other_kind = _kinds_of(other)[0]
            if other_kind is not kind or other.causes[reason] != own[reason]:
                raise TypeError(
                    f'{cls.__name__} carries {reason!r} as a {kind.__name__} '
                    f'({own[reason]}), and {other.__name__} as a {other_kind.__name__} '
                    f'({other.causes[reason]}): one string, one meaning -- name each fact apart'
                )
        if carriers and not any(reason in getattr(c, 'causes', ()) for c in cls.same_fact_as):
            raise TypeError(
                f'{cls.__name__} carries {reason!r}, which '
                f'{", ".join(sorted(c.__name__ for c in carriers))} carries: name one in '
                'same_fact_as if it is the same fact, or name the fact apart'
            )


def defined_classes() -> list[type[LumascopeError]]:
    """Every class below the root that Python holds and whose definition passed its check."""
    found, seen, pending = [], set(), [LumascopeError]
    while pending:
        for sub in pending.pop().__subclasses__():
            if sub in seen or not vars(sub).get('_defined'):
                continue
            seen.add(sub)
            pending.append(sub)
            found.append(sub)
    return found


def _carriers(reason: str, cls: type) -> list[type]:
    """Every other defined outcome class whose own ``causes`` holds ``reason``, ``cls``'s ancestors aside."""
    return [
        sub
        for sub in defined_classes()
        if sub not in cls.__mro__
        and isinstance(own := vars(sub).get('causes'), Mapping)
        and reason in own
    ]


class Fault(LumascopeError):  # noqa: N818 -- a kind of outcome, not an error
    """Something was attempted and failed.

    A fault's words are a developer's unless its type says they are written
    for the person: the reporter then shows them, and otherwise shows a
    generic sentence and keeps the words in the log. Its reasons map to no
    cause: a fault answers 500 whatever was asked.

    Attributes:
        words_for_person: True when the class writes its message for the
            person who will read it. Declared, never defaulted here: a
            default on this mixin would sit ahead of a value a fault
            inherits from its subject's class, and hide it. Unset reads as
            False.
    """

    words_for_person: ClassVar[bool]
    _needs_cause = False


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
