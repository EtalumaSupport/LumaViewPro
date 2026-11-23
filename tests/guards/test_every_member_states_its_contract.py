# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a member may raise is its docstring's ``Raises:`` section, and the section can be read.

The section is the member's contract, its one statement (W0-R25): a caller
learns there what it may be told and how to branch on it, and the harness
walk reads it to know what each bad argument and each injected fault must
answer. Before this guard nothing read the sections, so they could name a
class that does not exist, a built-in the caller cannot branch on, or a
refusal without the reason that tells its causes apart. This guard holds,
over every marked member of the declared API (``tests.marked_set``), a
module's marked function among them:

1. **Readable.** Each line at the entry indent is ``ClassName: words``,
   or the section is the one line ``Nothing.``; anything else there is
   prose a reader would take for an entry.
2. **Under the root.** Each class an entry names is a ``LumascopeError``.
   The nine built-in raises the sections name today are pinned, and the
   pin only shrinks: each is typed after W0.
3. **A refusal says which.** An entry naming a refusal class with more
   than one reason names the reason (``'not_connected'``). The
   entries that do not yet are pinned, and the pin only shrinks.
4. **Stated.** Every marked member has a section, ``Nothing.`` for one
   that raises nothing, so "has not said" is no state a member can be in;
   each one without is named.
5. **What is raised, not its subject.** No entry names an umbrella
   (``ConfigError``, ``HardwareError``): an umbrella is never raised, so
   the route's description would publish a class no client is answered
   with, under a status none of its children need answer.
6. **A reason in the form that is read.** A reason of the entry's class is
   quoted as ``'not_connected'`` (or after the word reason), the forms
   ``QUOTED_REASON`` reads; one written another way is published as if the
   entry had not named it, so the description carries every reason of the
   class or drops the one written.

A static read: no session.
"""

from __future__ import annotations

import inspect
import re

from _lumascope.lib.lumascope_error import LumascopeError
from _lumascope.modules.api_surface import (
    QUOTED_REASON,
    RAISES_ENTRY,
    RAISES_SECTION,
    _function_of,
    api,
    raises_of,
)
from _lumascope.modules.exceptions import Refusal
from tests.guards.test_every_outcome_joins_the_root import EXCEPTION_CLASSES, UMBRELLAS
from tests.marked_set import Marked, marked_members, marked_set

# The section, its entries and a quoted reason are read as the REST
# description reads them (api_surface); a line at the entry indent that is
# no entry is this guard's own check.
_SECTION, _ENTRY, _QUOTED_REASON = RAISES_SECTION, RAISES_ENTRY, QUOTED_REASON
_AT_ENTRY_INDENT = re.compile(r'^ {4}\S.*$', re.M)
# A name in an entry's words written as code, quoted or not: a reason of
# the entry's class among them is checked against what is read.
_CODE_TOKEN = re.compile(r"``'?([a-z][a-z0-9_]*)'?``")
# The section of a member that raises nothing a caller handles.
_NOTHING = '    Nothing.'

# The run starters' refusals, which state the whole class. What a run is
# refused for is what SequencedCaptureRunner.prepare() checks, most of the
# class for every kind of run, and it changes as prepare() does; a list of
# reasons here would be a copy of prepare() that nothing holds to it, and
# the first check added or removed there would publish a wrong list. An
# entry that quotes none is published as every reason of its class, which
# is never wrong. The set only shrinks: a pair leaves it when its entry
# names its reasons from a source that cannot drift from prepare().
_REASON_UNNAMED_PINNED = frozenset(
    {
        ('ProtocolRunner.start_autofocus', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.start_autofocus_all_steps', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.start_composite', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.start_protocol', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.start_single_scan', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.start_zstack', 'ProtocolRunRefusedError'),
        # Each run_ waiting form refuses as its start does.
        ('ProtocolRunner.run_autofocus', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_autofocus_all_steps', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_composite', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_protocol', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_single_scan', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_zstack', 'ProtocolRunRefusedError'),
    }
)


def _sections(marked: dict[str, Marked] | None = None):
    """``(name, section text or None)`` for every marked member of *marked*, the declared API by default."""
    for name, entry in sorted((marked_set() if marked is None else marked).items()):
        if entry.field:
            continue
        doc = inspect.getdoc(_function_of(entry.value)) or ''
        found = _SECTION.search(doc)
        yield name, found.group(1) if found else None


_ROOT = {cls.__name__: cls for cls in EXCEPTION_CLASSES if issubclass(cls, LumascopeError)}


def _multi_reason(cls: type) -> bool:
    return issubclass(cls, Refusal) and not isinstance(getattr(cls, 'reason', None), str)


def _prose(sections) -> list[str]:
    """Each line at the entry indent that is no entry, outside a section that is ``Nothing.`` alone."""
    return [
        f'{member}: {line.strip()!r}'
        for member, section in sections
        if section and section.rstrip() != _NOTHING
        for line in _AT_ENTRY_INDENT.findall(section)
        if not _ENTRY.match(line)
    ]


def _without_a_section(sections) -> list[str]:
    return [member for member, section in sections if section is None]


_UMBRELLA_NAMES = frozenset(cls.__name__ for cls in UMBRELLAS)


def _umbrella_entries(sections) -> list[str]:
    return [
        f'{member}: {name}'
        for member, section in sections
        if section
        for name, _ in _ENTRY.findall(section)
        if name in _UMBRELLA_NAMES
    ]


def _unread_reasons(sections) -> list[str]:
    """Each reason of an entry's class written in a form the description does not read."""
    found = []
    for member, section in sections:
        for name, words in _ENTRY.findall(section or ''):
            reasons = getattr(_ROOT.get(name), 'causes', {})
            read = {a or b for a, b in _QUOTED_REASON.findall(words)}
            unread = sorted({t for t in _CODE_TOKEN.findall(words) if t in reasons} - read)
            if unread:
                found.append(f'{member}: {name} {unread}')
    return found


def _own_reasons(name: str) -> frozenset[str]:
    """The reasons the class *name* can carry, its ``causes``; none for a class outside the root or without a table."""
    return frozenset(getattr(_ROOT.get(name), 'causes', {}))


def _foreign_reasons(sections) -> list[str]:
    """Each quoted reason that is not its entry's class's own, which the description publishes as the class's."""
    found = []
    for member, section in sections:
        for name, words in _ENTRY.findall(section or ''):
            own = _own_reasons(name)
            foreign = sorted({a or b for a, b in _QUOTED_REASON.findall(words)} - own)
            if own and foreign:
                found.append(f'{member}: {name} {foreign}')
    return found


def _doc(entry: Marked) -> str:
    return inspect.getdoc(_function_of(entry.value)) or ''


def _published(doc: str) -> dict[str, frozenset[str]]:
    """Each class *doc*'s section names, with the reasons published for it: those its entries quote, or every reason of the class when an entry quotes none."""
    published = {}
    for name, quoted in raises_of(doc):
        reasons = frozenset(quoted) if quoted else _own_reasons(name)
        published[name] = published.get(name, frozenset()) | reasons
    return published


def _waiting_pairs(marked: dict[str, Marked]) -> list[tuple[str, str]]:
    """``(waiting form, start)`` for every ``start_X`` with a waiting form ``X`` or ``run_X`` on its class.

    A waiting form starts as its start does and waits, so it raises what
    its start raises. A start whose body calls the bare form hands that
    form to a lane as the operation's work: the start raises only what
    refuses the hand-off, the bare form's raises reach the operation, and
    neither section covers the other, so that pair is not one.
    """
    pairs = []
    for name, entry in sorted(marked.items()):
        owner, _, member = name.rpartition('.')
        if entry.field or not member.startswith('start_'):
            continue
        bare = member.removeprefix('start_')
        if re.search(rf'\bself\.{bare}\b', inspect.getsource(_function_of(entry.value))):
            continue
        pairs.extend(
            (waiting, name)
            for waiting in (f'{owner}.run_{bare}', f'{owner}.{bare}')
            if waiting in marked and not marked[waiting].field
        )
    return pairs


def _uncovered(marked: dict[str, Marked]) -> list[str]:
    found = []
    for waiting, start in _waiting_pairs(marked):
        has = _published(_doc(marked[waiting]))
        for name, reasons in _published(_doc(marked[start])).items():
            lacking = 'the class' if name not in has else sorted(reasons - has[name])
            if lacking:
                found.append(f'{waiting} lacks what {start} raises: {name} {lacking}')
    return found


def test_every_section_is_a_list_of_entries():
    prose = _prose(_sections())
    assert prose == [], 'indent it under its entry, or make it one:\n' + '\n'.join(prose)


def test_every_class_named_is_under_the_root():
    outside = {
        (member, name)
        for member, section in _sections()
        if section
        for name, _ in _ENTRY.findall(section)
        if name not in _ROOT
    }
    assert outside == set(), (
        f'name the LumascopeError the member raises, or type its raise: {sorted(outside)}'
    )


def test_every_refusal_with_several_reasons_names_its_reason():
    unnamed = {
        (member, name)
        for member, section in _sections()
        if section
        for name, words in _ENTRY.findall(section)
        if name in _ROOT and _multi_reason(_ROOT[name]) and not _QUOTED_REASON.search(words)
    }
    assert unnamed - _REASON_UNNAMED_PINNED == set(), (
        f"quote the reason, as 'not_connected': {sorted(unnamed - _REASON_UNNAMED_PINNED)}"
    )
    assert _REASON_UNNAMED_PINNED - unnamed == set(), (
        f'named now, so take them out of the pin: {sorted(_REASON_UNNAMED_PINNED - unnamed)}'
    )


def test_every_member_has_a_section():
    sections = list(_sections())
    functions = {
        '_lumascope.rest.app.build_app',
        '_lumascope.lvp_logger.configure_logging',
    }
    assert functions <= {name for name, _ in sections}, 'the walk lost the functions'
    without = _without_a_section(sections)
    assert without == [], 'say what each raises, or Nothing.:\n' + '\n'.join(without)


def test_no_entry_names_an_umbrella():
    named = _umbrella_entries(_sections())
    assert named == [], 'name the classes under it that the member raises:\n' + '\n'.join(named)


def test_every_reason_is_quoted_as_it_is_read():
    unread = _unread_reasons(_sections())
    assert unread == [], "quote each as 'not_connected':\n" + '\n'.join(unread)


def test_every_quoted_reason_is_its_class_own():
    foreign = _foreign_reasons(_sections())
    assert foreign == [], (
        "a quoted name is published as the entry's class's reason; give it its own "
        'entry, or write it unquoted:\n' + '\n'.join(foreign)
    )


def test_a_waiting_form_names_what_its_start_raises():
    marked = {name: entry for name, entry in marked_set().items() if not entry.field}
    pairs = _waiting_pairs(marked)
    assert ('ProtocolRunner.run_zstack', 'ProtocolRunner.start_zstack') in pairs, (
        'the walk lost the run_ forms'
    )
    assert ('ScopeSession.go_to_step', 'ScopeSession.start_go_to_step') in pairs, (
        'the walk lost the bare waiting forms'
    )
    uncovered = _uncovered(marked)
    assert uncovered == [], 'name in the first what the second raises:\n' + '\n'.join(uncovered)


class _Planted:
    """Marked members the checks above must tell apart."""

    @api(reads=True)
    def unstated(self) -> None:
        """Says nothing of what it raises."""

    @api(reads=True)
    def nothing(self) -> None:
        """Raises nothing.

        Raises:
            Nothing.
        """

    @api(reads=True)
    def nothing_and_an_entry(self) -> None:
        """Says Nothing. beside an entry.

        Raises:
            Nothing.
            ConfigError: no session has bound the scope.
        """

    @api(reads=True)
    def an_umbrella(self) -> None:
        """Names the subject, not what is raised.

        Raises:
            HardwareError: the board did not answer.
        """

    @api(reads=True)
    def unread_and_read(self) -> None:
        """Quotes one reason as it is read, and writes another bare.

        Raises:
            ObjectiveUnknownError: ``'slot_unknown'``, the turret is lost;
                ``none_selected``, no objective is chosen. ``slot`` is the
                slot asked.
        """

    @api(reads=True)
    def a_foreign_reason(self) -> None:
        """Quotes, under one class, a reason of another.

        Raises:
            MoveFaultError: ``'stalled'``, the motor stalled; a park refused
                first is ``'monitor_ended'``.
        """

    @api(reads=True)
    def start_drift(self) -> None:
        """A start whose waiting form has drifted from it.

        Raises:
            ObjectiveUnknownError: ``'none_selected'``, no objective is chosen.
        """

    @api(reads=True)
    def run_drift(self) -> None:
        """Waits for ``start_drift`` and says less than it.

        Raises:
            Nothing.
        """

    @api(reads=True)
    def start_whole(self) -> None:
        """A start quoting one reason.

        Raises:
            ObjectiveUnknownError: ``'none_selected'``, no objective is chosen.
        """

    @api(reads=True)
    def run_whole(self) -> None:
        """Waits for ``start_whole``, stating the whole class, which covers it.

        Raises:
            ObjectiveUnknownError: no objective is known.
        """

    @api(reads=True)
    def start_handing(self) -> None:
        """Hands its bare form to a lane, whose raises reach the operation, not this call.

        Raises:
            Nothing.
        """
        return self.handing()

    @api(reads=True)
    def handing(self) -> None:
        """The work ``start_handing`` puts on a lane.

        Raises:
            ObjectiveUnknownError: ``'slot_unknown'``, the turret is lost.
        """


def test_the_checks_name_what_they_refuse():
    planted = list(_sections(marked_members({'_Planted': _Planted})))
    assert _without_a_section(planted) == ['_Planted.unstated']
    assert _prose(planted) == ["_Planted.nothing_and_an_entry: 'Nothing.'"]
    assert raises_of(inspect.getdoc(_Planted.nothing)) == []
    assert _umbrella_entries(planted) == [
        '_Planted.an_umbrella: HardwareError',
        '_Planted.nothing_and_an_entry: ConfigError',
    ]
    assert _unread_reasons(planted) == [
        "_Planted.unread_and_read: ObjectiveUnknownError ['none_selected']"
    ]
    assert _foreign_reasons(planted) == [
        "_Planted.a_foreign_reason: MoveFaultError ['monitor_ended']"
    ]
    planted_marked = marked_members({'_Planted': _Planted})
    assert _waiting_pairs(planted_marked) == [
        ('_Planted.run_drift', '_Planted.start_drift'),
        ('_Planted.run_whole', '_Planted.start_whole'),
    ]
    assert _uncovered(planted_marked) == [
        '_Planted.run_drift lacks what _Planted.start_drift raises: ObjectiveUnknownError the class'
    ]
