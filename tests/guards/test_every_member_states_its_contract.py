# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a member may raise is its docstring's ``Raises:`` section, and the section can be read.

The section is the member's contract, its one statement (W0-R25): a caller
learns there what it may be told and how to branch on it, and the harness
walk reads it to know what each bad argument and each injected fault must
answer. Before this guard nothing read the sections, so they could name a
class that does not exist, a built-in the caller cannot branch on, or a
refusal without the reason that tells its causes apart. This guard holds,
over every marked member:

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

A static read: no session.
"""

from __future__ import annotations

import inspect
import re

from lib.lumascope_error import LumascopeError
from modules import wire_encoding
from modules.api_surface import (
    QUOTED_REASON,
    RAISES_ENTRY,
    RAISES_SECTION,
    _function_of,
    api,
    mark_of,
    raises_of,
)
from modules.exceptions import Refusal
from tests.guards.test_every_outcome_joins_the_root import EXCEPTION_CLASSES

# The section, its entries and a quoted reason are read as the REST
# description reads them (api_surface); a line at the entry indent that is
# no entry is this guard's own check.
_SECTION, _ENTRY, _QUOTED_REASON = RAISES_SECTION, RAISES_ENTRY, QUOTED_REASON
_AT_ENTRY_INDENT = re.compile(r'^ {4}\S.*$', re.M)
# The section of a member that raises nothing a caller handles.
_NOTHING = '    Nothing.'

# Refusal entries that do not yet say which reason; finished after W0
# (NEXT_PRIMARY). A pair leaves this set when its entry names its reason.
_REASON_UNNAMED_PINNED = frozenset(
    {
        ('DiagnosticsAPI.exit_led_engineering_mode', 'HardwareCommandRefusedError'),
        ('CameraAPI.set_auto_exposure', 'CameraSettingUnsupportedError'),
        ('CameraAPI.set_auto_gain', 'CameraSettingUnsupportedError'),
        ('CameraAPI.set_black_level', 'CameraSettingOutOfRangeError'),
        ('CameraAPI.set_black_level', 'CameraSettingUnsupportedError'),
        ('CameraAPI.set_exposure_ms', 'CameraSettingOutOfRangeError'),
        ('CameraAPI.set_gain_db', 'CameraSettingOutOfRangeError'),
        ('CameraAPI.update_auto_gain_target_brightness', 'CameraSettingUnsupportedError'),
        ('ManualRecordingController.start', 'ObjectiveUnknownError'),
        ('ManualRecordingController.start', 'RecordingRefusedError'),
        ('MotionAPI.jog_step', 'ObjectiveUnknownError'),
        ('MotionAPI.move_turret', 'HardwareCommandRefusedError'),
        ('ProtocolRunner.start_autofocus', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.start_autofocus_all_steps', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_composite', 'ProtocolRunRefusedError'),
        # Each run_ waiting form refuses as its start does, and names its
        # reasons when its start does.
        ('ProtocolRunner.run_autofocus', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_autofocus_all_steps', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_protocol', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_single_scan', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_zstack', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.start_protocol', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.start_single_scan', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.start_zstack', 'ObjectiveUnknownError'),
        ('ProtocolRunner.start_zstack', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.start_composite', 'ProtocolRunRefusedError'),
        ('ProtocolsAPI.create_protocol', 'CatalogueNameRefusedError'),
        ('ProtocolsAPI.create_protocol', 'ProtocolRunRefusedError'),
        ('ProtocolsAPI.focus_z', 'ProtocolRunRefusedError'),
        ('ProtocolsAPI.step_targets', 'CatalogueNameRefusedError'),
        ('ScopeSession.apply_focus_to_layer_steps', 'ProtocolRunRefusedError'),
        ('ScopeSession.apply_tiling', 'ProtocolRunRefusedError'),
        ('ScopeSession.apply_zstacking', 'ProtocolRunRefusedError'),
        ('ScopeSession.recover_file_writer', 'HardwareCommandRefusedError'),
        ('ScopeSession.save_all_bookmarks', 'ProtocolRunRefusedError'),
        ('ScopeSession.save_focus', 'ProtocolRunRefusedError'),
        ('ScopeSession.save_layer_focus', 'ProtocolRunRefusedError'),
        ('ScopeSession.run_application_session', 'HardwareCommandRefusedError'),
        ('ScopeSession.start_go_to_step', 'ProtocolRunRefusedError'),
    }
)


def _sections(classes: dict[str, type] | None = None):
    """``('Class.member', section text or None)`` for every marked member of ``classes``, the project's by default."""
    for owner, cls in sorted(
        (wire_encoding.project_classes() if classes is None else classes).items()
    ):
        for name, member in vars(cls).items():
            if mark_of(member) is None:
                continue
            doc = inspect.getdoc(_function_of(member)) or ''
            found = _SECTION.search(doc)
            yield f'{owner}.{name}', found.group(1) if found else None


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
    without = _without_a_section(_sections())
    assert without == [], 'say what each raises, or Nothing.:\n' + '\n'.join(without)


class _Planted:
    """Three marked members the checks above must tell apart."""

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


def test_the_checks_name_what_they_refuse():
    planted = list(_sections({'_Planted': _Planted}))
    assert _without_a_section(planted) == ['_Planted.unstated']
    assert _prose(planted) == ["_Planted.nothing_and_an_entry: 'Nothing.'"]
    assert raises_of(inspect.getdoc(_Planted.nothing)) == []
