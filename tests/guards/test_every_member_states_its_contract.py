# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a member may raise is its docstring's ``Raises:`` section, and the section can be read.

The section is the member's contract, its one statement (W0-R25): a caller
learns there what it may be told and how to branch on it, and the harness
walk reads it to know what each bad argument and each injected fault must
answer. Before this guard nothing read the sections, so they could name a
class that does not exist, a built-in the caller cannot branch on, or a
refusal without the reason that tells its causes apart. This guard holds,
over every marked member:

1. **Readable.** Each line at the entry indent is ``ClassName: words``;
   anything else there is prose a reader would take for an entry.
2. **Under the root.** Each class an entry names is a ``LumascopeError``.
   The nine built-in raises the sections name today are pinned, and the
   pin only shrinks: each is typed after W0.
3. **A refusal says which.** An entry naming a refusal class with more
   than one reason names the reason (``'not_connected'``). The
   entries that do not yet are pinned, and the pin only shrinks.
4. **Counted.** The members with no section are counted, and the count
   only falls.

A static read: no session.
"""

from __future__ import annotations

import inspect
import re

from lib.lumascope_error import LumascopeError
from modules import wire_encoding
from modules.api_surface import _function_of, mark_of
from modules.exceptions import Refusal
from tests.guards.test_every_outcome_joins_the_root import EXCEPTION_CLASSES

_SECTION = re.compile(r'^Raises:\n((?:[ \t]+.*\n?|\n)+)', re.M)
_AT_ENTRY_INDENT = re.compile(r'^ {4}\S.*$', re.M)
_ENTRY = re.compile(r'^ {4}(\w+):(.*(?:\n {8,}.*)*)', re.M)
# A reason is written quoted (``'not_connected'``) or after the word
# (reason ``no_data``); either names it.
_QUOTED_REASON = re.compile(r"``'[a-z][a-z0-9_]*'``|reason\s+``[a-z][a-z0-9_]*``")

# Refusal entries that do not yet say which reason; finished after W0
# (NEXT_PRIMARY). A pair leaves this set when its entry names its reason.
_REASON_UNNAMED_PINNED = frozenset(
    {
        ('DiagnosticsAPI.exit_led_engineering_mode', 'HardwareCommandRefusedError'),
        ('ImagingAPI.set_auto_exposure_time', 'CameraSettingUnsupportedError'),
        ('ImagingAPI.set_auto_gain', 'ArgumentRefusedError'),
        ('ImagingAPI.set_auto_gain', 'CameraSettingUnsupportedError'),
        ('ImagingAPI.set_black_level', 'CameraSettingOutOfRangeError'),
        ('ImagingAPI.set_black_level', 'CameraSettingUnsupportedError'),
        ('ImagingAPI.set_exposure_ms', 'CameraSettingOutOfRangeError'),
        ('ImagingAPI.set_gain_db', 'CameraSettingOutOfRangeError'),
        ('ImagingAPI.update_auto_gain_target_brightness', 'CameraSettingUnsupportedError'),
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
        ('ProtocolsAPI.plate_transform', 'ArgumentRefusedError'),
        ('ProtocolsAPI.stage_targets', 'CatalogueNameRefusedError'),
        ('ProtocolsAPI.step_targets', 'CatalogueNameRefusedError'),
        ('ScopeSession.apply_focus_to_layer_steps', 'ProtocolRunRefusedError'),
        ('ScopeSession.apply_tiling', 'ProtocolRunRefusedError'),
        ('ScopeSession.apply_zstacking', 'ProtocolRunRefusedError'),
        ('ScopeSession.recover_file_writer', 'HardwareCommandRefusedError'),
        ('ScopeSession.save_all_bookmarks', 'ProtocolRunRefusedError'),
        ('ScopeSession.save_focus', 'ProtocolRunRefusedError'),
        ('ScopeSession.save_layer_focus', 'ProtocolRunRefusedError'),
        ('ScopeSession.select_labware', 'HardwareCommandRefusedError'),
        ('ScopeSession.run_application_session', 'HardwareCommandRefusedError'),
        ('ScopeSession.start_go_to_step', 'ProtocolRunRefusedError'),
    }
)

# The members with no Raises section at W0's end; it only falls. 165 since
# DiagnosticsAPI.motor_fan_spin_times, a sentinel read that raises nothing
# and so has no section to state (Eric, 2026-10-12); 166 since
# ScopeSession.remove_run_state_listener, a remove that raises nothing
# (Eric, 2026-10-13); 167 since ProtocolsAPI.tiling_config, a read of the
# scope's copy that bring-up refuses for it (Eric, 2026-10-13); 166 since
# IlluminationAPI.restore_led_state states the lease's refusal; 169 since
# ScopeSession.shown_channel and its listener pair, a read and an add and
# remove that raise nothing (Eric, 2026-10-13); the count does not yet tell
# a member that raises nothing from one that has not said.
_WITHOUT_A_SECTION_CEILING = 163


def _sections():
    """``('Class.member', section text or None)`` for every marked member."""
    for owner, cls in sorted(wire_encoding.project_classes().items()):
        for name, member in vars(cls).items():
            if mark_of(member) is None:
                continue
            doc = inspect.getdoc(_function_of(member)) or ''
            found = _SECTION.search(doc)
            yield f'{owner}.{name}', found.group(1) if found else None


_ROOT = {cls.__name__: cls for cls in EXCEPTION_CLASSES if issubclass(cls, LumascopeError)}


def _multi_reason(cls: type) -> bool:
    return issubclass(cls, Refusal) and not isinstance(getattr(cls, 'reason', None), str)


def test_every_section_is_a_list_of_entries():
    prose = [
        f'{member}: {line.strip()!r}'
        for member, section in _sections()
        if section
        for line in _AT_ENTRY_INDENT.findall(section)
        if not _ENTRY.match(line)
    ]
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


def test_the_members_without_a_section_only_fall():
    without = sum(1 for _, section in _sections() if section is None)
    assert without <= _WITHOUT_A_SECTION_CEILING, (
        f'{without} members state no Raises section, above the {_WITHOUT_A_SECTION_CEILING} '
        'pinned: give the new member its section'
    )
    assert without == _WITHOUT_A_SECTION_CEILING, f'down to {without}: lower the ceiling to it'
