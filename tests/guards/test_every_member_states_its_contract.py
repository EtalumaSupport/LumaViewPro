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

# Built-ins the sections name, each raised by its member's body today. Typed
# after W0 (NEXT_PRIMARY); a pair leaves this set when its raise is typed.
_BUILT_INS_PINNED = {
    ('HeldClaim.release', 'RuntimeError'),
    ('ProtocolsAPI.tiling_config', 'RuntimeError'),
    ('ProtocolsAPI.tiling_config', 'ValueError'),
    ('ScopeSession.diagnostic_claim', 'RuntimeError'),
    ('ScopeSession.load_plugins', 'RuntimeError'),
    ('ScopeSession.start_application_session', 'RuntimeError'),
    ('ScopeSession.start_metrics', 'RuntimeError'),
}

# Refusal entries that do not yet say which reason; finished after W0
# (NEXT_PRIMARY). A pair leaves this set when its entry names its reason.
_REASON_UNNAMED_PINNED = frozenset(
    {
        ('DiagnosticsAPI.exit_led_engineering_mode', 'HardwareCommandRefusedError'),
        ('IlluminationAPI.led_off', 'ArgumentRefusedError'),
        ('IlluminationAPI.led_on', 'ArgumentRefusedError'),
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
        ('ProtocolRunner.run_autofocus', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_autofocus_all_steps', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_composite', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_protocol', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_single_scan', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.run_zstack', 'ObjectiveUnknownError'),
        ('ProtocolRunner.run_zstack', 'ProtocolRunRefusedError'),
        ('ProtocolRunner.start_composite', 'ProtocolRunRefusedError'),
        ('ProtocolsAPI.create_protocol', 'CatalogueNameRefusedError'),
        ('ProtocolsAPI.create_protocol', 'ProtocolRunRefusedError'),
        ('ProtocolsAPI.focus_z', 'ProtocolRunRefusedError'),
        ('ProtocolsAPI.plate_transform', 'ArgumentRefusedError'),
        ('ProtocolsAPI.stage_targets', 'CatalogueNameRefusedError'),
        ('ProtocolsAPI.step_targets', 'CatalogueNameRefusedError'),
        ('RunHandle.stop', 'ProtocolRunRefusedError'),
        ('RuntimeState.resolve_current_objective', 'ObjectiveUnknownError'),
        ('ScopeSession.apply_focus_to_layer_steps', 'ProtocolRunRefusedError'),
        ('ScopeSession.apply_remedy', 'RemedyUnknownError'),
        ('ScopeSession.apply_tiling', 'ProtocolRunRefusedError'),
        ('ScopeSession.apply_zstacking', 'ProtocolRunRefusedError'),
        ('ScopeSession.assign_turret_objective', 'HardwareCommandRefusedError'),
        ('ScopeSession.capture_settings_snapshot', 'ObjectiveUnknownError'),
        ('ScopeSession.clear_current_turret_objective', 'HardwareCommandRefusedError'),
        ('ScopeSession.clear_current_turret_objective', 'ObjectiveUnknownError'),
        ('ScopeSession.clear_turret_objective', 'HardwareCommandRefusedError'),
        ('ScopeSession.create', 'SettingRefusedError'),
        ('ScopeSession.diagnostic_claim', 'DiagnosticRefusedError'),
        ('ScopeSession.new_protocol', 'ObjectiveUnknownError'),
        ('ScopeSession.objective_question', 'ObjectiveUnknownError'),
        ('ScopeSession.open_remembered_protocol', 'HardwareCommandRefusedError'),
        ('ScopeSession.open_remembered_protocol', 'ProtocolRunRefusedError'),
        ('ScopeSession.recover_file_writer', 'FileWriterNotStuckError'),
        ('ScopeSession.recover_file_writer', 'HardwareCommandRefusedError'),
        ('ScopeSession.save_all_bookmarks', 'ProtocolRunRefusedError'),
        ('ScopeSession.save_focus', 'ProtocolRunRefusedError'),
        ('ScopeSession.save_layer_focus', 'ProtocolRunRefusedError'),
        ('ScopeSession.save_settings', 'SettingsSaveRefusedError'),
        ('ScopeSession.select_labware', 'HardwareCommandRefusedError'),
        ('ScopeSession.select_objective', 'HardwareCommandRefusedError'),
        ('ScopeSession.set_binning_size', 'CameraSettingUnsupportedError'),
        ('ScopeSession.set_frame_size', 'CameraSettingOutOfRangeError'),
        ('ScopeSession.set_high_conversion_gain', 'CameraSettingUnsupportedError'),
        ('ScopeSession.set_line_noise_reduction', 'CameraSettingUnsupportedError'),
        ('ScopeSession.start_application_session', 'HardwareCommandRefusedError'),
        ('ScopeSession.start_go_to_step', 'ProtocolRunRefusedError'),
        ('ScopeSession.update_settings', 'SettingRefusedError'),
    }
)

# The members with no Raises section at W0's end; it only falls.
_WITHOUT_A_SECTION_CEILING = 164


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
    assert outside - _BUILT_INS_PINNED == set(), (
        'name the LumascopeError the member raises, or type its raise: '
        f'{sorted(outside - _BUILT_INS_PINNED)}'
    )
    assert _BUILT_INS_PINNED - outside == set(), (
        f'typed, so take them out of the pin: {sorted(_BUILT_INS_PINNED - outside)}'
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
