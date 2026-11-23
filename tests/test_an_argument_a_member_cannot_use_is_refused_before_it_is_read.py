# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An argument a member cannot use is refused at the member, as the request it is.

Each member here read its argument where it was first used: a stage offset
with a NaN axis converted to a NaN position, auto-gain settings with no target were
reported as the camera's rejection of a change it was never sent, a binning
of 0 divided by zero, and a diagnostic's duration of 10^12 s overflowed the
camera lane's wait. Each is now refused before the member acts, with the
reason a client branches on. The walk of every route
(``tests/guards/test_a_bad_request_is_answered_as_a_request.py``) holds the
class; this pins each member at the cheap end, on every push.
"""

from __future__ import annotations

import dataclasses

import pytest

from _lumascope.modules.coord_transformations import StageOffset
from _lumascope.modules import settings_paths
from _lumascope.modules.exceptions import (
    ArgumentRefusedError,
    CameraSettingUnsupportedError,
    SettingRefusedError,
)
from _lumascope.modules.lumascope_api.camera import AutoGainSettings, CameraAutoMode
from _lumascope.modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

_AUTO_GAIN = AutoGainSettings(
    target_brightness=0.5,
    min_gain_db=0.0,
    max_gain_db=20.0,
)


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    home_sim_scope(s.scope)
    # A protocol has steps only for a layer that acquires.
    s.set_channel_settings('BF', acquire='image')
    yield s
    s.shutdown()


_NAN_OFFSET = StageOffset(x_um=float('nan'), y_um=0.0)


def _rows(session):
    """Each member and argument, by the refusal and reason it answers."""
    scope = session.scope
    protocol = session.new_protocol()
    diagnostics = scope.diagnostics
    return [
        (
            'plate_to_stage, NaN offset x',
            ArgumentRefusedError,
            'not_a_number',
            lambda: scope.runtime_state.plate_to_stage(1.0, 1.0, stage_offset=_NAN_OFFSET),
        ),
        (
            'step_targets, NaN offset x',
            ArgumentRefusedError,
            'not_a_number',
            lambda: scope.protocols.step_targets(protocol, 0, stage_offset=_NAN_OFFSET),
        ),
        (
            'plate_transform, NaN offset x',
            ArgumentRefusedError,
            'not_a_number',
            lambda: scope.runtime_state.plate_transform(stage_offset=_NAN_OFFSET),
        ),
        (
            'set_auto_gain, Continuous with no settings',
            ArgumentRefusedError,
            'missing_key',
            lambda: scope.camera.set_auto_gain(CameraAutoMode.CONTINUOUS),
        ),
        (
            'set_auto_gain, Off with settings',
            ArgumentRefusedError,
            'not_read',
            lambda: scope.camera.set_auto_gain(CameraAutoMode.OFF, _AUTO_GAIN),
        ),
        (
            'set_auto_gain, a one-shot',
            ArgumentRefusedError,
            'not_offered',
            lambda: scope.camera.set_auto_gain(CameraAutoMode.ONCE, _AUTO_GAIN),
        ),
        (
            'set_auto_gain, text exposure bound',
            ArgumentRefusedError,
            'not_a_number',
            lambda: scope.camera.set_auto_gain(
                CameraAutoMode.CONTINUOUS, dataclasses.replace(_AUTO_GAIN, max_exposure_ms='x')
            ),
        ),
        (
            'set_auto_gain, minimum gain above maximum',
            ArgumentRefusedError,
            'gain_range_inverted',
            lambda: scope.camera.set_auto_gain(
                CameraAutoMode.CONTINUOUS,
                dataclasses.replace(_AUTO_GAIN, min_gain_db=25.0, max_gain_db=20.0),
            ),
        ),
        (
            'set_auto_gain, a target above full scale',
            ArgumentRefusedError,
            'wrong_kind',
            lambda: scope.camera.set_auto_gain(
                CameraAutoMode.CONTINUOUS, dataclasses.replace(_AUTO_GAIN, target_brightness=5.0)
            ),
        ),
        (
            'set_auto_gain, a target of 0',
            ArgumentRefusedError,
            'wrong_kind',
            lambda: scope.camera.set_auto_gain(
                CameraAutoMode.CONTINUOUS, dataclasses.replace(_AUTO_GAIN, target_brightness=0)
            ),
        ),
        (
            'set_auto_gain, a negative minimum gain',
            ArgumentRefusedError,
            'wrong_kind',
            lambda: scope.camera.set_auto_gain(
                CameraAutoMode.CONTINUOUS, dataclasses.replace(_AUTO_GAIN, min_gain_db=-10.0)
            ),
        ),
        (
            'set_auto_gain, an exposure ceiling of 0',
            ArgumentRefusedError,
            'wrong_kind',
            lambda: scope.camera.set_auto_gain(
                CameraAutoMode.CONTINUOUS, dataclasses.replace(_AUTO_GAIN, max_exposure_ms=0)
            ),
        ),
        (
            'apply, a target above full scale',
            ArgumentRefusedError,
            'wrong_kind',
            lambda: scope.camera.apply(
                1.0,
                5.0,
                auto_gain=CameraAutoMode.CONTINUOUS,
                auto_gain_settings=dataclasses.replace(_AUTO_GAIN, target_brightness=5.0),
            ),
        ),
        (
            'apply, Continuous with no auto-gain settings',
            ArgumentRefusedError,
            'missing_key',
            lambda: scope.camera.apply(1.0, 5.0, auto_gain=CameraAutoMode.CONTINUOUS),
        ),
        (
            'apply, auto-gain settings with no mode',
            ArgumentRefusedError,
            'not_read',
            lambda: scope.camera.apply(1.0, 5.0, auto_gain_settings=_AUTO_GAIN),
        ),
        (
            'frame_at_binning, 0',
            CameraSettingUnsupportedError,
            'binning_unsupported',
            lambda: session.scope.camera.frame_at_binning(0),
        ),
        (
            'run_camera_bandwidth_test, no frames',
            ArgumentRefusedError,
            'out_of_range',
            lambda: diagnostics.run_camera_bandwidth_test(0),
        ),
        (
            'run_camera_bandwidth_test, negative frames',
            ArgumentRefusedError,
            'out_of_range',
            lambda: diagnostics.run_camera_bandwidth_test(-1),
        ),
        (
            'run_grab_lifecycle_benchmark, too many cycles',
            ArgumentRefusedError,
            'out_of_range',
            lambda: diagnostics.run_grab_lifecycle_benchmark(num_cycles=10_001),
        ),
        (
            'run_grab_lifecycle_benchmark, negative delay',
            ArgumentRefusedError,
            'out_of_range',
            lambda: diagnostics.run_grab_lifecycle_benchmark(inter_cycle_delay_s=-1.0),
        ),
        (
            # Seconds: past a minute is refused, where the old milliseconds took 61.
            'run_grab_lifecycle_benchmark, a delay past a minute',
            ArgumentRefusedError,
            'out_of_range',
            lambda: diagnostics.run_grab_lifecycle_benchmark(inter_cycle_delay_s=61.0),
        ),
        (
            'run_pylon_diagnostic_probe, negative duration',
            ArgumentRefusedError,
            'out_of_range',
            lambda: diagnostics.run_pylon_diagnostic_probe(duration_s=-1.0),
        ),
        (
            'run_pylon_diagnostic_probe, 10^12 s',
            ArgumentRefusedError,
            'out_of_range',
            lambda: diagnostics.run_pylon_diagnostic_probe(duration_s=1e12),
        ),
    ]


def test_every_argument_a_member_cannot_use_is_refused_with_its_reason(session):
    wrong = []
    for case, refusal, reason, call in _rows(session):
        try:
            call()
        except refusal as refused:
            if refused.reason != reason:
                wrong.append(f'{case}: refused {refused.reason!r}, not {reason!r}')
        except Exception as raised:  # each offender is named in the one failure
            wrong.append(
                f'{case}: raised {type(raised).__name__} {getattr(raised, "reason", None)!r}'
            )
        else:
            wrong.append(f'{case}: was not refused')
    assert not wrong, '\n'.join(wrong)


def test_the_stored_auto_gain_settings_arm_the_camera(session):
    # The settings the session reads from the stored block are the record
    # the camera's auto-gain takes, so a stored settings file arms it unrefused.
    session.scope.camera.set_auto_gain(CameraAutoMode.CONTINUOUS, session.get_auto_gain_settings())
    assert session.scope.camera.auto_gain is CameraAutoMode.CONTINUOUS


def test_auto_gain_is_turned_off_with_no_settings(session):
    # Off reads no settings, so it is asked for with none: it was refused
    # 'missing_key' for want of a target it never reads.
    camera = session.scope.camera
    camera.set_auto_gain(CameraAutoMode.CONTINUOUS, _AUTO_GAIN)

    camera.set_auto_gain(CameraAutoMode.OFF)

    assert camera.auto_gain is CameraAutoMode.OFF


def test_an_auto_gain_argument_takes_what_the_stored_setting_takes(session):
    # One rule for the stored ``protocol.autogain`` block and the settings a
    # caller hands the arm: each value the setting's own range refuses is
    # refused as an argument, in that range's words, and each it takes arms.
    # A gain past the camera's own limit is in the setting's range; the
    # camera's clamp of it is not this rule's.
    template = complete_settings()
    camera = session.scope.camera
    cases = [
        ('target_brightness', 1.0),
        ('target_brightness', 1.5),
        ('target_brightness', 0.0),
        ('min_gain_db', 0.0),
        ('min_gain_db', -0.5),
        ('max_gain_db', 1000.0),
        ('max_gain_db', -1.0),
    ]
    wrong = []
    for key, value in cases:
        try:
            settings_paths.check_value(template, f'protocol.autogain.{key}', value)
            stored = None
        except SettingRefusedError as refused:
            # The setting's clause, 'it must be <range>, not <value>', less its ends.
            stored = refused.detail.removeprefix('it must be ').rsplit(', not ', 1)[0]
        try:
            camera.set_auto_gain(
                CameraAutoMode.CONTINUOUS, dataclasses.replace(_AUTO_GAIN, **{key: value})
            )
            argued = None
        except ArgumentRefusedError as refused:
            argued = (refused.reason, str(refused))
        if stored is None and argued is not None:
            wrong.append(f'{key}={value}: the setting takes it, the argument {argued}')
        elif stored is not None and (argued is None or stored not in argued[1]):
            wrong.append(f'{key}={value}: the setting is {stored!r}, the argument {argued}')
    assert not wrong, '\n'.join(wrong)


def test_a_binning_this_camera_offers_is_still_previewed(session):
    size = session.scope.capabilities.camera_binning_sizes[-1]

    preview = session.scope.camera.frame_at_binning(size)

    assert preview['width'] > 0 and preview['height'] > 0
