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

import pytest

from modules.coord_transformations import StageOffset
from modules.exceptions import ArgumentRefusedError, CameraSettingUnsupportedError
from modules.lumascope_api.camera import CameraAutoMode
from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

_AUTO_GAIN = {'target_brightness': 0.5, 'min_gain_db': 0.0, 'max_gain_db': 20.0}


@pytest.fixture
def session(tmp_path):
    s = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    home_sim_scope(s.scope)
    # A protocol has steps only for a layer that acquires.
    s.set_layer_acquire('BF', 'image')
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
            'set_auto_gain, no target',
            ArgumentRefusedError,
            'missing_key',
            lambda: scope.camera.set_auto_gain(True, {'min_gain_db': 0, 'max_gain_db': 20}),
        ),
        (
            'set_auto_gain, text exposure bound',
            ArgumentRefusedError,
            'not_a_number',
            lambda: scope.camera.set_auto_gain(True, {**_AUTO_GAIN, 'max_exposure_ms': 'x'}),
        ),
        (
            'set_auto_gain, minimum gain above maximum',
            ArgumentRefusedError,
            'gain_range_inverted',
            lambda: scope.camera.set_auto_gain(
                True, {**_AUTO_GAIN, 'min_gain_db': 25.0, 'max_gain_db': 20.0}
            ),
        ),
        (
            'set_auto_gain, a key it does not take',
            ArgumentRefusedError,
            'key_unknown',
            lambda: scope.camera.set_auto_gain(True, {**_AUTO_GAIN, 'max_duration': 1.0}),
        ),
        (
            'apply, empty auto-gain settings',
            ArgumentRefusedError,
            'missing_key',
            lambda: scope.camera.apply(1.0, 5.0, auto_gain=True, auto_gain_settings={}),
        ),
        (
            'frame_at_binning, 0',
            CameraSettingUnsupportedError,
            'binning_unsupported',
            lambda: session.camera.frame_at_binning(0),
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
            lambda: diagnostics.run_grab_lifecycle_benchmark(inter_cycle_delay_ms=-1.0),
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
    # The dictionary the session hands every arm holds only keys the camera's
    # auto-gain takes, so a stored settings file arms it unrefused.
    assert session.scope.camera.set_auto_gain(True, session.get_auto_gain_settings()) is not False
    assert session.scope.camera.auto_gain is CameraAutoMode.CONTINUOUS


def test_a_binning_this_camera_offers_is_still_previewed(session):
    size = session.scope.capabilities.camera_binning_sizes[-1]

    preview = session.camera.frame_at_binning(size)

    assert preview['width'] > 0 and preview['height'] > 0
