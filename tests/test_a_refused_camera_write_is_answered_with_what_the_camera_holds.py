# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera write the camera refused is answered with what the camera holds.

A setting is often several node writes, and the camera can take some before
it refuses one: a frame size whose width landed and whose height did not, a
binning one axis of which landed, an auto-gain whose gain loop started and
whose exposure loop was refused. The refusal used to say "It remains at the
previous size" and the API kept holding the old value, while the camera held
something else. Now ``_camera_write`` reads back each value the write holds
when the camera refuses it: the held value is what the camera reads, the
refusal's ``in_effect`` and words say it, and frame validity is treated as
for a write nobody answered, so a capture waits for a frame taken after it.
A refused auto-gain arm also turns both auto modes Off, so no loop runs
without its arm. A write that landed but whose value could not be read back
is held at what the camera reads after it, or as not known.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from pypylon import genicam

from _lumascope.drivers import camera_profiles
from _lumascope.drivers.exceptions import CameraWriteFailedError
from _lumascope.modules.exceptions import CameraSettingRejected
from _lumascope.modules.frame_validity import FrameValiditySource
from _lumascope.modules.lumascope_api import Lumascope
from _lumascope.modules.lumascope_api.camera import AutoGainSettings, CameraAutoMode
from _lumascope.modules.scope_session import ScopeSession
from _lumascope.rest.app import build_app
from tests.camera_fakes import bare_pylon_camera
from tests.scope_fakes import give_camera_and_imaging, give_camera_capabilities, give_stub_lanes
from tests.settings_fixtures import complete_settings, installation_of_its_own

_AUTO_GAIN = AutoGainSettings(
    target_brightness=0.5,
    min_gain_db=0.0,
    max_gain_db=20.0,
)


def test_a_refused_binning_is_answered_with_the_binning_the_camera_holds(sim_scope):
    camera = sim_scope.camera
    sim_scope.simulation.refuse_camera_write('binning', 'raise')

    with pytest.raises(CameraSettingRejected) as refused:
        camera.set_binning(2)

    assert refused.value.in_effect == 1 == camera.binning
    assert str(refused.value) == (
        'The camera did not accept 2x2 binning. It is at 1x1 binning. '
        'Try again, or check the USB connection if this repeats.'
    )


def test_a_refused_write_makes_the_next_capture_wait_for_a_frame_after_it(sim_scope):
    """The binning's source was invalidated only when the write landed: a
    refusal that had moved one node left a frame from before it acceptable."""
    validity = sim_scope.imaging._frame_validity
    sim_scope.simulation.refuse_camera_write('binning', 'raise')

    with pytest.raises(CameraSettingRejected):
        sim_scope.camera.set_binning(2)

    assert FrameValiditySource.BINNING in validity.changes()


def test_a_level_that_landed_unread_is_answered_as_not_known(sim_scope):
    """The camera took the black level, and neither its read-back nor the
    read after it answered: the write is not refused, and the level is not
    known, never the request."""
    camera = sim_scope.camera
    sim_scope.simulation.refuse_camera_write('black_level', 'unreported')
    sim_scope.simulation.refuse_camera_read('black_level', 'failed')

    assert camera.set_black_level(10.0) is None
    assert camera.black_level is None


def test_a_refused_auto_gain_arm_turns_both_auto_modes_off_and_clears_a_standing_arm(sim_scope):
    camera = sim_scope.camera
    camera.set_auto_gain(CameraAutoMode.CONTINUOUS, _AUTO_GAIN)
    sim_scope.simulation.refuse_camera_write('auto_gain', 'raise')

    with pytest.raises(CameraSettingRejected) as refused:
        camera.set_auto_gain(CameraAutoMode.CONTINUOUS, _AUTO_GAIN)

    assert camera._auto_gain_arm is None, 'no arm stands behind a loop the camera may run'
    assert (camera.auto_gain, camera.auto_exposure) == (CameraAutoMode.OFF, CameraAutoMode.OFF)
    assert refused.value.in_effect == CameraAutoMode.OFF
    assert (
        str(refused.value) == 'The camera did not take the auto-gain change. Its auto-gain is Off.'
    )


def test_a_refused_auto_gain_off_is_not_sent_again_and_keeps_its_arm(sim_scope):
    """The loop may still run, so its arm stays: the lock and the restore
    still know a loop chose the gain and exposure."""
    camera = sim_scope.camera
    camera.set_auto_gain(CameraAutoMode.CONTINUOUS, _AUTO_GAIN)
    sim_scope.simulation.refuse_camera_write('auto_gain', 'raise')

    with pytest.raises(CameraSettingRejected):
        camera.set_auto_gain(CameraAutoMode.OFF)

    assert camera._auto_gain_arm is not None
    assert camera.auto_gain == CameraAutoMode.CONTINUOUS, 'what the camera reads, not resent'


@pytest.fixture
def pylon_camera():
    """The real CameraAPI over a real PylonCamera on a fake SDK handle whose
    ``GainAuto`` and ``ExposureAuto`` keep what they are set to, and whose
    ``ExposureAuto`` refuses ``Continuous``."""
    cam = bare_pylon_camera()
    # The daA3840's profile, so the camera has hardware auto-gain.
    cam.profile = camera_profiles.lookup_profile('daA3840-45um')
    cam.cam_image_handler = None
    cam._device_removed = False
    for name in ('GainAuto', 'ExposureAuto'):
        node = getattr(cam.active, name)
        node.GetValue.return_value = 'Off'
        node.SetValue.side_effect = lambda v, node=node: setattr(node.GetValue, 'return_value', v)
    exposure_auto = cam.active.ExposureAuto

    def refuse_continuous(value):
        if value == 'Continuous':
            raise genicam.OutOfRangeException('Continuous is not offered')
        exposure_auto.GetValue.return_value = value

    exposure_auto.SetValue.side_effect = refuse_continuous
    # The real Pylon driver over a fake SDK handle, so one auto mode lands and
    # the other is refused; the simulated camera refuses an auto-gain write
    # whole, never one of its two nodes (a simulator gap).
    # a stand-in by design: one of two auto-mode nodes refused, which the simulator cannot do
    lumascope = Lumascope.__new__(Lumascope)
    lumascope._camera_driver = cam
    give_stub_lanes(lumascope)
    give_camera_capabilities(lumascope, cam)
    return give_camera_and_imaging(lumascope), cam


def test_a_gain_loop_the_camera_took_is_turned_off_when_its_exposure_loop_is_refused(
    pylon_camera,
):
    camera, cam = pylon_camera

    with pytest.raises(CameraSettingRejected) as refused:
        camera.set_auto_gain(CameraAutoMode.CONTINUOUS, _AUTO_GAIN)

    assert isinstance(refused.value.__cause__, CameraWriteFailedError)
    assert (cam.active.GainAuto.GetValue(), cam.active.ExposureAuto.GetValue()) == ('Off', 'Off')
    assert (camera.auto_gain, camera.auto_exposure) == (CameraAutoMode.OFF, CameraAutoMode.OFF)
    assert camera._auto_gain_arm is None


def test_a_refused_gain_reaches_a_rest_client_with_what_the_camera_holds(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    try:
        held = session.scope.camera.set_gain_db(3.0)
        session.scope.simulation.refuse_camera_write('gain', 'raise')
        with TestClient(build_app(session)) as client:
            refused = client.put('/api/v1/scope/camera/gain_db', json={'gain_db': 9.0})
    finally:
        session.shutdown()
        session.scope.disconnect()

    assert refused.status_code == 500, refused.text
    body = refused.json()
    assert (body['reason'], body['setting'], body['requested'], body['in_effect']) == (
        'camera_rejected',
        'gain_db',
        9.0,
        held,
    )


_HALF_LANDED_FRAME = """
import dataclasses, json, sys, tempfile
sys.path[:0] = ['src', '.']
try:
    import pypylon  # noqa: F401
except ImportError:
    print(json.dumps({'sdk': False})); sys.exit(0)
from _lumascope.drivers import camera as camera_module
from _lumascope.modules.exceptions import CameraSettingRejected
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

with tempfile.TemporaryDirectory() as folder:
    session = ScopeSession.create(complete_settings(live_folder=folder), source_path=sys.argv[1])
    try:
        if not session.scope.camera_connected:
            print(json.dumps({'sdk': True, 'connected': False})); sys.exit(0)
        camera = session.scope.camera
        driver = session.scope._camera_driver
        planned = camera_module.plan_aoi

        # The emulator's grid takes every planned window, so the plan is
        # replaced for one call: a width it takes and a height it refuses.
        def half_landing(*args, **kwargs):
            plan = planned(*args, **kwargs)
            return dataclasses.replace(
                plan, acq_width=int(driver.active.Width.GetMin()), acq_height=10**7
            )

        camera_module.plan_aoi = half_landing
        try:
            camera.set_frame_size(1024, 1008)
            refused = None
        except CameraSettingRejected as rejected:
            refused = rejected
        finally:
            camera_module.plan_aoi = planned
        read = camera.read_frame_size()
        held = camera.frame_size
        print(json.dumps({
            'sdk': True,
            'connected': True,
            'reason': refused and refused.reason,
            'in_effect': refused and refused.in_effect,
            'held': [held.width, held.height],
            'read': [read.width, read.height],
        }))
    finally:
        session.shutdown()
        session.scope.disconnect()
"""


def test_a_frame_size_half_landed_on_the_emulator_is_held_as_the_camera_reads_it():
    """Basler's emulator (``PYLON_CAMEMU``), in a child process: the width
    lands, the height is refused, and the API holds, and the refusal says,
    the frame the camera reads, not the one before the write. The child's
    scope is built on an installation of its own, so a scope held on this
    checkout's lock port does not refuse it."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, '-I', '-c', _HALF_LANDED_FRAME, str(installation_of_its_own())],
        cwd=str(repo),
        env={**os.environ, 'PYLON_CAMEMU': '1'},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    if not answer['sdk']:
        pytest.skip('pypylon is not installed on this host')
    if not answer['connected']:
        pytest.skip("pypylon's camera emulator did not open on this host")

    assert answer['reason'] == 'camera_rejected'
    assert answer['held'] == answer['read'] != [1900, 1900]
    assert answer['in_effect'] == {'width': answer['read'][0], 'height': answer['read'][1]}
