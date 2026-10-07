# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A step on a scope with no XY stage still moves to its own Z.

A run's step move went to Z only after moving X and Y, so a step with no
X/Y left the focus wherever it was while the saved image recorded the
step's Z -- an image that says it was taken at a focus it was not.

A step with no plate position at all is refused where the protocol is
built until the Z-only row fixes every reader of an unknown position
(ruled F2); until then the step carries the position it was saved with,
and the scope, having no XY stage, moves only Z.

Driven through the Session on a simulated LS820, a scope with Z and no XY
stage.
"""

import json
import pathlib
import threading

import pytest
import tifffile

from modules.scope_session import ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings
from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings
from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

START_Z_UM = 500.0
STEP_Z_UM = 3000.0


@pytest.fixture
def zonly_session(tmp_path):
    session = ScopeSession.create(
        complete_settings(**_settings(tmp_path, microscope='LS820')), simulate=True
    )
    home_sim_scope(session.scope)
    session.scope.motion.move_absolute('Z', START_Z_UM)
    yield session
    session.shutdown()


def _step_at(z_um):
    step = {**_make_single_step_protocol().step(idx=0), 'Z': z_um}
    return _build_real_protocol([step])


def _run_and_wait_for_files(session, tmp_path, protocol):
    runner = session.create_protocol_runner()
    files_written = threading.Event()
    run = runner.run_single_scan(
        protocol=protocol,
        sequence_name='zonly',
        parent_dir=str(tmp_path),
        callbacks={
            'run_complete': lambda **kw: None,
            'files_complete': lambda **kw: files_written.set(),
        },
    )
    assert files_written.wait(COMPLETION_TIMEOUT), 'the run never finished its files'
    assert run.wait(timeout_s=COMPLETION_TIMEOUT) is not None


def test_the_run_ends_at_the_steps_z_and_the_image_says_so(zonly_session, tmp_path):
    motion = zonly_session.scope.motion
    assert not zonly_session.scope.capabilities.has_xy_stage
    assert motion.get_actual_position('Z') == pytest.approx(START_Z_UM)

    _run_and_wait_for_files(zonly_session, tmp_path, _step_at(STEP_Z_UM))

    assert motion.get_actual_position('Z') == pytest.approx(STEP_Z_UM)
    (image,) = [
        p for p in pathlib.Path(tmp_path).rglob('*') if p.suffix.lower() in ('.tif', '.tiff')
    ]
    with tifffile.TiffFile(str(image)) as tf:
        plane = json.loads(tf.pages[0].tags['ImageDescription'].value)['Plane']
    assert plane['PositionZ'] == pytest.approx(STEP_Z_UM)


def test_the_image_states_no_plate_position_on_a_scope_without_one(zonly_session, tmp_path):
    """The step carries a plate X and Y, but this scope has no XY stage and
    never went there: the image states its Z and no plate position, rather
    than the step's X and Y as a place it was taken."""
    _run_and_wait_for_files(zonly_session, tmp_path, _step_at(STEP_Z_UM))

    (image,) = [
        p for p in pathlib.Path(tmp_path).rglob('*') if p.suffix.lower() in ('.tif', '.tiff')
    ]
    with tifffile.TiffFile(str(image)) as tf:
        plane = json.loads(tf.pages[0].tags['ImageDescription'].value)['Plane']
    assert 'PositionX' not in plane
    assert 'PositionY' not in plane
    assert plane['PositionZ'] == pytest.approx(STEP_Z_UM)
