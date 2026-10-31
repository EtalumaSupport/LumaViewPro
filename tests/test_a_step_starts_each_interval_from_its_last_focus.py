# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A step with autofocus starts each interval from the focus it last found.

A run takes one copy of the protocol and moves each step to the Z in that
copy, where its sweep is centred. A z-stack group wrote its found focus into
the copy every time; a plain step wrote it only in an Autofocus All Steps
scan, so in an ordinary time-lapse every interval swept around the saved Z
again and a sample that had moved was searched for where it used to be.

The found focus now goes into the run's copy for every step; the protocol
the caller passed in keeps its saved Z. A sweep that finds no focus writes
nothing, so the next interval starts from the Z the step had.

Driven end to end on the simulated scope through the Session: each
interval's sweep window is read from the Z positions the motion API
reports, and each interval's focus from the Z its saved frame records.
"""

import datetime

import pandas as pd
import pytest

from modules.image_utils import read_file_record
from modules.axes import StageAxis
from modules.protocol import Protocol
from modules.run_events import RunEvents
from tests.protocol_fixtures import protocol_of
from tests.test_a_late_write_records_its_frame import _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 120.0
SAVED_Z_UM = 5000.0  # the step's Z, as _step writes it
FOCAL_Z_UM = 5050.0  # inside the 10x sweep around SAVED_Z_UM
# Far outside that sweep: every frame of it is equally blurred, so the curve is flat.
OUT_OF_REACH_Z_UM = 9000.0


def _two_intervals():
    period = datetime.timedelta(seconds=1.0)
    return protocol_of(
        config={
            'version': Protocol.CURRENT_VERSION,
            'steps': pd.DataFrame([_step('S1', 0, x=20.0, gain=1.0, autofocus=True)]),
            'period': period,
            'duration': period * 2,
            'labware_id': '6 well microplate',
            'capture_root': '',
            'tiling': '1x1',
        },
    )


def _run_two_intervals(tmp_path, *, first_focal_z_um):
    """Run the protocol; return it, the lowest Z each interval's sweep reached, and each frame's Z.

    The sample's focal plane is at ``first_focal_z_um`` for interval 1 and
    at FOCAL_Z_UM for interval 2. The stage starts at the saved Z, so the
    lowest Z of an interval that focuses is its sweep's: the first window's
    low edge and its approach, a fixed distance below the Z the step
    started from. A sweep that finds no focus goes on lower.
    """
    protocol = _two_intervals()
    lowest = {}

    with open_composite_session(headless_settings(tmp_path)) as (session, runner):
        camera = session.scope._camera_driver
        camera.set_test_pattern(True, 'focus_target')
        session.scope.imaging.enable_streaming()
        camera.set_focal_z(first_focal_z_um)
        session.scope.motion.move_z(SAVED_Z_UM)

        def on_scan_started(scan_index, *_):
            if scan_index == 1:
                camera.set_focal_z(FOCAL_Z_UM)
            lowest[scan_index] = float('inf')

        def on_position(axis, at):
            if axis is StageAxis.Z and lowest and at.position_um is not None:
                interval = max(lowest)
                lowest[interval] = min(lowest[interval], at.position_um)

        session.scope.motion.add_position_listener(on_position)
        outcome = runner.start_protocol(
            protocol,
            parent_dir=str(tmp_path / 'runs'),
            events=RunEvents(scan_started=on_scan_started),
        ).result(timeout=WAIT_S)

    assert outcome.status == 'completed', outcome
    frames_z = [
        read_file_record(path)['z_pos_um'] for path in sorted((tmp_path / 'runs').rglob('*.tiff'))
    ]
    assert len(frames_z) == 2, frames_z
    return protocol, (lowest[0], lowest[1]), frames_z


@pytest.mark.slow
def test_the_next_interval_sweeps_around_the_focus_the_last_one_found(tmp_path):
    protocol, (low_1, low_2), (found_1, _) = _run_two_intervals(
        tmp_path, first_focal_z_um=FOCAL_Z_UM
    )

    assert found_1 != SAVED_Z_UM, 'interval 1 found no focus, so it proves nothing'
    # Interval 1 swept from the saved Z; interval 2's window sits the same
    # distance below the focus interval 1 found.
    assert low_2 - found_1 == pytest.approx(low_1 - SAVED_Z_UM, abs=1.0), (
        f'interval 2 swept from {low_2}: its window is not around the focus '
        f'interval 1 found, {found_1}'
    )
    assert protocol.steps()['Z'].tolist() == [SAVED_Z_UM], (
        "the run wrote its found focus into the caller's protocol"
    )


@pytest.mark.slow
def test_an_interval_that_found_no_focus_leaves_the_next_to_focus(tmp_path):
    protocol, _, (found_1, found_2) = _run_two_intervals(
        tmp_path, first_focal_z_um=OUT_OF_REACH_Z_UM
    )

    # A failed sweep puts the stage back and the step captures there.
    assert found_1 == SAVED_Z_UM, 'interval 1 found a focus, so it proves nothing'
    # Nothing was written for interval 2 to start from, and it found the
    # sample inside the window around the Z the step had.
    assert found_2 != SAVED_Z_UM
    assert found_2 == pytest.approx(FOCAL_Z_UM, abs=10.0), found_2
    assert protocol.steps()['Z'].tolist() == [SAVED_Z_UM]
