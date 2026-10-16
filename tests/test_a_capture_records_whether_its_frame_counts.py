# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture records whether the frame it returns counts for each change.

Whether a frame counts for a change is decided in one place,
``FrameValidity.judge``, from facts the frame and the change both carry: the
frame arrived after the change, the change's count was met by frames that
arrived before it, and an axis the change waits on had settled before it
arrived. A capture judges the frame it returns against each source's last
change and keeps the answers in its record, its saved file and one line of
``api.log``, naming who counted each frame toward the wait.
"""

import logging
import threading
import time

import numpy as np
import pytest

from modules import image_save, image_utils
from modules.exceptions import ArgumentTypeRefusedError
from modules.frame_validity import Counter, FrameValidity
from modules.labware_loader import WellPlateLoader
from modules.layer_record import release_color
from tests.camera_fakes import delivered_frame

_PIXELS = np.zeros((2, 2), dtype=np.uint8)


def _frame(seq: int, arrived_ns: int | None = None):
    return delivered_frame(_PIXELS, seq=seq, arrived_ns=arrived_ns)


def _judgment(record, source: str):
    return next(j for j in record.judgments if j.change.source == source)


def test_a_frame_counts_only_once_the_count_was_met_before_it():
    fv = FrameValidity(lambda: 0)
    fv.invalidate('gain')  # skip 2, recorded at frame 0
    fv.count_frame(1, by=Counter.GRAB)

    early = fv.judge(_frame(2), fv.changes()['gain'])
    fv.count_frame(2, by=Counter.DRAIN)
    late = fv.judge(_frame(3), fv.changes()['gain'])

    assert (early.counts, early.frames_after) == (False, 2)
    assert (late.counts, late.frames_after) == (True, 3)
    assert [(c.seq, c.by) for c in late.change.credits] == [
        (1, Counter.GRAB),
        (2, Counter.DRAIN),
    ]


def test_a_motion_frame_counts_only_once_its_axis_settled_before_it():
    settled = threading.Event()
    fv = FrameValidity(lambda: 0)
    fv.set_settle_check(lambda source: settled.is_set())
    change = fv.invalidate('z_move')
    for seq in (1, 2):
        fv.count_frame(seq, by=Counter.DRAIN)

    moving = fv.judge(_frame(3), fv.changes()['z_move'])
    settled.set()
    at_ns = time.monotonic_ns()
    fv.motion_settled('z_move', at_ns)
    before = fv.judge(_frame(3, arrived_ns=at_ns - 1), fv.changes()['z_move'])
    after = fv.judge(_frame(3, arrived_ns=at_ns + 1), fv.changes()['z_move'])

    assert moving.counts is False and moving.ms_after_in_force is None
    assert before.counts is False
    assert after.counts is True
    # A stamp from before the change was made is not the change's own.
    fv.invalidate('z_move')
    fv.motion_settled('z_move', change.written_ns)
    assert fv.changes()['z_move'].in_force_ns is None


def test_a_frame_the_preview_counted_names_the_preview(sim_scope, caplog):
    imaging = sim_scope.imaging
    change = None
    arrived = threading.Event()

    def on_frame(frame):
        if change is not None and frame.seq > change.at_seq:
            arrived.set()

    imaging.add_frame_listener(on_frame, name='after the gain write')
    imaging.set_gain_db(imaging.get_gain_db() + 1.0)
    change = imaging.frame_validity.changes()['gain']
    assert arrived.wait(5.0), 'no frame arrived after the gain write'
    imaging.remove_frame_listener(on_frame)
    imaging.get_image_from_buffer()

    with caplog.at_level(logging.INFO, logger='LVP.api'):
        imaging.capture_and_wait(accept_dark=True)

    gain = _judgment(imaging.last_capture_info['frame_record'], 'gain')
    assert gain.counts
    assert gain.change.credits[0].by is Counter.PREVIEW
    assert gain.change.written_ns == change.written_ns
    line = next(r.getMessage() for r in caplog.records if 'capture_and_wait: frame' in r.message)
    assert f'gain counts +{gain.frames_after}fr' in line
    assert f'{gain.change.credits[0].seq}:preview' in line


def test_frames_the_captures_drain_counted_name_the_drain(sim_scope, caplog):
    """With nothing else counting, the capture's own drain meets the
    change's count, and its record and its log line say so."""
    imaging = sim_scope.imaging
    imaging.set_gain_db(imaging.get_gain_db() + 1.0)

    with caplog.at_level(logging.INFO, logger='LVP.api'):
        imaging.capture_and_wait(accept_dark=True)

    gain = _judgment(imaging.last_capture_info['frame_record'], 'gain')
    assert gain.counts
    assert [credit.by for credit in gain.change.credits] == [Counter.DRAIN] * gain.change.skip
    line = next(r.getMessage() for r in caplog.records if 'capture_and_wait: frame' in r.message)
    assert ','.join(f'{c.seq}:drain' for c in gain.change.credits) in line


def test_frames_a_grab_counted_name_the_grab(sim_scope, caplog):
    """An ungated grab after a change counts the frames it takes, so a
    capture after enough of them waits on none, and its record and its log
    line name the grab as what counted them."""
    imaging = sim_scope.imaging
    imaging.set_gain_db(imaging.get_gain_db() + 1.0)
    for _ in range(imaging.frame_validity.changes()['gain'].skip):
        imaging.get_image(force_new_capture=True)

    with caplog.at_level(logging.INFO, logger='LVP.api'):
        imaging.capture_and_wait(accept_dark=True)

    gain = _judgment(imaging.last_capture_info['frame_record'], 'gain')
    assert gain.counts
    assert [credit.by for credit in gain.change.credits] == [Counter.GRAB] * gain.change.skip
    line = next(r.getMessage() for r in caplog.records if 'capture_and_wait: frame' in r.message)
    assert ','.join(f'{c.seq}:grab' for c in gain.change.credits) in line


def test_a_move_is_in_force_when_its_axis_settles(sim_scope):
    motion = sim_scope.motion
    motion.home('Z')
    motion.move_absolute('Z', motion.get_current_position('Z') + 100.0)

    record_change = sim_scope.imaging.frame_validity.changes()['z_move']
    sim_scope.imaging.capture_and_wait(accept_dark=True)
    z = _judgment(sim_scope.imaging.last_capture_info['frame_record'], 'z_move')

    assert record_change.in_force_ns is not None
    assert record_change.in_force_ns >= record_change.written_ns
    assert z.counts and z.ms_after_in_force >= 0


def test_the_saved_file_carries_the_judgments(sim_scope, tmp_path):
    # The structured TIFF: OME-XML keeps only schema attributes, so on that
    # path the judgments go as the frame id and camera ticks already do.
    sim_scope.imaging.set_exposure_ms(20.0)
    image = sim_scope.imaging.capture_and_wait(accept_dark=True, force_to_8bit=True)
    record = sim_scope.imaging.last_capture_info['frame_record']
    metadata = image_save.generate_image_metadata(
        sim_scope,
        channel='BF',
        plate_x_mm=12.0,
        plate_y_mm=8.0,
        stage_z_um=4321.0,
        objective_id='10x Oly',
        frame_record=record,
        labware=WellPlateLoader().get_plate('6 well microplate'),
        well_label='A1',
    )
    path = tmp_path / 'judged.tif'
    image_utils.write_tiff(
        data=image,
        file_loc=path,
        metadata=metadata,
        ome=False,
        color=release_color('BF'),
        significant_bits=8,
        save_encoding='8bit',
    )

    saved = image_utils.read_postproc_input_metadata(path)

    assert saved['frame_validity'] == [j.fields() for j in record.judgments]
    assert _judgment(record, 'exposure').counts


def test_a_listeners_frame_is_judged_through_the_api(sim_scope):
    imaging = sim_scope.imaging
    imaging.set_gain_db(imaging.get_gain_db() + 1.0)
    at_seq = imaging.frame_validity.changes()['gain'].at_seq
    later = []
    arrived = threading.Event()

    def on_frame(frame):
        if frame.seq > at_seq:
            later.append(frame)
            arrived.set()

    imaging.add_frame_listener(on_frame, name='after the gain write')
    assert arrived.wait(5.0), 'no frame arrived after the gain write'
    imaging.remove_frame_listener(on_frame)

    gain = next(j for j in imaging.judge_frame(later[0]) if j.change.source == 'gain')

    assert gain.frame_seq == later[0].seq
    assert gain.frames_after == later[0].seq - at_seq
    # Nothing counted toward the write before this frame: it does not count.
    assert gain.counts is False


def test_judging_something_that_is_not_a_frame_is_refused(sim_scope):
    with pytest.raises(ArgumentTypeRefusedError) as refused:
        sim_scope.imaging.judge_frame(_PIXELS)
    assert refused.value.reason == 'wrong_argument_type'
