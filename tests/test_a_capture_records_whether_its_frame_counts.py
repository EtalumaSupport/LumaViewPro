# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture records whether the frame it returns counts for each change.

Whether a frame counts for a change is decided in one place,
``FrameValidity.judge``, from facts the frame and the change carry, by
counting frames: the frame arrived after the change, the change's count was
met by frames that arrived before it, and an axis the change waits on had
settled before it arrived. A capture judges the frame it returns against
each source's last change and keeps the answers in its record, its saved
file and one line of ``api.log``, naming who counted each frame toward the
wait.
"""

import threading

import numpy as np
import pytest

from _lumascope.lib.log_helpers import VERBOSE
from _lumascope.drivers import host_clock
from _lumascope.modules import image_save, image_utils
from _lumascope.modules.exceptions import ArgumentTypeRefusedError
from _lumascope.modules.frame_validity import Counter, FrameRule, FrameValidity
from _lumascope.modules.labware_loader import WellPlateLoader
from _lumascope.modules.channel_record import release_color
from tests.camera_fakes import delivered_frame

_PIXELS = np.zeros((2, 2), dtype=np.uint8)


def _frame(seq: int, arrived_ns: int | None = None):
    return delivered_frame(_PIXELS, seq=seq, arrived_ns=arrived_ns)


_PROVENANCE = {
    'run_id': 'run-1',
    'host': 'bench-pc',
    'os': 'Windows 11',
    'lvp_version': '4.0.0',
    'plugin_version': '1.0.85',
    'measured_date': '2026-10-20',
    'microscope': 'LS850',
}


def _judgment(record, source: str):
    return next(j for j in record.judgments if j.change.source == source)


def test_a_frame_counts_only_once_the_count_was_met_before_it():
    fv = FrameValidity(lambda: 0)
    fv.invalidate('gain', began_ns=None)  # skip 2, recorded at frame 0
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


def test_a_rule_states_a_whole_count():
    """Every rule carries a whole count of frames, 0 or more; its
    provenance is the run that measured it, or None for a shipped one."""
    wrong = {
        'a count of frames that is not whole': 2.5,
        'a negative count': -1,
        'a count that is a bool': True,
    }
    accepted = []
    for name, count in wrong.items():
        try:
            FrameRule(count=count, provenance=None)
            accepted.append(name)
        except ValueError:
            pass

    assert accepted == []
    assert FrameRule(count=4, provenance=_PROVENANCE).provenance == _PROVENANCE


def test_a_move_counts_the_third_frame_to_arrive_after_its_stop():
    """The frames that arrive throughout a long move count nothing
    toward it; of those arriving after its stop, the first two are dropped
    and the third counts."""
    delivered = 0
    settled = threading.Event()
    fv = FrameValidity(lambda: delivered)
    fv.set_settle_check(lambda source: settled.is_set())
    change = fv.invalidate('z_move', began_ns=None)

    def arrive() -> int:
        nonlocal delivered
        delivered += 1
        fv.count_frame(delivered, by=Counter.DRAIN)
        return delivered

    for _ in range(10):
        arrive()
    moving = fv.judge(_frame(delivered + 1), fv.changes()['z_move'])
    settled.set()
    stop_ns = host_clock.steady_ns()
    fv.motion_settled('z_move', stop_ns, moving_seen_ns=None)
    after_stop = [arrive() for _ in range(3)]
    stamped = fv.changes()['z_move']

    assert moving.counts is False and moving.after_in_force_ms is None
    assert [
        fv.judge(_frame(seq, arrived_ns=stop_ns + seq), stamped).counts for seq in after_stop
    ] == [False, False, True]
    # A stamp from before the change was made is not the change's own.
    fv.invalidate('z_move', began_ns=None)
    fv.motion_settled('z_move', change.written_ns, moving_seen_ns=None)
    assert fv.changes()['z_move'].in_force_ns is None


def test_a_frame_the_preview_counted_names_the_preview(sim_scope, caplog):
    imaging = sim_scope.imaging
    change = None
    arrived = threading.Event()

    def on_frame(frame):
        if change is not None and frame.seq > change.at_seq:
            arrived.set()

    imaging.add_frame_listener(on_frame, name='after the gain write')
    sim_scope.camera.set_gain_db(sim_scope.camera.read_gain_db() + 1.0)
    change = imaging._frame_validity.changes()['gain']
    assert arrived.wait(5.0), 'no frame arrived after the gain write'
    imaging.remove_frame_listener(on_frame)
    imaging.get_image_from_buffer()

    with caplog.at_level(VERBOSE, logger='LVP.api'):
        record = imaging.capture_still(accept_dark=True).record

    gain = _judgment(record, 'gain')
    assert gain.counts
    assert gain.change.credits[0].by is Counter.PREVIEW
    assert gain.change.written_ns == change.written_ns
    line = next(r.getMessage() for r in caplog.records if 'capture_still: frame' in r.message)
    assert f'gain counts +{gain.frames_after}fr' in line
    assert f'{gain.change.credits[0].seq}:preview' in line


def test_frames_the_captures_drain_counted_name_the_drain(sim_scope, caplog):
    """With nothing else counting, the capture's own drain meets the
    change's count, and its record and its log line say so."""
    imaging = sim_scope.imaging
    sim_scope.camera.set_gain_db(sim_scope.camera.read_gain_db() + 1.0)

    with caplog.at_level(VERBOSE, logger='LVP.api'):
        record = imaging.capture_still(accept_dark=True).record

    gain = _judgment(record, 'gain')
    assert gain.counts
    assert [credit.by for credit in gain.change.credits] == [Counter.DRAIN] * gain.change.skip
    line = next(r.getMessage() for r in caplog.records if 'capture_still: frame' in r.message)
    assert ','.join(f'{c.seq}:drain' for c in gain.change.credits) in line


def test_frames_a_grab_counted_name_the_grab(sim_scope, caplog):
    """An ungated grab after a change counts the frames it takes, so a
    capture after enough of them waits on none, and its record and its log
    line name the grab as what counted them."""
    imaging = sim_scope.imaging
    sim_scope.camera.set_gain_db(sim_scope.camera.read_gain_db() + 1.0)
    # A frame already on its way when the gain was written can be the one a
    # grab takes, and it does not count, so the grabs go on until the count
    # is met; a few spare grabs end the loop if grabs stop counting.
    for _ in range(imaging._frame_validity.changes()['gain'].skip + 3):
        if imaging._frame_validity.changes()['gain'].remaining == 0:
            break
        imaging.get_image(force_new_capture=True)

    with caplog.at_level(VERBOSE, logger='LVP.api'):
        record = imaging.capture_still(accept_dark=True).record

    gain = _judgment(record, 'gain')
    assert gain.counts
    assert [credit.by for credit in gain.change.credits] == [Counter.GRAB] * gain.change.skip
    line = next(r.getMessage() for r in caplog.records if 'capture_still: frame' in r.message)
    assert ','.join(f'{c.seq}:grab' for c in gain.change.credits) in line


def test_a_move_is_in_force_when_its_axis_settles(sim_scope):
    motion = sim_scope.motion
    motion.home()
    motion.move_z(motion.get_current_position('Z') + 100.0)

    record_change = sim_scope.imaging._frame_validity.changes()['z_move']
    z = _judgment(sim_scope.imaging.capture_still(accept_dark=True).record, 'z_move')

    assert record_change.in_force_ns is not None
    assert record_change.in_force_ns >= record_change.written_ns
    assert z.counts and z.after_in_force_ms >= 0


def test_the_saved_file_carries_the_judgments(sim_scope, tmp_path):
    # The structured TIFF: OME-XML keeps only schema attributes, so on that
    # path the judgments go as the frame id and camera ticks already do.
    sim_scope.camera.set_exposure_ms(20.0)
    still = sim_scope.imaging.capture_still(accept_dark=True, force_to_8bit=True)
    image, record = still.image, still.record
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
        fallback_focus=None,
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
        white_bits=None,
    )

    saved = image_utils.read_file_record(path)

    assert saved['frame_validity'] == [j.fields() for j in record.judgments]
    assert _judgment(record, 'exposure').counts


def test_a_full_depth_file_carries_the_judgments_as_plain_words(sim_scope, tmp_path):
    # A 16-bit frame goes through tifffile's ImageJ writer, which writes the
    # plane's dict as its text and is read back as a literal: a closed set's
    # member there wrote its repr, no literal, and the whole plane came back
    # as a string every post-processing build then failed on.
    sim_scope.camera.set_exposure_ms(20.0)
    still = sim_scope.imaging.capture_still(accept_dark=True, force_to_8bit=True)
    image, record = still.image, still.record
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
        fallback_focus=None,
    )
    path = tmp_path / 'judged_full_depth.tif'
    image_utils.write_tiff(
        data=image.astype(np.uint16) << 4,
        file_loc=path,
        metadata=metadata,
        ome=False,
        color=release_color('BF'),
        significant_bits=12,
        save_encoding='msb_aligned',
        white_bits=None,
    )

    saved = image_utils.read_file_record(path)

    assert saved['frame_validity'] == [j.fields() for j in record.judgments]
    assert {type(j['source']) for j in saved['frame_validity']} == {str}


def test_a_listeners_frame_is_judged_through_the_api(sim_scope):
    imaging = sim_scope.imaging
    sim_scope.camera.set_gain_db(sim_scope.camera.read_gain_db() + 1.0)
    at_seq = imaging._frame_validity.changes()['gain'].at_seq
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
