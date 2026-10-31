# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A capture records whether the frame it returns counts for each change.

Whether a frame counts for a change is decided in one place,
``FrameValidity.judge``, from facts the frame, the change and the clock
mapping carry, by the rule the change's source runs. Under the credit rule
the frame arrived after the change, the change's count was met by frames
that arrived before it, and an axis the change waits on had settled before
it arrived; under the clock rule it started exposing, on the camera's clock,
after the change plus the source's measured delay. A capture judges the frame it returns against each source's last
change and keeps the answers in its record, its saved file and one line of
``api.log``, naming who counted each frame toward the wait.
"""

import logging
import threading

import numpy as np
import pytest

from drivers import host_clock
from modules import image_save, image_utils
from modules.exceptions import ArgumentTypeRefusedError
from modules.frame_validity import Counter, FrameRule, FrameValidity, Rule
from modules.lumascope_api.frame_record import ClockMapping
from modules.labware_loader import WellPlateLoader
from modules.layer_record import release_color
from tests.camera_fakes import delivered_frame

_PIXELS = np.zeros((2, 2), dtype=np.uint8)


def _frame(seq: int, arrived_ns: int | None = None, started_ticks: int | None = None):
    chunks = None if started_ticks is None else {'Timestamp': started_ticks}
    return delivered_frame(_PIXELS, seq=seq, arrived_ns=arrived_ns, chunks=chunks)


_PROVENANCE = {
    'run_id': 'run-1',
    'host': 'bench-pc',
    'os': 'Windows 11',
    'lvp_version': '4.0.0',
    'plugin_version': '1.0.85',
    'measured_date': '2026-10-20',
}


def _clocked(source: str, delay_us: float, settle_check=None) -> FrameValidity:
    """Frame validity whose ``source`` runs the clock rule at ``delay_us``."""
    fv = FrameValidity(lambda: 0)
    if settle_check is not None:
        fv.set_settle_check(settle_check)
    rules = dict(fv.credit_rules())
    rules[source] = FrameRule(
        rule=Rule.CLOCK, count=rules[source].count, delay_us=delay_us, provenance=_PROVENANCE
    )
    fv.use_rules(rules)
    return fv


def _mapping_at(host_ns: int) -> ClockMapping:
    """A nanosecond camera clock reading 1_000_000 at ``host_ns``, known to 50 ns."""
    return ClockMapping(host_ns=host_ns, camera_ticks=1_000_000, tick_hz=10**9, uncertainty_ns=50)


def _judgment(record, source: str):
    return next(j for j in record.judgments if j.change.source == source)


def test_a_frame_counts_only_once_the_count_was_met_before_it():
    fv = FrameValidity(lambda: 0)
    fv.invalidate('gain', began_ns=None)  # skip 2, recorded at frame 0
    fv.count_frame(1, by=Counter.GRAB)

    early = fv.judge(_frame(2), fv.changes()['gain'], None)
    fv.count_frame(2, by=Counter.DRAIN)
    late = fv.judge(_frame(3), fv.changes()['gain'], None)

    assert (early.counts, early.frames_after) == (False, 2)
    assert (late.counts, late.frames_after) == (True, 3)
    assert [(c.seq, c.by) for c in late.change.credits] == [
        (1, Counter.GRAB),
        (2, Counter.DRAIN),
    ]


def test_a_clock_rule_frame_counts_once_it_started_after_the_change_plus_its_delay():
    """No frame need be counted first: the frame's own start decides, on the
    camera's clock, against the write plus the measured delay plus the
    mapping's uncertainty."""
    fv = _clocked('gain', delay_us=1000.0)
    change = fv.invalidate('gain', began_ns=None)
    mapping = _mapping_at(change.written_ns)
    qualifies_after = 1_000_000 + 1_000_000 + 50

    at = fv.judge(_frame(2, started_ticks=qualifies_after), change, mapping)
    after = fv.judge(_frame(2, started_ticks=qualifies_after + 1), change, mapping)

    assert change.rule.rule is Rule.CLOCK
    assert (at.counts, at.qualifies_after_ticks) == (False, qualifies_after)
    assert (after.counts, after.frame_start_ticks) == (True, qualifies_after + 1)
    assert after.fields()['rule'] == 'clock'
    assert after.fields()['qualifies_after_ticks'] == qualifies_after


def test_a_clock_rule_frame_whose_start_cannot_be_known_does_not_count():
    """Nothing shows such a frame started late enough: without the mapping
    or the frame's ``Timestamp`` chunk it is not taken."""
    fv = _clocked('gain', delay_us=0.0)
    change = fv.invalidate('gain', began_ns=None)
    late = 10**15

    no_mapping = fv.judge(_frame(2, started_ticks=late), change, None)
    no_chunk = fv.judge(_frame(2), change, _mapping_at(change.written_ns))

    assert (no_mapping.counts, no_mapping.qualifies_after_ticks) == (False, None)
    assert (no_chunk.counts, no_chunk.frame_start_ticks) == (False, None)


def test_a_clock_rule_move_is_judged_from_its_settle():
    settled = threading.Event()
    fv = _clocked('z_move', delay_us=0.0, settle_check=lambda source: settled.is_set())
    fv.invalidate('z_move', began_ns=None)
    late = 10**15

    moving = fv.judge(_frame(2, started_ticks=late), fv.changes()['z_move'], None)
    settled.set()
    at_ns = host_clock.steady_ns()
    fv.motion_settled('z_move', at_ns, moving_seen_ns=None)
    change = fv.changes()['z_move']
    mapping = _mapping_at(at_ns)
    before = fv.judge(_frame(2, started_ticks=1_000_000), change, mapping)
    after = fv.judge(_frame(2, started_ticks=1_000_000 + 51), change, mapping)

    assert (moving.counts, moving.qualifies_after_ticks) == (False, None)
    assert (before.counts, after.counts) == (False, True)


def test_an_unmeasured_move_on_a_clock_camera_counts_a_frame_started_after_its_stop():
    """A camera with a clock latch and no measured delay for a move: a frame
    that started exposing before the axis stopped, though it arrived after,
    does not count for the move; one that started after the stop does."""
    settled = threading.Event()
    fv = FrameValidity(lambda: 0)
    fv.set_settle_check(lambda source: settled.is_set())
    fv.use_rules(FrameValidity.rules_from_record(None, has_clock_latch=True))
    fv.invalidate('z_move', began_ns=None)
    settled.set()
    stop_ns = host_clock.steady_ns()
    fv.motion_settled('z_move', stop_ns, moving_seen_ns=None)
    change = fv.changes()['z_move']
    mapping = _mapping_at(stop_ns)

    during = fv.judge(_frame(3, arrived_ns=stop_ns + 10**8, started_ticks=999_000), change, mapping)
    after = fv.judge(
        _frame(4, arrived_ns=stop_ns + 10**8, started_ticks=1_000_051), change, mapping
    )

    assert change.rule.rule is Rule.CLOCK
    assert (during.counts, after.counts) == (False, True)


def test_a_rule_states_a_whole_count_and_its_delay_under_the_clock_only():
    """Every rule carries a whole count; the clock rule a delay and the run
    behind it; the credit rule no delay, its provenance the run that measured
    its count or None for a shipped one."""
    wrong = {
        'a clock rule without a delay': (Rule.CLOCK, 2, None, _PROVENANCE),
        'a clock rule without provenance': (Rule.CLOCK, 2, 1.0, None),
        'a credit rule with a delay': (Rule.CREDIT, 2, 1.0, None),
        'a count of frames that is not whole': (Rule.CREDIT, 2.5, None, None),
        'a negative count': (Rule.CREDIT, -1, None, None),
        'a count that is a bool': (Rule.CREDIT, True, None, None),
    }
    accepted = []
    for name, fields in wrong.items():
        try:
            FrameRule(*fields)
            accepted.append(name)
        except ValueError:
            pass

    assert accepted == []
    assert FrameRule(Rule.CREDIT, 4, None, _PROVENANCE).provenance == _PROVENANCE


def test_a_move_counts_the_third_frame_to_arrive_after_its_stop():
    """No clock: the frames that arrive throughout a long move count nothing
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
    moving = fv.judge(_frame(delivered + 1), fv.changes()['z_move'], None)
    settled.set()
    stop_ns = host_clock.steady_ns()
    fv.motion_settled('z_move', stop_ns, moving_seen_ns=None)
    after_stop = [arrive() for _ in range(3)]
    stamped = fv.changes()['z_move']

    assert moving.counts is False and moving.after_in_force_ms is None
    assert [
        fv.judge(_frame(seq, arrived_ns=stop_ns + seq), stamped, None).counts for seq in after_stop
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
    change = imaging.frame_validity.changes()['gain']
    assert arrived.wait(5.0), 'no frame arrived after the gain write'
    imaging.remove_frame_listener(on_frame)
    imaging.get_image_from_buffer()

    with caplog.at_level(logging.INFO, logger='LVP.api'):
        record = imaging.capture_still(accept_dark=True).record

    gain = _judgment(record, 'gain')
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
    sim_scope.camera.set_gain_db(sim_scope.camera.read_gain_db() + 1.0)

    with caplog.at_level(logging.INFO, logger='LVP.api'):
        record = imaging.capture_still(accept_dark=True).record

    gain = _judgment(record, 'gain')
    assert gain.counts
    assert [credit.by for credit in gain.change.credits] == [Counter.DRAIN] * gain.change.skip
    line = next(r.getMessage() for r in caplog.records if 'capture_and_wait: frame' in r.message)
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
    for _ in range(imaging.frame_validity.changes()['gain'].skip + 3):
        if imaging.frame_validity.changes()['gain'].remaining == 0:
            break
        imaging.get_image(force_new_capture=True)

    with caplog.at_level(logging.INFO, logger='LVP.api'):
        record = imaging.capture_still(accept_dark=True).record

    gain = _judgment(record, 'gain')
    assert gain.counts
    assert [credit.by for credit in gain.change.credits] == [Counter.GRAB] * gain.change.skip
    line = next(r.getMessage() for r in caplog.records if 'capture_and_wait: frame' in r.message)
    assert ','.join(f'{c.seq}:grab' for c in gain.change.credits) in line


def test_a_move_is_in_force_when_its_axis_settles(sim_scope):
    motion = sim_scope.motion
    motion.home('Z')
    motion.move_z(motion.get_current_position('Z') + 100.0)

    record_change = sim_scope.imaging.frame_validity.changes()['z_move']
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

    gain = next(j for j in imaging.judge_frame(later[0], None) if j.change.source == 'gain')

    assert gain.frame_seq == later[0].seq
    assert gain.frames_after == later[0].seq - at_seq
    # Nothing counted toward the write before this frame: it does not count.
    assert gain.counts is False


def test_judging_something_that_is_not_a_frame_is_refused(sim_scope):
    with pytest.raises(ArgumentTypeRefusedError) as refused:
        sim_scope.imaging.judge_frame(_PIXELS, None)
    assert refused.value.reason == 'wrong_argument_type'
