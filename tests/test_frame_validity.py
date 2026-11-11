# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for FrameValidity module."""

import threading
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from _lumascope.drivers import host_clock
from _lumascope.modules.exceptions import CaptureFailedError
from _lumascope.modules.frame_validity import Counter, FrameValidity


class _Frames:
    """Stands in for the camera's delivered-frame count.

    FrameValidity asks this how many frames have arrived whenever a write
    is recorded, and is handed an ordinal for each frame counted. Driving
    both from one object is what lets a test say "this frame arrived
    before that write" exactly, with no clock involved.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.delivered = 0

    def __call__(self) -> int:
        """How many frames have arrived so far."""
        with self._lock:
            return self.delivered

    def arrive(self) -> int:
        """A new frame arrives; return its ordinal.

        Locked because the thread-safety tests call this concurrently, and
        two threads handed the SAME ordinal would be deduped into one frame
        -- the test would then be measuring its own helper.
        """
        with self._lock:
            self.delivered += 1
            return self.delivered


# Shared by every test in the file. Only ever counts up, so one test's
# frames can never look older than an earlier test's write.
_frames = _Frames()


def _f() -> int:
    """The ordinal of a frame arriving now."""
    return _frames.arrive()


def _owed_for(fv: FrameValidity, source: str) -> int:
    """The frames ``source``'s change is still owed: ``owed`` with every
    other source excluded."""
    others = tuple(s for s in FrameValidity.SKIP_FRAMES if s != source)
    return fv.owed(exclude_sources=others).frames


class TestBasicInvalidation:
    """Core invalidation and frame counting behavior, under the credit rule."""

    def test_initially_nothing_is_owed(self):
        fv = FrameValidity(_frames)
        assert fv.owed().frames == 0

    def test_invalidate_owes_the_sources_count(self):
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)
        assert fv.owed().frames == 2

    def test_counting_frames_meets_the_count(self):
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)
        fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.owed().frames == 1
        fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.owed().frames == 0


class TestMultipleSources:
    """Multiple concurrent invalidation sources."""

    def test_sources_at_different_times(self):
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)  # needs 2 more frames from frame 0
        fv.count_frame(_f(), by=Counter.GRAB)  # frame 1
        fv.invalidate('gain', began_ns=None)  # needs 2 more frames from frame 1
        # led needs 1 more (target=2), gain needs 2 more (target=3)
        assert fv.owed().frames == 2
        fv.count_frame(_f(), by=Counter.GRAB)  # frame 2 -- led settles
        assert fv.owed().frames == 1
        fv.count_frame(_f(), by=Counter.GRAB)  # frame 3 -- gain settles
        assert fv.owed().frames == 0

    def test_reinvalidate_same_source(self):
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)
        fv.count_frame(_f(), by=Counter.GRAB)
        fv.invalidate('led', began_ns=None)  # re-invalidate resets the skip count
        assert fv.owed().frames == 2
        fv.count_frame(_f(), by=Counter.GRAB)
        fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.owed().frames == 0

    def test_all_known_sources(self):
        fv = FrameValidity(_frames)
        for source in FrameValidity.SKIP_FRAMES:
            fv.invalidate(source, began_ns=None)
        max_skip = max(FrameValidity.SKIP_FRAMES.values())
        assert fv.owed().frames == max_skip
        for _ in range(max_skip):
            fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.owed().frames == 0


class TestExcludeSources:
    """An excluded source is not owed; the others still are.

    Asks whether a frame counts ignoring a change, so the frame can predate
    it: the display's slow-frame check leaves the motion sources out
    (ui/scope_display.py `_camera_settled`). A frame that is measured or
    recorded excludes nothing; autofocus waits out each Z move.
    """

    def test_exclude_z_move(self):
        fv = FrameValidity(_frames)
        fv.invalidate('z_move', began_ns=None)
        assert fv.owed().frames == 2
        assert fv.owed(exclude_sources=('z_move',)).frames == 0

    def test_exclude_z_move_with_other_pending(self):
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)
        fv.count_frame(_f(), by=Counter.GRAB)
        fv.invalidate('z_move', began_ns=None)  # z_move invalidated 1 frame later than led
        assert fv.owed(exclude_sources=('z_move',)).frames == 1
        fv.count_frame(_f(), by=Counter.GRAB)
        # LED settled (target=2, counter=2), z_move still pending (target=3)
        assert fv.owed(exclude_sources=('z_move',)).frames == 0
        assert fv.owed().frames == 1  # z_move still owed overall
        fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.owed().frames == 0  # now everything settled


class TestUnknownSource:
    """A source without a skip count is refused: a silent default would be a
    count nobody chose."""

    def test_unknown_source_is_refused(self):
        fv = FrameValidity(_frames)
        with pytest.raises(ValueError, match='custom_thing'):
            fv.invalidate('custom_thing', began_ns=None)
        assert fv.owed().frames == 0


class TestEachSourceIsOwedItsOwn:
    """What each source is owed, and the frame counter."""

    def test_each_source_is_owed_its_count(self):
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)
        fv.invalidate('exposure', began_ns=None)
        assert _owed_for(fv, 'led') == 2
        assert _owed_for(fv, 'exposure') == 3

    def test_frame_counter_tracks_grabs(self):
        fv = FrameValidity(_frames)
        assert fv.frame_counter == 0
        fv.count_frame(_f(), by=Counter.GRAB)
        fv.count_frame(_f(), by=Counter.GRAB)
        fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.frame_counter == 3

    def test_a_move_is_counted_from_its_stop(self):
        """A frame that arrives while the axis moves was exposed during the
        move, so none counts toward it: the move's count starts where its
        stop is stamped, as a write's starts where the write returns, and
        a later stamp of the axis settled starts it again."""
        fv = FrameValidity(_frames)
        moving = True
        fv.set_settle_check(lambda source: not moving)
        fv.invalidate('z_move', began_ns=None)
        for _ in range(500):
            fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.owed().frames == 2

        moving = False
        fv.motion_settled('z_move', host_clock.steady_ns(), moving_seen_ns=None)
        assert fv.owed().frames == 2
        fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.owed().frames == 1
        fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.owed().frames == 0

        fv.motion_settled('z_move', host_clock.steady_ns(), moving_seen_ns=None)
        assert fv.owed().frames == 2


class TestThreadSafety:
    """Basic thread safety verification."""

    def test_concurrent_invalidate_and_count(self):
        fv = FrameValidity(_frames)
        errors = []

        def invalidator():
            try:
                for _ in range(1000):
                    fv.invalidate('led', began_ns=None)
            except Exception as e:
                errors.append(e)

        def counter():
            try:
                for _ in range(1000):
                    fv.count_frame(_f(), by=Counter.GRAB)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=invalidator),
            threading.Thread(target=counter),
            threading.Thread(target=invalidator),
            threading.Thread(target=counter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert fv.frame_counter == 2000

    def test_concurrent_reads(self):
        """Readers never raise while a writer adds and settles sources.

        The writer is what makes the readers' lock load-bearing: invalidate
        records a source's first change, so a reader iterating the changes
        without the lock sees them change size under it.
        """
        fv = FrameValidity(_frames)
        sources = ('led', 'gain', 'exposure', 'pixel_format', 'frame_size', 'binning')
        errors = []
        readers_done = threading.Event()

        def writer():
            try:
                i = 0
                while not readers_done.is_set():
                    fv.invalidate(sources[i % len(sources)], began_ns=None)
                    fv.count_frame(_f(), by=Counter.GRAB)
                    i += 1
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(5000):
                    _ = fv.owed()
                    _ = fv.invalidation_counts
                    _ = fv.frame_counter
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        readers = [threading.Thread(target=reader) for _ in range(4)]
        # At the default 5 ms switch interval a thread rarely loses the GIL
        # inside one short dict iteration, and an unlocked reader passes by
        # luck; a microsecond interval makes the interleaving the lock guards
        # against happen on every run. Restored before any assertion.
        prior_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            writer_thread.start()
            for t in readers:
                t.start()
            for t in readers:
                t.join()
        finally:
            readers_done.set()
            writer_thread.join()
            sys.setswitchinterval(prior_interval)

        assert errors == []
        assert fv.frame_counter > 0  # the writer ran


class TestSetTarget:
    """set_target(source, value) records the requested value for chunk-match;
    None leaves the source with no target, as target() reads it."""

    def test_set_target_none_clears(self):
        fv = FrameValidity(_frames)
        fv.set_target('gain', 5.0)
        assert fv.target('gain') == 5.0
        fv.set_target('gain', None)
        assert fv.target('gain') is None


class TestChunkMatch:
    """chunk_match(source, chunk_value, tolerance) for diagnostic checks."""

    def test_match_within_default_tolerance(self):
        fv = FrameValidity(_frames)
        fv.set_target('gain', 5.0)
        # gain default tolerance is 0.001 dB (bench-measured ~20x headroom
        # over Pylon SDK genicam round-trip noise)
        assert fv.chunk_match('gain', 5.0005) is True
        assert fv.chunk_match('gain', 4.9995) is True

    def test_no_match_outside_default_tolerance(self):
        fv = FrameValidity(_frames)
        fv.set_target('gain', 5.0)
        # 0.2 dB delta is well outside the 0.001 dB default
        assert fv.chunk_match('gain', 5.2) is False
        assert fv.chunk_match('gain', 4.8) is False

    def test_no_match_when_target_unset(self):
        fv = FrameValidity(_frames)
        # No set_target call
        assert fv.chunk_match('gain', 5.0) is False

    def test_no_match_when_chunk_value_none(self):
        fv = FrameValidity(_frames)
        fv.set_target('gain', 5.0)
        assert fv.chunk_match('gain', None) is False

    def test_explicit_tolerance_override(self):
        fv = FrameValidity(_frames)
        fv.set_target('gain', 5.0)
        # Default 0.001 would reject; explicit 0.5 accepts
        assert fv.chunk_match('gain', 5.4, tolerance=0.5) is True
        assert fv.chunk_match('gain', 5.4, tolerance=0.1) is False

    def test_exposure_microseconds_default(self):
        fv = FrameValidity(_frames)
        fv.set_target('exposure', 14530.0)
        # exposure default tolerance is 2 us (bench-measured: integer-us
        # set values are bit-exact at the SDK layer)
        assert fv.chunk_match('exposure', 14530.5) is True
        assert fv.chunk_match('exposure', 14531.5) is True
        assert fv.chunk_match('exposure', 14533.0) is False


class TestAutoGainSettle:
    """Hardware continuous-AG settling as a deterministic skip-frame source.

    auto_gain is a measure of the instrumentation (frames for hardware AG to
    settle against the lit scene), not of the sample. It clears by frame count
    only -- never early from frame content -- which is what keeps it in the
    settle-tracker layer and reusable by a future software-AG loop.
    """

    @pytest.fixture(autouse=True)
    def _restore_skip_frames(self):
        """Save and restore SKIP_FRAMES so the override test doesn't leak."""
        original = dict(FrameValidity.SKIP_FRAMES)
        yield
        FrameValidity.SKIP_FRAMES.clear()
        FrameValidity.SKIP_FRAMES.update(original)

    def test_decrements_to_valid_by_frame_count(self):
        fv = FrameValidity(_frames)
        skip = FrameValidity.SKIP_FRAMES['auto_gain']
        fv.invalidate('auto_gain', began_ns=None)
        for _ in range(skip):
            assert fv.owed().frames > 0
            fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.owed().frames == 0

    def test_not_a_motion_source(self):
        """auto_gain settles on count alone; no settle-callback gating."""
        assert 'auto_gain' not in FrameValidity.MOTION_SOURCES


class TestCaptureTimeChunkVerification:
    """The grab with ``verify_chunk_targets=True`` passes over a frame whose
    chunk metadata disagrees with the recorded gain / exposure target.

    Readiness is settled by frame count for every camera; the chunk check
    runs after it and may only REJECT a frame. A chunk reports the register
    in force when the frame was tagged, not the light integrated into it, so
    a chunk that disagrees proves the frame predates the write, while one
    that agrees proves nothing (FrameValidity.CHUNK_VALIDATABLE_SOURCES).

    The simulated camera tags each frame with the exposure and gain it was
    made under; a frame made before a write and stored after it
    (``simulation.delay_delivery``) carries the old values.
    """

    def test_stale_chunk_frame_rejected_until_match(self, sim_scope):
        """Frames made before an exposure write are passed over; the grab
        returns the first whose chunk names the new exposure."""
        sim_scope.simulation.delay_delivery(3)
        # The first frame stored after the delay says the backlog stands: from
        # here, three frames made before the write are on their way.
        sim_scope.imaging.get_image(force_new_capture=True)
        new_ms = sim_scope.camera.exposure_ms / 2
        sim_scope.camera.set_exposure_ms(new_ms)

        # The private grab, for the frame it returns: the public get_image
        # hands back the pixels alone.
        grabbed = sim_scope.imaging._get_image_impl(
            force_new_capture=True,
            verify_chunk_targets=True,
            timeout_s=5.0,
        )
        assert grabbed.frame.chunks['ExposureTime'] == pytest.approx(new_ms * 1000.0)

    def test_persistent_stale_chunks_end_chunk_rejected_naming_the_setting(self, sim_scope):
        # Every frame names the exposure the camera runs at; a target it
        # never reaches makes each one stale.
        assert sim_scope.camera.exposure_ms != 3.73, 'precondition'
        sim_scope.imaging._frame_validity.set_target('exposure', 3730.0)

        with pytest.raises(CaptureFailedError) as refused:
            sim_scope.imaging.get_image(
                force_new_capture=True, verify_chunk_targets=True, timeout_s=0.5
            )
        assert (refused.value.reason, refused.value.source) == ('chunk_rejected', 'exposure')

    def test_no_targets_means_no_verification(self, sim_scope):
        """Hardware auto-gain clears targets; chunk mismatch must not block."""
        fv = sim_scope.imaging._frame_validity
        fv.set_target('gain', None)
        fv.set_target('exposure', None)

        image = sim_scope.imaging.get_image(
            force_new_capture=True, verify_chunk_targets=True, timeout_s=0.5
        )
        assert isinstance(image, np.ndarray)

    def test_chunkless_camera_skips_verification(self):
        """Cameras without chunk support rely on skip-count settling alone."""
        from _lumascope.modules.scope_session import ScopeSession
        from tests.settings_fixtures import complete_settings

        # The Classic's FX2 frames carry no chunks.
        session = ScopeSession.create(
            complete_settings(microscope='LS620'), simulate=True, warn_pre_release=False
        )
        try:
            session.scope.imaging._frame_validity.set_target('exposure', 3730.0)

            image = session.scope.imaging.get_image(
                force_new_capture=True, verify_chunk_targets=True, timeout_s=0.5
            )
        finally:
            session.shutdown()
        assert isinstance(image, np.ndarray)

    def test_default_get_image_unverified(self, sim_scope):
        """Without the flag the primitive stays ungated (live preview path)."""
        assert sim_scope.camera.exposure_ms != 3.73, 'precondition'
        sim_scope.imaging._frame_validity.set_target('exposure', 3730.0)

        image = sim_scope.imaging.get_image(force_new_capture=True, timeout_s=0.5)
        assert isinstance(image, np.ndarray)


class TestInvalidationCounts:
    """The monotone per-source invalidation history a capture compares across
    its grab window.

    Pending-state snapshots provably cannot detect a mid-window
    invalidation that frames have already settled (a poller frame plus the
    capture's own grab consume a 2-skip entry exactly); the counts exist
    so the capture has erasure-proof evidence. These pins guard the two
    properties that proof rests on: monotone growth and frame-immunity.
    """

    def test_counts_start_empty(self):
        fv = FrameValidity(_frames)
        assert fv.invalidation_counts == {}

    def test_each_invalidate_increments_its_source(self):
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)
        fv.invalidate('led', began_ns=None)
        fv.invalidate('gain', began_ns=None)
        assert fv.invalidation_counts == {'led': 2, 'gain': 1}

    def test_counting_frames_never_touches_counts(self):
        """Frames settle pending state; they must not erase the history."""
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)
        before = fv.invalidation_counts
        for _ in range(5):
            fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.owed().frames == 0  # pending fully settled
        assert fv.invalidation_counts == before  # history intact

    def test_settled_invalidation_still_visible_in_compare(self):
        """The seam the counts exist to close: invalidate, let exactly the
        skip count of frames settle it, and nothing is owed while the
        counts compare still differs."""
        fv = FrameValidity(_frames)
        counts_snap = fv.invalidation_counts

        fv.invalidate('led', began_ns=None)  # skip = 2
        fv.count_frame(_f(), by=Counter.GRAB)
        fv.count_frame(_f(), by=Counter.GRAB)

        assert fv.owed().frames == 0
        assert fv.invalidation_counts != counts_snap  # counts: caught

    def test_first_invalidation_adds_a_key_the_snapshot_lacks(self):
        """The compare spans both key sets, not shared keys only."""
        fv = FrameValidity(_frames)
        snap = fv.invalidation_counts
        fv.invalidate('turret', began_ns=None)
        assert 'turret' not in snap
        assert fv.invalidation_counts != snap

    def test_property_returns_a_copy(self):
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)
        held = fv.invalidation_counts
        fv.invalidate('led', began_ns=None)
        assert held == {'led': 1}
        assert fv.invalidation_counts == {'led': 2}


class TestOwedMoving:
    """``owed().moving``, the capture deadline's pause: the motion sources
    owed a frame because their axis reads moving."""

    def test_empty_without_callback(self):
        fv = FrameValidity(_frames)
        fv.invalidate('z_move', began_ns=None)
        assert fv.owed().moving == frozenset()

    def test_reports_pending_unsettled_motion(self):
        fv = FrameValidity(_frames)
        fv.set_settle_check(lambda source: False)
        fv.invalidate('z_move', began_ns=None)
        assert fv.owed().moving == {'z_move'}

    def test_settled_motion_not_reported(self):
        fv = FrameValidity(_frames)
        fv.set_settle_check(lambda source: True)
        fv.invalidate('z_move', began_ns=None)
        assert fv.owed().moving == frozenset()

    def test_non_motion_sources_never_reported(self):
        fv = FrameValidity(_frames)
        fv.set_settle_check(lambda source: False)
        fv.invalidate('led', began_ns=None)
        assert fv.owed().moving == frozenset()

    def test_exclude_sources_honored(self):
        """A source the caller excludes from readiness is not reported as
        moving either: the capture deadline never pauses on a move the
        caller left out of its wait."""
        fv = FrameValidity(_frames)
        fv.set_settle_check(lambda source: False)
        fv.invalidate('z_move', began_ns=None)
        assert fv.owed(exclude_sources=('z_move',)).moving == frozenset()

    def test_a_move_in_force_is_moving_again_while_its_axis_reads_moving(self):
        """A move is owed a frame whenever its axis reads moving, its count
        met or not, so the pause is read from the same changes: a move in
        force with its count met, then a several-move command opened, which
        reads every axis as moving."""
        fv = FrameValidity(_frames)
        settled = {'now': True}
        fv.set_settle_check(lambda source: settled['now'])
        fv.invalidate('z_move', began_ns=None)
        fv.motion_settled('z_move', host_clock.steady_ns(), moving_seen_ns=None)
        for _ in range(FrameValidity.SKIP_FRAMES['z_move'] + 1):
            fv.count_frame(_f(), by=Counter.GRAB)
        assert (fv.owed().frames, fv.owed().moving) == (0, frozenset())

        settled['now'] = False

        assert fv.owed().moving == {'z_move'}
        assert fv.owed().frames == 1


class TestFramesOlderThanTheWriteDoNotCount:
    """A frame already delivered when a register was written cannot show the
    new state, so it must not retire that write's skip count.

    This is the defect behind captures returning a frame from before the
    change that requested them: the live preview grabs a frame, the write
    lands, and the preview's count of that older frame spends a budget it
    was never part of.

    The comparison is on arrival ordinals rather than clock readings so it
    survives the clock moving: wall time runs backwards across a DST
    fall-back, an NTP correction or a host resume, and a wait measured
    against a timestamp that jumped forward would never complete at all.
    """

    def test_a_frame_delivered_before_the_write_does_not_credit(self):
        fv = FrameValidity(_frames)
        already_delivered = _f()
        fv.invalidate('led', began_ns=None)
        fv.count_frame(already_delivered, by=Counter.GRAB)
        assert fv.owed().frames == 2

    def test_frames_after_the_write_still_settle_it(self):
        fv = FrameValidity(_frames)
        already_delivered = _f()
        fv.invalidate('led', began_ns=None)
        fv.count_frame(already_delivered, by=Counter.GRAB)
        fv.count_frame(_f(), by=Counter.GRAB)
        fv.count_frame(_f(), by=Counter.GRAB)
        assert fv.owed().frames == 0

    def test_each_source_is_judged_against_its_own_write(self):
        """led is written first, then a frame arrives, then gain is written.
        That frame is post-led and pre-gain: it credits led alone."""
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)
        between = _f()
        fv.invalidate('gain', began_ns=None)
        fv.count_frame(between, by=Counter.GRAB)
        assert _owed_for(fv, 'led') == 1
        assert _owed_for(fv, 'gain') == 2

    def test_a_frame_counted_again_after_a_newer_one_does_not_credit_twice(self):
        """The drain counts two frames, and the preview's count of the first,
        grabbed before the second arrived, lands last. Two frames have
        arrived, so exposure's three-frame wait is not met: crediting the
        late count would let a capture take a frame one short of its wait."""
        fv = FrameValidity(_frames)
        fv.invalidate('exposure', began_ns=None)
        first, second = _f(), _f()
        fv.count_frame(first, by=Counter.DRAIN)
        fv.count_frame(second, by=Counter.DRAIN)
        fv.count_frame(first, by=Counter.PREVIEW)
        assert fv.owed().frames == 1

    def test_an_ordinal_is_required(self):
        """A caller with no ordinal cannot be told apart from one holding a
        stale frame, so there is no safe way to credit it."""
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)
        with pytest.raises(TypeError):
            fv.count_frame(by=Counter.GRAB)

    def test_who_counted_is_required(self):
        """A frame counted with no counter named could be the preview's, and a
        capture's record of its wait would then misname who met it."""
        fv = FrameValidity(_frames)
        fv.invalidate('led', began_ns=None)
        with pytest.raises(TypeError):
            fv.count_frame(1)

    def test_a_frame_source_is_required(self):
        """Without one, a write cannot be placed in the frame stream and
        every frame would credit every pending source -- the defect this
        class exists to prevent, reintroduced by an omission."""
        with pytest.raises(TypeError):
            FrameValidity()


def test_no_public_attribute_of_the_scope_hands_out_the_store(sim_scope):
    """The store goes to no caller: only the changers inside the API record
    a change or count a frame, so no caller can count a frame of its own
    and make a stale frame count. A caller asks the imaging API's reads
    (``changes_in_force``, ``changes_in_force_limit_s``, ``judge_frame``,
    ``recorded_changes``). Walks the scope's public attributes and each
    one's own, naming every one that is the store."""
    offenders = []
    for name, value in vars(sim_scope).items():
        if name.startswith('_'):
            continue
        if isinstance(value, FrameValidity):
            offenders.append(f'scope.{name}')
        for inner_name, inner in getattr(value, '__dict__', {}).items():
            if not inner_name.startswith('_') and isinstance(inner, FrameValidity):
                offenders.append(f'scope.{name}.{inner_name}')
    assert offenders == [], f'the frame-validity store is handed out at {offenders}'
