"""Tests for FrameValidity module."""

import threading
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from modules.frame_validity import FrameValidity


class TestBasicInvalidation:
    """Core invalidation and frame counting behavior."""

    def test_initially_valid(self):
        fv = FrameValidity()
        assert fv.is_valid
        assert fv.frames_until_valid() == 0

    def test_invalidate_makes_invalid(self):
        fv = FrameValidity()
        fv.invalidate('led')
        assert not fv.is_valid
        assert fv.frames_until_valid() == 2

    def test_counting_frames_restores_validity(self):
        fv = FrameValidity()
        fv.invalidate('led')
        fv.count_frame()
        assert not fv.is_valid
        assert fv.frames_until_valid() == 1
        fv.count_frame()
        assert fv.is_valid
        assert fv.frames_until_valid() == 0

    def test_extra_frames_after_valid(self):
        fv = FrameValidity()
        fv.invalidate('led')
        for _ in range(5):
            fv.count_frame()
        assert fv.is_valid
        assert fv.frames_until_valid() == 0

    def test_reset_clears_state(self):
        fv = FrameValidity()
        fv.invalidate('led')
        fv.count_frame()
        fv.reset()
        assert fv.is_valid
        assert fv.frame_counter == 0
        assert fv.pending_sources == {}

    def test_no_pending_sources_initially(self):
        fv = FrameValidity()
        assert fv.pending_sources == {}
        assert fv.frame_counter == 0


class TestMultipleSources:
    """Multiple concurrent invalidation sources."""

    def test_two_sources_same_time(self):
        fv = FrameValidity()
        fv.invalidate('led')
        fv.invalidate('gain')
        assert fv.frames_until_valid() == 2
        fv.count_frame()
        fv.count_frame()
        assert fv.is_valid

    def test_sources_at_different_times(self):
        fv = FrameValidity()
        fv.invalidate('led')       # needs 2 more frames from frame 0
        fv.count_frame()            # frame 1
        fv.invalidate('gain')      # needs 2 more frames from frame 1
        # led needs 1 more (target=2), gain needs 2 more (target=3)
        assert fv.frames_until_valid() == 2
        fv.count_frame()            # frame 2 — led settles
        assert fv.frames_until_valid() == 1
        fv.count_frame()            # frame 3 — gain settles
        assert fv.is_valid

    def test_reinvalidate_same_source(self):
        fv = FrameValidity()
        fv.invalidate('led')
        fv.count_frame()
        fv.invalidate('led')  # re-invalidate resets the skip count
        assert fv.frames_until_valid() == 2
        fv.count_frame()
        fv.count_frame()
        assert fv.is_valid

    def test_all_known_sources(self):
        fv = FrameValidity()
        for source in FrameValidity.SKIP_FRAMES:
            fv.invalidate(source)
        assert fv.frames_until_valid() == 2
        fv.count_frame()
        fv.count_frame()
        assert fv.is_valid
        assert fv.pending_sources == {}

    def test_rapid_invalidation_between_frames(self):
        """Invalidate multiple times before any frame is grabbed."""
        fv = FrameValidity()
        fv.invalidate('led')
        fv.invalidate('gain')
        fv.invalidate('exposure')
        fv.invalidate('xy_move')
        fv.invalidate('z_move')
        # All invalidated at frame 0, all need 2 frames
        assert fv.frames_until_valid() == 2
        fv.count_frame()
        fv.count_frame()
        assert fv.is_valid


class TestExcludeSources:
    """Exclude sources for autofocus-style usage."""

    def test_exclude_z_move(self):
        fv = FrameValidity()
        fv.invalidate('z_move')
        assert not fv.is_valid
        assert fv.is_valid_for(exclude_sources=('z_move',))
        assert fv.frames_until_valid(exclude_sources=('z_move',)) == 0

    def test_exclude_z_move_with_other_pending(self):
        fv = FrameValidity()
        fv.invalidate('led')
        fv.count_frame()
        fv.invalidate('z_move')  # z_move invalidated 1 frame later than led
        assert not fv.is_valid_for(exclude_sources=('z_move',))
        assert fv.frames_until_valid(exclude_sources=('z_move',)) == 1
        fv.count_frame()
        # LED settled (target=2, counter=2), z_move still pending (target=3)
        assert fv.is_valid_for(exclude_sources=('z_move',))
        assert not fv.is_valid  # z_move still pending overall
        fv.count_frame()
        assert fv.is_valid  # now everything settled

    def test_exclude_multiple_sources(self):
        fv = FrameValidity()
        fv.invalidate('z_move')
        fv.invalidate('xy_move')
        fv.invalidate('led')
        assert fv.frames_until_valid(exclude_sources=('z_move', 'xy_move')) == 2
        fv.count_frame()
        fv.count_frame()
        assert fv.is_valid_for(exclude_sources=('z_move', 'xy_move'))

    def test_exclude_nonexistent_source(self):
        fv = FrameValidity()
        fv.invalidate('led')
        assert fv.frames_until_valid(exclude_sources=('nonexistent',)) == 2

    def test_exclude_all_pending(self):
        fv = FrameValidity()
        fv.invalidate('z_move')
        assert fv.frames_until_valid(exclude_sources=('z_move',)) == 0
        assert fv.is_valid_for(exclude_sources=('z_move',))


class TestUnknownSource:
    """Unknown sources use default skip count."""

    def test_unknown_source_uses_default(self):
        fv = FrameValidity()
        fv.invalidate('custom_thing')
        assert fv.frames_until_valid() == FrameValidity.DEFAULT_SKIP_FRAMES

    def test_unknown_source_settles(self):
        fv = FrameValidity()
        fv.invalidate('something_new')
        for _ in range(FrameValidity.DEFAULT_SKIP_FRAMES):
            fv.count_frame()
        assert fv.is_valid


class TestPendingSourcesDebug:
    """Debug properties for introspection."""

    def test_pending_sources_tracking(self):
        fv = FrameValidity()
        fv.invalidate('led')
        fv.invalidate('gain')
        pending = fv.pending_sources
        assert 'led' in pending
        assert 'gain' in pending

    def test_pending_cleared_after_settle(self):
        fv = FrameValidity()
        fv.invalidate('led')
        fv.count_frame()
        fv.count_frame()
        assert fv.pending_sources == {}

    def test_frame_counter_tracks_grabs(self):
        fv = FrameValidity()
        assert fv.frame_counter == 0
        fv.count_frame()
        fv.count_frame()
        fv.count_frame()
        assert fv.frame_counter == 3

    def test_frame_counter_not_affected_by_invalidation(self):
        fv = FrameValidity()
        fv.count_frame()
        fv.count_frame()
        fv.invalidate('led')
        assert fv.frame_counter == 2


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_invalidate_then_immediate_check(self):
        fv = FrameValidity()
        fv.invalidate('led')
        assert fv.frames_until_valid() == 2
        assert not fv.is_valid

    def test_count_without_invalidation(self):
        fv = FrameValidity()
        fv.count_frame()
        fv.count_frame()
        assert fv.is_valid
        assert fv.frame_counter == 2

    def test_invalidate_after_many_frames(self):
        fv = FrameValidity()
        for _ in range(100):
            fv.count_frame()
        fv.invalidate('led')
        assert fv.frames_until_valid() == 2
        assert fv.frame_counter == 100

    def test_settle_during_counting(self):
        """Sources auto-clear from pending when count_frame crosses threshold."""
        fv = FrameValidity()
        fv.invalidate('led')  # target = 0 + 2 = 2
        fv.invalidate('gain')  # target = 0 + 2 = 2
        fv.count_frame()  # frame 1
        assert len(fv.pending_sources) == 2
        fv.count_frame()  # frame 2 — both settle
        assert len(fv.pending_sources) == 0

    def test_frames_until_valid_never_negative(self):
        fv = FrameValidity()
        fv.invalidate('led')
        for _ in range(10):
            fv.count_frame()
        assert fv.frames_until_valid() == 0


class TestThreadSafety:
    """Basic thread safety verification."""

    def test_concurrent_invalidate_and_count(self):
        fv = FrameValidity()
        errors = []

        def invalidator():
            try:
                for _ in range(1000):
                    fv.invalidate('led')
            except Exception as e:
                errors.append(e)

        def counter():
            try:
                for _ in range(1000):
                    fv.count_frame()
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
        fv = FrameValidity()
        fv.invalidate('led')
        errors = []

        def reader():
            try:
                for _ in range(1000):
                    _ = fv.is_valid
                    _ = fv.frames_until_valid()
                    _ = fv.pending_sources
                    _ = fv.frame_counter
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
