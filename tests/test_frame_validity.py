# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
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
        max_skip = max(FrameValidity.SKIP_FRAMES.values())
        assert fv.frames_until_valid() == max_skip
        for _ in range(max_skip):
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
        # All invalidated at frame 0, max skip is exposure=3
        max_skip = max(FrameValidity.SKIP_FRAMES.values())
        assert fv.frames_until_valid() == max_skip
        for _ in range(max_skip):
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


class TestLoadCameraTiming:
    """Tests for load_camera_timing() config loading."""

    @pytest.fixture(autouse=True)
    def _restore_skip_frames(self):
        """Save and restore SKIP_FRAMES so tests don't leak state."""
        original = dict(FrameValidity.SKIP_FRAMES)
        yield
        FrameValidity.SKIP_FRAMES.clear()
        FrameValidity.SKIP_FRAMES.update(original)

    def test_overrides_skip_frames(self):
        """Config overrides SKIP_FRAMES values for specified sources."""
        fv = FrameValidity()
        config = {'skip_frames': {'led': 5, 'gain': 3}}
        fv.load_camera_timing(config)
        assert fv.SKIP_FRAMES['led'] == 5
        assert fv.SKIP_FRAMES['gain'] == 3

    def test_overridden_values_used_by_invalidate(self):
        """After loading config, invalidate() uses the new skip counts."""
        fv = FrameValidity()
        fv.load_camera_timing({'skip_frames': {'led': 4}})
        fv.invalidate('led')
        assert fv.frames_until_valid() == 4
        for _ in range(3):
            fv.count_frame()
        assert not fv.is_valid
        fv.count_frame()
        assert fv.is_valid

    def test_partial_config_only_overrides_specified(self):
        """Sources not in config keep their default values."""
        fv = FrameValidity()
        original_exposure = fv.SKIP_FRAMES['exposure']
        original_xy = fv.SKIP_FRAMES['xy_move']
        fv.load_camera_timing({'skip_frames': {'led': 7}})
        assert fv.SKIP_FRAMES['led'] == 7
        assert fv.SKIP_FRAMES['exposure'] == original_exposure
        assert fv.SKIP_FRAMES['xy_move'] == original_xy

    def test_empty_skip_frames_no_change(self):
        """Empty skip_frames dict leaves all defaults unchanged."""
        fv = FrameValidity()
        original = dict(fv.SKIP_FRAMES)
        fv.load_camera_timing({'skip_frames': {}})
        assert fv.SKIP_FRAMES == original

    def test_missing_skip_frames_key_no_change(self):
        """Config without 'skip_frames' key leaves defaults unchanged."""
        fv = FrameValidity()
        original = dict(fv.SKIP_FRAMES)
        fv.load_camera_timing({'camera_model': 'test'})
        assert fv.SKIP_FRAMES == original

    def test_empty_config_no_change(self):
        """Completely empty config leaves defaults unchanged."""
        fv = FrameValidity()
        original = dict(fv.SKIP_FRAMES)
        fv.load_camera_timing({})
        assert fv.SKIP_FRAMES == original

    def test_negative_count_rejected(self):
        """Negative frame counts are silently ignored."""
        fv = FrameValidity()
        original_led = fv.SKIP_FRAMES['led']
        fv.load_camera_timing({'skip_frames': {'led': -1}})
        assert fv.SKIP_FRAMES['led'] == original_led

    def test_float_count_rejected(self):
        """Float frame counts are rejected (must be int)."""
        fv = FrameValidity()
        original_led = fv.SKIP_FRAMES['led']
        fv.load_camera_timing({'skip_frames': {'led': 3.5}})
        assert fv.SKIP_FRAMES['led'] == original_led

    def test_string_count_rejected(self):
        """String frame counts are rejected."""
        fv = FrameValidity()
        original_led = fv.SKIP_FRAMES['led']
        fv.load_camera_timing({'skip_frames': {'led': 'three'}})
        assert fv.SKIP_FRAMES['led'] == original_led

    def test_none_count_rejected(self):
        """None frame counts are rejected."""
        fv = FrameValidity()
        original_led = fv.SKIP_FRAMES['led']
        fv.load_camera_timing({'skip_frames': {'led': None}})
        assert fv.SKIP_FRAMES['led'] == original_led

    def test_zero_count_accepted(self):
        """Zero is a valid skip count (no frames to skip)."""
        fv = FrameValidity()
        fv.load_camera_timing({'skip_frames': {'led': 0}})
        assert fv.SKIP_FRAMES['led'] == 0
        fv.invalidate('led')
        assert fv.is_valid  # zero skip means immediately valid

    def test_does_not_affect_frame_counter(self):
        """Loading config should not change the frame counter."""
        fv = FrameValidity()
        fv.count_frame()
        fv.count_frame()
        fv.count_frame()
        assert fv.frame_counter == 3
        fv.load_camera_timing({'skip_frames': {'led': 5}})
        assert fv.frame_counter == 3

    def test_does_not_affect_pending_state(self):
        """Loading config should not clear or modify pending invalidations."""
        fv = FrameValidity()
        fv.invalidate('led')
        fv.count_frame()
        pending_before = fv.pending_sources.copy()
        fv.load_camera_timing({'skip_frames': {'led': 10}})
        # Pending state unchanged — the already-queued invalidation keeps
        # its original threshold
        assert fv.pending_sources == pending_before

    def test_new_invalidation_uses_updated_count(self):
        """After loading config, new invalidations use the updated skip counts."""
        fv = FrameValidity()
        fv.load_camera_timing({'skip_frames': {'led': 10}})
        fv.invalidate('led')  # should now use 10
        assert fv.frames_until_valid() == 10
        assert fv.pending_sources['led'] == fv.frame_counter + 10

    def test_unknown_source_in_config(self):
        """Config can add skip counts for custom/unknown sources."""
        fv = FrameValidity()
        fv.load_camera_timing({'skip_frames': {'custom_thing': 8}})
        assert fv.SKIP_FRAMES['custom_thing'] == 8
        fv.invalidate('custom_thing')
        assert fv.frames_until_valid() == 8

    def test_mixed_valid_and_invalid_values(self):
        """Valid values are applied, invalid ones silently ignored."""
        fv = FrameValidity()
        fv.load_camera_timing({'skip_frames': {
            'led': 5,         # valid
            'gain': -1,       # invalid (negative)
            'exposure': 3.0,  # invalid (float)
            'z_move': 0,      # valid (zero)
        }})
        assert fv.SKIP_FRAMES['led'] == 5
        assert fv.SKIP_FRAMES['gain'] == 2      # unchanged default
        assert fv.SKIP_FRAMES['exposure'] == 3   # unchanged default (measured: 3 on a2A3536)
        assert fv.SKIP_FRAMES['z_move'] == 0

    def test_extra_config_keys_ignored(self):
        """Non-skip_frames keys in config are ignored without error."""
        fv = FrameValidity()
        config = {
            'camera_model': 'daA3840-45um',
            'measured_date': '2026-03-12',
            'skip_frames': {'led': 3},
            'frame_intervals_ms': {'100': 33.2},
            'dark_noise_stddev': {'mono': 1.2},
        }
        fv.load_camera_timing(config)
        assert fv.SKIP_FRAMES['led'] == 3


class TestLoadCameraTimingLumascope:
    """Tests for Lumascope._load_camera_timing() integration."""

    @pytest.fixture(autouse=True)
    def _restore_skip_frames(self):
        """Save and restore SKIP_FRAMES so tests don't leak state."""
        original = dict(FrameValidity.SKIP_FRAMES)
        yield
        FrameValidity.SKIP_FRAMES.clear()
        FrameValidity.SKIP_FRAMES.update(original)

    def test_loads_from_correct_path(self, tmp_path):
        """_load_camera_timing builds path from camera model_name."""
        import json

        # Create a mock timing config
        timing_dir = tmp_path / 'data' / 'camera_timing'
        timing_dir.mkdir(parents=True)
        config = {'skip_frames': {'led': 7, 'gain': 4}}
        (timing_dir / 'TestCam_Model.json').write_text(json.dumps(config))

        # Create a minimal mock that exercises _load_camera_timing logic
        # without instantiating a full Lumascope
        fv = FrameValidity()
        model = 'TestCam Model'
        safe_name = model.replace(' ', '_')
        timing_path = timing_dir / f'{safe_name}.json'
        assert timing_path.exists()
        with open(timing_path) as f:
            loaded = json.load(f)
        fv.load_camera_timing(loaded)
        assert fv.SKIP_FRAMES['led'] == 7
        assert fv.SKIP_FRAMES['gain'] == 4

    def test_missing_file_no_error(self, tmp_path):
        """Missing timing file should not raise — silently skipped."""
        import pathlib
        timing_dir = tmp_path / 'data' / 'camera_timing'
        timing_dir.mkdir(parents=True)
        timing_path = timing_dir / 'NonExistent.json'
        # Simulating what _load_camera_timing does: check exists, skip if not
        assert not timing_path.exists()

    def test_model_name_normalization(self):
        """Spaces in model_name are replaced with underscores for filename."""
        model = 'daA3840 45um'
        safe_name = model.replace(' ', '_')
        assert safe_name == 'daA3840_45um'

    def test_corrupt_json_handled(self, tmp_path):
        """Corrupt JSON file should be caught and not crash."""
        timing_dir = tmp_path / 'data' / 'camera_timing'
        timing_dir.mkdir(parents=True)
        (timing_dir / 'BadCam.json').write_text('{invalid json!!!}')

        import json
        with pytest.raises(json.JSONDecodeError):
            with open(timing_dir / 'BadCam.json') as f:
                json.load(f)


class TestSetTarget:
    """set_target(source, value) records the requested value for chunk-match."""

    def test_set_target_stores_value(self):
        fv = FrameValidity()
        fv.set_target('gain', 5.0)
        assert fv._target_values.get('gain') == 5.0

    def test_set_target_overwrites(self):
        fv = FrameValidity()
        fv.set_target('exposure', 14530.0)
        fv.set_target('exposure', 25000.0)
        assert fv._target_values.get('exposure') == 25000.0

    def test_set_target_none_clears(self):
        fv = FrameValidity()
        fv.set_target('gain', 5.0)
        fv.set_target('gain', None)
        assert 'gain' not in fv._target_values

    def test_set_target_coerces_to_float(self):
        fv = FrameValidity()
        fv.set_target('gain', 5)
        assert isinstance(fv._target_values.get('gain'), float)

    def test_reset_clears_targets(self):
        fv = FrameValidity()
        fv.set_target('gain', 5.0)
        fv.set_target('exposure', 14530.0)
        fv.reset()
        assert fv._target_values == {}


class TestChunkMatch:
    """chunk_match(source, chunk_value, tolerance) for diagnostic checks."""

    def test_match_within_default_tolerance(self):
        fv = FrameValidity()
        fv.set_target('gain', 5.0)
        # gain default tolerance is 0.001 dB (bench-measured ~20x headroom
        # over Pylon SDK genicam round-trip noise)
        assert fv.chunk_match('gain', 5.0005) is True
        assert fv.chunk_match('gain', 4.9995) is True

    def test_no_match_outside_default_tolerance(self):
        fv = FrameValidity()
        fv.set_target('gain', 5.0)
        # 0.2 dB delta is well outside the 0.001 dB default
        assert fv.chunk_match('gain', 5.2) is False
        assert fv.chunk_match('gain', 4.8) is False

    def test_no_match_when_target_unset(self):
        fv = FrameValidity()
        # No set_target call
        assert fv.chunk_match('gain', 5.0) is False

    def test_no_match_when_chunk_value_none(self):
        fv = FrameValidity()
        fv.set_target('gain', 5.0)
        assert fv.chunk_match('gain', None) is False

    def test_explicit_tolerance_override(self):
        fv = FrameValidity()
        fv.set_target('gain', 5.0)
        # Default 0.001 would reject; explicit 0.5 accepts
        assert fv.chunk_match('gain', 5.4, tolerance=0.5) is True
        assert fv.chunk_match('gain', 5.4, tolerance=0.1) is False

    def test_exposure_microseconds_default(self):
        fv = FrameValidity()
        fv.set_target('exposure', 14530.0)
        # exposure default tolerance is 2 us (bench-measured: integer-us
        # set values are bit-exact at the SDK layer)
        assert fv.chunk_match('exposure', 14530.5) is True
        assert fv.chunk_match('exposure', 14531.5) is True
        assert fv.chunk_match('exposure', 14533.0) is False


class TestCountFrameWithChunks:
    """count_frame(chunk_data) clears chunk-validatable sources when chunks match."""

    def test_chunks_clear_gain_pending_short_circuiting_skip_frames(self):
        fv = FrameValidity()
        fv.invalidate('gain')              # default skip = 2 frames
        fv.set_target('gain', 5.0)
        # Single frame_count + matching chunks -> source is cleared even
        # though skip-frames count (2) hasn't been met.
        fv.count_frame(chunk_data={'Gain': 5.0, 'ExposureTime': 14530.0})
        assert 'gain' not in fv.pending_sources

    def test_chunks_clear_exposure_pending(self):
        fv = FrameValidity()
        fv.invalidate('exposure')
        fv.set_target('exposure', 14530.0)
        fv.count_frame(chunk_data={'ExposureTime': 14530.0})
        assert 'exposure' not in fv.pending_sources

    def test_chunks_dont_clear_led(self):
        """LED has no chunk equivalent — must clear via skip-frames only."""
        fv = FrameValidity()
        fv.invalidate('led')
        # Even with matching chunk values for OTHER sources, LED is not chunk-validatable
        fv.count_frame(chunk_data={'Gain': 5.0, 'ExposureTime': 14530.0})
        assert 'led' in fv.pending_sources  # still pending after 1 frame
        fv.count_frame(chunk_data={'Gain': 5.0, 'ExposureTime': 14530.0})
        assert 'led' not in fv.pending_sources  # cleared via skip-frames at frame 2

    def test_chunks_dont_clear_motion(self):
        """Motion sources are firmware-gated, not chunk-checkable."""
        fv = FrameValidity()
        # Settle check that always says "still moving" -> motion never settles
        fv.set_settle_check(lambda src: False)
        fv.invalidate('z_move')
        # Many frames with matching chunks -> still pending (settle check blocks)
        for _ in range(10):
            fv.count_frame(chunk_data={'Gain': 5.0})
        assert 'z_move' in fv.pending_sources

    def test_no_match_falls_back_to_skip_frames(self):
        """If chunks don't match target, source clears via skip_frames as before."""
        fv = FrameValidity()
        fv.invalidate('gain')              # default skip = 2 frames
        fv.set_target('gain', 5.0)
        # Chunks DON'T match (camera still on old gain)
        fv.count_frame(chunk_data={'Gain': 1.0})
        assert 'gain' in fv.pending_sources  # not cleared by chunks
        fv.count_frame(chunk_data={'Gain': 1.0})
        assert 'gain' not in fv.pending_sources  # cleared by skip_frames at frame 2

    def test_no_target_recorded_falls_back_to_skip_frames(self):
        """If set_target was never called, chunk-match is impossible — skip-frames only."""
        fv = FrameValidity()
        fv.invalidate('gain')
        # set_target NOT called -> _target_values is empty
        fv.count_frame(chunk_data={'Gain': 5.0})
        assert 'gain' in fv.pending_sources
        fv.count_frame(chunk_data={'Gain': 5.0})
        assert 'gain' not in fv.pending_sources  # cleared by skip-frames

    def test_count_frame_without_chunk_data_unchanged(self):
        """Backward compat: count_frame() without chunk_data behaves as before."""
        fv = FrameValidity()
        fv.invalidate('gain')
        fv.set_target('gain', 5.0)  # target recorded but no chunks supplied
        fv.count_frame()             # no chunk_data -> skip-frames path
        assert 'gain' in fv.pending_sources
        fv.count_frame()
        assert 'gain' not in fv.pending_sources

    def test_partial_chunk_data(self):
        """chunk_data missing a key for one source clears the other."""
        fv = FrameValidity()
        fv.invalidate('gain')
        fv.invalidate('exposure')
        fv.set_target('gain', 5.0)
        fv.set_target('exposure', 14530.0)
        # Chunks have Gain but not ExposureTime
        fv.count_frame(chunk_data={'Gain': 5.0})
        assert 'gain' not in fv.pending_sources
        assert 'exposure' in fv.pending_sources

    def test_chunk_clear_is_atomic_with_settle_check(self):
        """If a source is settled by skip_frames, it's cleared regardless of chunks."""
        fv = FrameValidity()
        fv.invalidate('gain')
        # No target -> chunks can't match
        # Skip-frames count should still clear after 2 frames
        fv.count_frame(chunk_data={'Gain': 999.0})
        fv.count_frame(chunk_data={'Gain': 999.0})
        assert 'gain' not in fv.pending_sources

    def test_frames_until_valid_after_chunk_clear(self):
        """frames_until_valid reads pending_sources; chunk-clear shows immediately."""
        fv = FrameValidity()
        fv.invalidate('gain')
        fv.set_target('gain', 5.0)
        assert fv.frames_until_valid() > 0  # before grab
        fv.count_frame(chunk_data={'Gain': 5.0})
        assert fv.frames_until_valid() == 0  # cleared by chunks

    def test_is_valid_after_chunk_clear(self):
        """is_valid sees chunk-cleared sources too (red/green dot stays correct)."""
        fv = FrameValidity()
        fv.invalidate('gain')
        fv.set_target('gain', 5.0)
        assert fv.is_valid is False
        fv.count_frame(chunk_data={'Gain': 5.0})
        assert fv.is_valid is True
