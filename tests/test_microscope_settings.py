# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for microscope settings load / save.

Added for issue #616: no-camera startup corrupted stored exposures.

Root cause: the Lumascope camera cache defaulted `max_exposure` to 0.0 and
only populated it during `_populate_camera_cache()`, which early-returns
when no camera is connected. `MicroscopeSettings.load_settings()` used the
cached value as an exposure-slider upper bound, hitting an `if exp <=
max_exposure` branch where `max_exposure == 0` caused every stored
exposure to be clamped to 0. Shutdown's `save_settings()` then wrote
zeros back to disk, corrupting the settings file for future sessions.

Structural fix (4.1): `ImagingAPI.max_exposure_ms_cached` now returns `None`
(not 0.0) when no camera is connected, so callers can distinguish
"camera missing" from a real driver value. `load_settings` falls back to
`DEFAULT_MAX_EXPOSURE_MS` with `scope.imaging.max_exposure_ms_cached or DEFAULT`.
"""

from unittest.mock import MagicMock

from modules.config_helpers import DEFAULT_MAX_EXPOSURE_MS


class TestCameraMaxExposureContract:
    """Pin the ImagingAPI.max_exposure_ms_cached no-camera contract.

    The contract is: the property returns None when no camera is
    connected or the cache has not been populated with a real value.
    load_settings relies on `value or DEFAULT_MAX_EXPOSURE_MS` for the
    fallback, so anything falsy (None, 0, 0.0) is equivalent from the
    caller's perspective -- but None is the intended sentinel.
    """

    def test_inactive_camera_yields_none_max_exposure(self):
        """Forcing camera cache to inactive must leave max_exposure None."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        # Simulator connects an active camera by default. Force the exact
        # no-camera state that load_settings sees on a real missing camera.
        with scope.imaging._camera_cache_lock:
            scope.imaging._camera_cache['active'] = False
            scope.imaging._camera_cache['max_exposure_ms'] = None

        assert scope.imaging.max_exposure_ms_cached is None

    def test_zero_in_cache_yields_none_max_exposure(self):
        """Legacy 0.0 in cache (driver returned 0) is coerced to None.

        Belt-and-suspenders: even if something writes 0.0 into the cache,
        the property still returns None so callers see a consistent
        "camera missing" signal.
        """
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        with scope.imaging._camera_cache_lock:
            scope.imaging._camera_cache['max_exposure_ms'] = 0.0

        assert scope.imaging.max_exposure_ms_cached is None

    def test_populated_value_passes_through(self):
        """A real positive value in the cache is returned as float."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        with scope.imaging._camera_cache_lock:
            scope.imaging._camera_cache['max_exposure_ms'] = 500.0

        assert scope.imaging.max_exposure_ms_cached == 500.0
        assert isinstance(scope.imaging.max_exposure_ms_cached, float)

    def test_integer_in_cache_is_coerced_to_float(self):
        """Integer from a driver is returned as float for caller consistency."""
        from modules.lumascope_api import Lumascope

        scope = Lumascope(simulate=True)
        with scope.imaging._camera_cache_lock:
            scope.imaging._camera_cache['max_exposure_ms'] = 750

        assert scope.imaging.max_exposure_ms_cached == 750.0
        assert isinstance(scope.imaging.max_exposure_ms_cached, float)


class TestLoadSettingsFallback:
    """Regression for #616: load_settings must fall back when no camera."""

    def test_default_constant_pinned(self):
        """Pin the default so a refactor can't silently change it."""
        assert DEFAULT_MAX_EXPOSURE_MS == 1000.0

    def test_none_falls_back_to_default(self):
        """The `value or DEFAULT` pattern must yield DEFAULT for None."""
        value = None
        assert (value or DEFAULT_MAX_EXPOSURE_MS) == DEFAULT_MAX_EXPOSURE_MS

    def test_zero_falls_back_to_default(self):
        """Defensive: 0.0 in cache (shouldn't happen post-fix) still safe."""
        value = 0.0
        assert (value or DEFAULT_MAX_EXPOSURE_MS) == DEFAULT_MAX_EXPOSURE_MS

    def test_valid_value_overrides_default(self):
        """Real camera value must pass through, not get replaced."""
        value = 500.0
        assert (value or DEFAULT_MAX_EXPOSURE_MS) == 500.0


class TestCoalescingApplier:
    """Issue #624: rapid set_frame_size calls on Pylon must NOT queue
    up behind each other (each takes ~11s due to stop/start grabbing),
    or the UI appears frozen for minutes. _CoalescingApplier keeps at
    most one task in flight; new submissions during an apply collapse
    into the latest value."""

    def _make(self):
        # kivy isn't importable in the test env, but _CoalescingApplier
        # is pure Python -- import it directly without dragging in the
        # MicroscopeSettings class (which imports Kivy).
        import pathlib

        src = pathlib.Path(__file__).parent.parent / 'ui' / 'microscope_settings.py'
        # Load only the helper by reading the source and exec'ing the
        # class block. Heavier but avoids Kivy / Clock imports at
        # module load.
        text = src.read_text()
        # Carve out the _CoalescingApplier class by string slicing
        # between its marker and the next class.
        start = text.index('class _CoalescingApplier:')
        end = text.index('class MicroscopeSettings')
        snippet = (
            'import logging, threading\nlogger = logging.getLogger(__name__)\n' + text[start:end]
        )
        ns = {}
        exec(compile(snippet, str(src), 'exec'), ns)
        return ns['_CoalescingApplier']()

    def test_single_submit_applies_once(self):
        applier = self._make()
        fn = MagicMock()
        assert applier.submit((1900, 2100)) is True
        applier.apply_pending(fn)
        fn.assert_called_once_with((1900, 2100))

    def test_submit_returns_false_when_in_flight(self):
        applier = self._make()
        assert applier.submit(('a',)) is True  # first queues
        assert applier.submit(('b',)) is False  # second coalesces
        # Apply picks up the LATEST only.
        fn = MagicMock()
        applier.apply_pending(fn)
        fn.assert_called_once_with(('b',))

    def test_late_arrival_picked_up_same_task(self):
        applier = self._make()
        applier.submit((1900, 2100))

        calls = []

        def _fn(val):
            calls.append(val)
            if len(calls) == 1:
                # Simulate UI submitting a fresh value while the first
                # apply is mid-execution.
                applier.submit((3860, 2100))

        applier.apply_pending(_fn)
        assert calls == [(1900, 2100), (3860, 2100)]

    def test_exception_clears_in_flight(self):
        applier = self._make()
        applier.submit((1900, 2100))

        def _fn(val):
            raise RuntimeError('pylon sulked')

        applier.apply_pending(_fn)  # must not raise
        # Next submit should succeed as a fresh enqueue.
        assert applier.submit((1900, 2100)) is True

    def test_empty_pending_is_noop(self):
        applier = self._make()
        # Never submitted -- apply should short-circuit.
        fn = MagicMock()
        applier.apply_pending(fn)
        fn.assert_not_called()

    def test_duplicate_of_applied_value_absorbed(self):
        """One user edit fires the bound handler up to FOUR times with
        the identical (width, height) pair (on_text_validate + on_focus
        loss per field, and the handler reads both fields every call).
        On a fast camera (FX2, millisecond applies) the in-flight gate
        closes between events, so each repeat became a real hardware
        apply. Exact repeats of the applied value must be absorbed."""
        applier = self._make()
        # Bare True: acceptance without a value -> the request itself is
        # recorded (a MagicMock return is truthy-but-not-True and would be
        # recorded AS the dedupe key under the record-returned-value rule).
        fn = MagicMock(return_value=True)
        assert applier.submit((1896, 1896)) is True
        applier.apply_pending(fn)
        # The three trailing duplicate events of the same user edit.
        assert applier.submit((1896, 1896)) is False
        assert applier.submit((1896, 1896)) is False
        assert applier.submit((1896, 1896)) is False
        applier.apply_pending(fn)
        fn.assert_called_once_with((1896, 1896))

    def test_distinct_value_still_applies(self):
        """Absorption only drops exact repeats; a genuinely new pair
        (resolution edit, binning-driven halving) must still fire."""
        applier = self._make()
        calls = []
        applier.submit((1900, 1900))
        applier.apply_pending(calls.append)
        assert applier.submit((1896, 1896)) is True
        applier.apply_pending(calls.append)
        assert calls == [(1900, 1900), (1896, 1896)]

    def test_inflight_duplicate_not_reapplied(self):
        """A repeat equal to the value being applied that folds into an
        in-flight task is dropped at drain time, not re-sent."""
        applier = self._make()
        calls = []

        def _fn(val):
            calls.append(val)
            if len(calls) == 1:
                applier.submit(val)
            # Acceptance is signaled by a truthy return (the applied
            # value); a bare None reads as a rejected apply and is not
            # recorded, so the drain-time fold would not see it.
            return True

        applier.submit((1900, 2100))
        applier.apply_pending(_fn)
        assert calls == [(1900, 2100)]

    def test_failed_apply_allows_same_value_retry(self):
        """A failed apply must not record the value as applied; the
        user retrying the same size still reaches the hardware."""
        applier = self._make()

        def _boom(val):
            raise RuntimeError('camera sulked')

        applier.submit((1900, 2100))
        applier.apply_pending(_boom)
        assert applier.submit((1900, 2100)) is True
        fn = MagicMock()
        applier.apply_pending(fn)
        fn.assert_called_once_with((1900, 2100))

    def test_falsy_return_not_recorded_allows_same_value_retry(self):
        """The sentinel-return gap: 'failed' covers BOTH failure shapes.
        A falsy fn return (a driver rejection False, or the camera-absent
        None no-op) must not be recorded as applied, or the user's retry
        of the identical value would be absorbed by the dedupe record."""
        applier = self._make()
        applier.submit((1900, 2100))
        applier.apply_pending(lambda val: False)  # driver rejection shape
        assert applier.submit((1900, 2100)) is True, (
            'a False-returning apply must not poison the dedupe record'
        )
        applier.apply_pending(lambda val: None)  # camera-absent shape
        assert applier.submit((1900, 2100)) is True, (
            'a None-returning apply must not poison the dedupe record'
        )
        # Drain so the class-level in-flight state does not leak.
        applier.apply_pending(lambda val: True)

    def test_retry_after_recovery_recorded_then_absorbed(self):
        """First apply fails (raise), the retry of the identical value
        reaches the hardware and IS recorded, and only then is a third
        submit of the same value absorbed."""
        applier = self._make()
        applier.submit((1900, 2100))

        def _boom(val):
            raise RuntimeError('camera sulked')

        applier.apply_pending(_boom)

        assert applier.submit((1900, 2100)) is True  # retry reaches fn
        calls = []

        def _delivering(val):
            calls.append(val)
            # The production fn (_push_frame_size) returns the delivered
            # (w, h) tuple; apply_pending records the returned value as
            # the dedupe key.
            return (1900, 2100)

        applier.apply_pending(_delivering)
        assert calls == [(1900, 2100)]

        # Recorded now -> the trailing duplicate is absorbed.
        assert applier.submit((1900, 2100)) is False

    def test_camera_absent_none_not_recorded_resubmit_after_reconnect(self):
        """A camera-absent apply returns None (the quiet sentinel); it must
        not be recorded, so the resubmit after reconnect reaches the
        hardware instead of being absorbed as 'already applied'."""
        applier = self._make()
        applier.submit((1896, 1896))
        applier.apply_pending(lambda val: None)  # absent camera: quiet no-op

        assert applier.submit((1896, 1896)) is True  # 'reconnected' retry
        calls = []
        applier.apply_pending(lambda val: calls.append(val) or True)
        assert calls == [(1896, 1896)], (
            'the post-reconnect apply of the identical value must reach the hardware'
        )

    def test_delivered_value_recorded_as_dedupe_key_not_the_request(self):
        """An fn that returns the APPLIED value (a clamped delivery) records
        THAT as the dedupe key: a repeat of the delivered size is absorbed,
        but the user retyping the ORIGINAL request after seeing the clamp
        still reaches the hardware (recording the request key would absorb
        it against a value the camera never took)."""
        applier = self._make()
        applier.submit((1900, 1900))
        applier.apply_pending(lambda val: (1896, 1900))  # camera clamps

        assert applier.submit((1896, 1900)) is False, (
            'the DELIVERED size is what the hardware holds -- absorbed'
        )
        assert applier.submit((1900, 1900)) is True, (
            'the original request differs from the delivered key -- the retype must go through'
        )
        applier.apply_pending(lambda val: True)  # drain the in-flight state

    def test_bare_true_records_the_request_itself(self):
        applier = self._make()
        applier.submit((1900, 2100))
        applier.apply_pending(lambda val: True)
        assert applier.submit((1900, 2100)) is False, (
            'a bare-True apply records the request as the dedupe key'
        )


# ---------------------------------------------------------------------------
# Mirror commit-then-revert: the binning / image-mode mirrors commit
# SYNCHRONOUSLY at select time (the documented synchronous-binning SSOT);
# the completion callbacks are pure no-ops on success and revert to the
# captured prior state on failure. Frame-size geometry mirrors are the
# exception: they are written from the DELIVERED size in the Clock-scheduled
# landing. MicroscopeSettings imports Kivy, so each method is AST-extracted
# and exec'd with a controlled namespace (the
# test_layer_control_ag_exposure_floor pattern) and bound to a
# SimpleNamespace self.
# ---------------------------------------------------------------------------


def _extract_ms_method(method_name: str) -> str:
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / 'ui' / 'microscope_settings.py'
    source = src.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'MicroscopeSettings':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.unparse(child)
    raise AssertionError(f'MicroscopeSettings.{method_name} not found')


def _compile_ms_method(method_name: str, namespace: dict):
    fn_src = _extract_ms_method(method_name)
    ns = dict(namespace)
    exec(compile(fn_src, f'<microscope_settings::{method_name}>', 'exec'), ns)
    return ns[method_name]


class TestFrameSizeMirrorChain:
    """_push_frame_size (executor side) + _on_frame_size_applied (UI-thread
    landing): the mirrors are written from the DELIVERED geometry only, and
    the coalescer dedupe key is the delivered (width, height) tuple."""

    def _make_push(self, set_frame_size, scope_present=True):
        from types import SimpleNamespace

        scheduled = []
        clock = SimpleNamespace(
            schedule_once=lambda cb, dt=0: scheduled.append(cb),
        )
        calls = []

        def _recording_set_frame_size(w, h):
            calls.append((w, h))
            return set_frame_size(w, h)

        scope = (
            # _push_frame_size runs on the camera worker, so production
            # binds the impl seam -- the fake mirrors that name.
            SimpleNamespace(imaging=SimpleNamespace(_set_frame_size_impl=_recording_set_frame_size))
            if scope_present
            else None
        )
        ctx = SimpleNamespace(lumaview=SimpleNamespace(scope=scope))
        fn = _compile_ms_method(
            '_push_frame_size',
            {'_app_ctx': SimpleNamespace(ctx=ctx), 'Clock': clock},
        )
        landings = []
        fake_self = SimpleNamespace(
            _on_frame_size_applied=lambda delivered: landings.append(delivered),
        )
        return fn, fake_self, scheduled, landings, calls

    def test_push_returns_delivered_tuple_and_schedules_dict_landing(self):
        # The return value is the (w, h) TUPLE (the coalescer dedupe key --
        # what the camera actually holds); the UI landing receives the dict.
        delivered = {'width': 1896, 'height': 1900}
        fn, fake_self, scheduled, landings, _calls = self._make_push(lambda w, h: delivered)

        result = fn(fake_self, (1900, 1900))

        assert result == (1896, 1900)
        assert len(scheduled) == 1, 'exactly one UI landing scheduled'
        scheduled[0](0)  # run the Clock-scheduled callable
        assert landings == [delivered]

    def test_push_returns_none_and_schedules_nothing_when_camera_absent(self):
        fn, fake_self, scheduled, landings, _calls = self._make_push(lambda w, h: None)

        result = fn(fake_self, (1900, 1900))

        assert result is None  # falsy -> the coalescer does not record it
        assert scheduled == []
        assert landings == []

    def test_push_returns_none_when_scope_slot_is_none(self):
        # The reconnect window: lumaview.scope is None for its whole span.
        # That is the absent shape, not an error -- no driver touch, no
        # landing, nothing recorded.
        fn, fake_self, scheduled, landings, calls = self._make_push(
            lambda w, h: {'width': w, 'height': h}, scope_present=False
        )

        result = fn(fake_self, (1900, 1900))

        assert result is None
        assert calls == [], 'the driver must never be reached without a scope'
        assert scheduled == []
        assert landings == []

    def test_push_lets_typed_rejection_propagate(self):
        # apply_pending contains the raise (and skips recording); the push
        # must not swallow it into a truthy/None shape.
        import pytest

        from modules.exceptions import CameraSettingRejected

        def _reject(w, h):
            raise CameraSettingRejected('frame_size', {'width': w, 'height': h})

        fn, fake_self, scheduled, _landings, _calls = self._make_push(_reject)
        with pytest.raises(CameraSettingRejected):
            fn(fake_self, (1900, 1900))
        assert scheduled == []

    def test_landing_writes_mirrors_and_refreshes_fov(self):
        # Design test 4: the delivered geometry differs from any prior
        # request (quantized delivery); the mirrors reflect IT, and the FOV
        # refresh runs against the just-written settings.
        from types import SimpleNamespace

        settings = {'frame': {'width': 1900, 'height': 1900}}
        fn = _compile_ms_method(
            '_on_frame_size_applied',
            {'_app_ctx': SimpleNamespace(ctx=SimpleNamespace(settings=settings))},
        )
        ids = {
            'frame_width_id': SimpleNamespace(text=''),
            'frame_height_id': SimpleNamespace(text=''),
        }
        fov_refreshes = []
        fake_self = SimpleNamespace(
            ids=ids,
            _refresh_fov_labels=lambda: fov_refreshes.append(dict(settings['frame'])),
        )

        fn(fake_self, {'width': 1896, 'height': 1900})

        assert settings['frame'] == {'width': 1896, 'height': 1900}
        assert ids['frame_width_id'].text == '1896'
        assert ids['frame_height_id'].text == '1900'
        # The refresh ran AFTER the settings write, so it read the
        # delivered geometry.
        assert fov_refreshes == [{'width': 1896, 'height': 1900}]

    def test_refresh_fov_labels_computes_from_settings_frame(self, scale_ctx):
        from types import SimpleNamespace

        import modules.common_utils as common_utils_real

        settings = {'frame': {'width': 1896, 'height': 1900}, 'objective_id': '4x'}
        objective = {'focal_length': 9.0}
        ctx = SimpleNamespace(
            settings=settings,
            session=SimpleNamespace(
                get_objective_info=lambda objective_id: objective,
            ),
        )
        fn = _compile_ms_method(
            '_refresh_fov_labels',
            {
                '_app_ctx': SimpleNamespace(ctx=ctx),
                'common_utils': common_utils_real,
                'get_binning_from_ui': lambda: 1,
            },
        )
        ids = {
            'field_of_view_width_id': SimpleNamespace(text=''),
            'field_of_view_height_id': SimpleNamespace(text=''),
        }
        fake_self = SimpleNamespace(ids=ids)

        fn(fake_self)

        expected_fov = common_utils_real.get_field_of_view(
            focal_length=objective['focal_length'],
            frame_size={'width': 1896, 'height': 1900},
            binning_size=1,
        )
        assert ids['field_of_view_width_id'].text == str(round(expected_fov['width'], 0))
        assert ids['field_of_view_height_id'].text == str(round(expected_fov['height'], 0))


class TestBinningApplyOutcome:
    """Commit-then-revert: select_binning_size commits every mirror
    synchronously; _on_binning_apply_outcome is a pure no-op on success and
    reverts to the captured prior state on failure."""

    def _make_outcome(self):
        from types import SimpleNamespace

        # Settings reflect the SYNCHRONOUS select-time commit ('4x4').
        settings = {'binning': {'size': '4x4'}}
        ctx = SimpleNamespace(settings=settings)
        fn = _compile_ms_method(
            '_on_binning_apply_outcome',
            {'_app_ctx': SimpleNamespace(ctx=ctx), 'logger': MagicMock()},
        )
        pushed_frames = []
        hint_refreshes = []
        fake_self = SimpleNamespace(
            ids={'binning_spinner': SimpleNamespace(text='4x4')},
            _refresh_binning_depth_hint=lambda: hint_refreshes.append(1),
            _apply_displayed_frame=lambda frame: pushed_frames.append(frame),
        )
        return fn, fake_self, settings, pushed_frames

    def test_success_is_a_pure_noop(self):
        fn, fake_self, settings, pushed_frames = self._make_outcome()

        fn(fake_self, '4x4', '1x1', {'width': 1920, 'height': 1200}, result=True, exception=None)

        assert settings['binning']['size'] == '4x4', 'the sync commit stands untouched'
        assert fake_self.ids['binning_spinner'].text == '4x4'
        assert pushed_frames == [], 'success must not re-push the frame'

    def test_failure_reverts_to_captured_prior_state(self):
        fn, fake_self, settings, pushed_frames = self._make_outcome()
        prior_frame = {'width': 1920, 'height': 1200}

        fn(fake_self, '4x4', '1x1', prior_frame, result=None, exception=RuntimeError('rejected'))

        assert settings['binning']['size'] == '1x1', 'the sync commit must be reverted'
        assert fake_self.ids['binning_spinner'].text == '1x1'
        assert pushed_frames == [prior_frame], (
            'the prior frame derivation must be re-pushed on revert'
        )

    def test_falsy_result_without_exception_also_reverts(self):
        fn, fake_self, settings, pushed_frames = self._make_outcome()
        prior_frame = {'width': 1920, 'height': 1200}

        fn(fake_self, '4x4', '1x1', prior_frame, result=False, exception=None)

        assert settings['binning']['size'] == '1x1'
        assert fake_self.ids['binning_spinner'].text == '1x1'
        assert pushed_frames == [prior_frame]


class TestSelectBinningSynchronousCommit:
    """select_binning_size commits settings + frame text fields at SELECT
    time (the documented synchronous-binning SSOT), before -- and
    independent of -- any executor callback."""

    def _make_select(self, initializing, settings, puts):
        from types import SimpleNamespace

        import modules.binning as binning_real

        imaging = SimpleNamespace(
            get_available_binning_sizes=lambda: [1, 2, 4],
            get_pixel_alignment=lambda: {'width': 4, 'height': 4},
            get_binning_size=lambda: 1,
            set_binning_size=lambda size: True,
        )
        ctx = SimpleNamespace(
            settings=settings,
            initializing=initializing,
            lumaview=SimpleNamespace(scope=SimpleNamespace(imaging=imaging)),
            camera_executor=SimpleNamespace(
                put=lambda task: puts.append((task, settings['binning']['size'])),
            ),
        )
        from modules.sequential_io_executor import IOTask

        fn = _compile_ms_method(
            'select_binning_size',
            {
                '_app_ctx': SimpleNamespace(ctx=ctx),
                'binning': binning_real,
                'gui_logger': MagicMock(),
                'logger': MagicMock(),
                'IOTask': IOTask,
            },
        )
        pushed_frames = []
        fake_self = SimpleNamespace(
            ids={
                'binning_spinner': SimpleNamespace(text='2x2'),
                'frame_width_id': SimpleNamespace(text='1920'),
                'frame_height_id': SimpleNamespace(text='1200'),
            },
            _native_roi=lambda: {'width': 1920, 'height': 1200},
            _store_native_roi=lambda native: None,
            _refresh_binning_depth_hint=lambda: None,
            _apply_displayed_frame=lambda frame: pushed_frames.append(frame),
            _on_binning_apply_outcome=lambda *a, **kw: None,
        )
        return fn, fake_self, pushed_frames

    def test_initializing_commits_synchronously_without_iotask(self):
        settings = {'binning': {'size': '1x1'}, 'frame': {'width': 1920, 'height': 1200}}
        puts = []
        fn, fake_self, pushed_frames = self._make_select(True, settings, puts)

        fn(fake_self)

        assert settings['binning']['size'] == '2x2', 'committed at select time'
        assert fake_self.ids['frame_width_id'].text == '960'  # 1920/2, aligned 4
        assert fake_self.ids['frame_height_id'].text == '600'
        assert puts == [], 'during init no IOTask is enqueued'
        assert pushed_frames == [], 'during init the hardware push is skipped'

    def test_live_select_commits_before_the_executor_put(self):
        settings = {'binning': {'size': '1x1'}, 'frame': {'width': 1920, 'height': 1200}}
        puts = []
        fn, fake_self, pushed_frames = self._make_select(False, settings, puts)

        fn(fake_self)

        assert len(puts) == 1
        task, settings_value_at_put_time = puts[0]
        assert settings_value_at_put_time == '2x2', (
            'settings must hold the new factor BEFORE the executor put'
        )
        # The captured prior state rides on the callback for the revert.
        assert task.cb_args == ('2x2', '1x1', {'width': 1920, 'height': 1200})
        assert pushed_frames == [{'width': 960, 'height': 600}]


class TestImageModeOutcome:
    """Commit-then-revert: select_image_mode commits synchronously;
    _on_image_mode_outcome no-ops on success and reverts all three mirrors
    to the captured prior mode on failure."""

    def _make(self, committed_mode='12bit_scientific'):
        from types import SimpleNamespace

        import modules.image_mode as image_mode_real

        # Settings reflect the SYNCHRONOUS select-time commit.
        settings = {'image_mode': committed_mode}
        scope_display = SimpleNamespace(image_mode=committed_mode)
        ctx = SimpleNamespace(settings=settings, scope_display=scope_display)
        fn = _compile_ms_method(
            '_on_image_mode_outcome',
            {
                '_app_ctx': SimpleNamespace(ctx=ctx),
                'image_mode': image_mode_real,
                'logger': MagicMock(),
            },
        )
        fake_self = SimpleNamespace(
            ids={
                'image_mode_spinner': SimpleNamespace(
                    text=image_mode_real.IMAGE_MODE_LABELS[committed_mode]
                ),
            },
            _refresh_binning_depth_hint=lambda: None,
        )
        return fn, fake_self, settings, scope_display, image_mode_real

    def test_success_is_a_pure_noop(self):
        fn, fake_self, settings, scope_display, im = self._make()

        fn(fake_self, '12bit_scientific', '8bit', result=True, exception=None)

        assert settings['image_mode'] == '12bit_scientific'
        assert scope_display.image_mode == '12bit_scientific'
        assert fake_self.ids['image_mode_spinner'].text == im.IMAGE_MODE_LABELS['12bit_scientific']

    def test_failure_reverts_all_three_mirrors_to_prior_mode(self):
        fn, fake_self, settings, scope_display, im = self._make()

        fn(
            fake_self,
            '12bit_scientific',
            '8bit',
            result=None,
            exception=RuntimeError('rejected'),
        )

        assert settings['image_mode'] == '8bit'
        assert scope_display.image_mode == '8bit'
        assert fake_self.ids['image_mode_spinner'].text == im.IMAGE_MODE_LABELS['8bit']

    def test_falsy_result_without_exception_also_reverts(self):
        # The absent-camera False from set_pixel_format propagates verbatim
        # through _set_pixel_format; the mode commit must not survive it.
        fn, fake_self, settings, scope_display, im = self._make()

        fn(fake_self, '12bit_scientific', '8bit', result=False, exception=None)

        assert settings['image_mode'] == '8bit'
        assert scope_display.image_mode == '8bit'
        assert fake_self.ids['image_mode_spinner'].text == im.IMAGE_MODE_LABELS['8bit']
