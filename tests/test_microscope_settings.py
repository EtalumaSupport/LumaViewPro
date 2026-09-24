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

import pytest

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
        # The failure is raised to the caller, which reports it; the gate is
        # open again by then, so the next edit is not wedged behind it.
        applier = self._make()
        applier.submit((1900, 2100))

        def _fn(val):
            raise RuntimeError('pylon sulked')

        with pytest.raises(RuntimeError, match='pylon sulked'):
            applier.apply_pending(_fn)
        # Next submit should succeed as a fresh enqueue.
        assert applier.submit((1900, 2100)) is True

    def test_empty_pending_is_noop(self):
        applier = self._make()
        # Never submitted -- apply should short-circuit.
        fn = MagicMock()
        applier.apply_pending(fn)
        fn.assert_not_called()

    def test_distinct_value_still_applies(self):
        """A new pair submitted after a drain (resolution edit,
        binning-driven halving) is enqueued and applied."""
        applier = self._make()
        calls = []
        applier.submit((1900, 1900))
        applier.apply_pending(calls.append)
        assert applier.submit((1896, 1896)) is True
        applier.apply_pending(calls.append)
        assert calls == [(1900, 1900), (1896, 1896)]


# ---------------------------------------------------------------------------
# The handlers read the widget and hand the value to the Session member on
# the camera lane (submit_reported); the Session applies and stores. The
# redraws that run after the camera answers show the store: the selectors,
# the frame boxes, the display mode, the FOV. MicroscopeSettings imports
# Kivy, so each method is AST-extracted and exec'd with a controlled
# namespace (the test_layer_control_ag_exposure_floor pattern) and bound to
# a SimpleNamespace self.
# ---------------------------------------------------------------------------


def _ms_source_tree():
    """The module's AST, read once so every extractor below shares one read."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / 'ui' / 'microscope_settings.py'
    return ast.parse(src.read_text())


def _extract_ms_constant(name: str):
    """A module-level constant, taken from the source the methods come from.

    Hand-copying it into the test would let the two drift apart in silence,
    which is the whole failure this extraction harness exists to avoid.
    """
    import ast

    for node in _ms_source_tree().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f'microscope_settings.{name} not found')


def _extract_ms_method(method_name: str) -> str:
    import ast

    tree = _ms_source_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'MicroscopeSettings':
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return ast.unparse(child)
    raise AssertionError(f'MicroscopeSettings.{method_name} not found')


def _bind_write_frame_text(fake_self):
    """Give a fake panel the REAL frame-box write funnel.

    Stubbing it would hide the focus guard from every test that writes these
    boxes, and the guard is the only reason the funnel exists.
    """
    fn = _compile_ms_method('_write_frame_text', {})
    return lambda width, height: fn(fake_self, width, height)


def _compile_ms_method(method_name: str, namespace: dict):
    fn_src = _extract_ms_method(method_name)
    ns = dict(namespace)
    exec(compile(fn_src, f'<microscope_settings::{method_name}>', 'exec'), ns)
    return ns[method_name]


class _RecordingGuiLogger:
    """Records the GUI log calls a redraw makes, in order."""

    def __init__(self, ids=None):
        self.calls = []
        self._ids = ids or {}

    def note_write_back(self, name, value):
        # The text the spinner holds when the declaration is made, so a test
        # can tell a declaration before the write from one after it.
        spinner = {'BINNING': 'binning_spinner', 'IMAGE_MODE': 'image_mode_spinner'}[name]
        self.calls.append(('note_write_back', name, value, self._ids[spinner].text))

    def frame_size(self, width, height, binning_size):
        self.calls.append(('frame_size', width, height, binning_size))


class TestTheFramingRedrawShowsTheStore:
    """_framing_applied (the redraw after the camera answers a binning pick
    or a frame edit) shows what the Session stored: the delivered frame in
    the boxes and the FOV, the stored binning in the selector."""

    def _make(self, settings, spinner_text):
        from types import SimpleNamespace

        ids = {
            'binning_spinner': SimpleNamespace(text=spinner_text),
            'frame_width_id': SimpleNamespace(text='', focus=False),
            'frame_height_id': SimpleNamespace(text='', focus=False),
        }
        log = _RecordingGuiLogger(ids)
        namespace = {'_app_ctx': SimpleNamespace(ctx=SimpleNamespace(settings=settings))}
        namespace['gui_logger'] = log
        applied = _compile_ms_method('_framing_applied', namespace)
        redraw = _compile_ms_method('_redraw_framing', namespace)
        fov_refreshes = []
        fake_self = SimpleNamespace(
            ids=ids,
            _refresh_binning_depth_hint=lambda: None,
            refresh_fov_labels=lambda: fov_refreshes.append(dict(settings['frame'])),
            _ui_binning_size=lambda: int(settings['binning']['size'].split('x')[0]),
        )
        fake_self._write_frame_text = _bind_write_frame_text(fake_self)
        fake_self._redraw_framing = lambda: redraw(fake_self)
        return (lambda: applied(fake_self)), ids, log, fov_refreshes

    def test_the_redraw_writes_the_boxes_and_the_fov_from_the_delivered_frame(self):
        # The delivered geometry differs from any request (quantized
        # delivery); the Session stored IT, and the redraw shows it.
        settings = {'frame': {'width': 1896, 'height': 1900}, 'binning': {'size': '1x1'}}
        applied, ids, log, fov_refreshes = self._make(settings, '1x1')

        applied()

        assert ids['frame_width_id'].text == '1896'
        assert ids['frame_height_id'].text == '1900'
        # The FOV was computed from the stored, delivered frame.
        assert fov_refreshes == [{'width': 1896, 'height': 1900}]
        assert log.calls == [('frame_size', 1896, 1900, 1)], (
            'the framing the camera answered with is recorded once'
        )

    def test_a_refused_pick_puts_the_selector_back_and_declares_it(self):
        # The Session refused the pick and stored nothing, so the selector
        # goes back to the stored factor and the boxes to the stored frame;
        # the selector write is the app's, declared before it dispatches.
        settings = {'frame': {'width': 960, 'height': 600}, 'binning': {'size': '2x2'}}
        applied, ids, log, _fov = self._make(settings, '8x8')

        applied()

        assert ids['binning_spinner'].text == '2x2'
        assert ids['frame_width_id'].text == '960'
        assert ids['frame_height_id'].text == '600'
        assert ('note_write_back', 'BINNING', '2x2', '8x8') in log.calls, (
            f'the selector restore was not declared before the write: {log.calls}'
        )

    def _refresh_fov(self, scale_capabilities, monkeypatch, objective):
        """Run the refresh against a scope whose active objective is
        ``objective`` (None when unknown); returns the two label texts."""
        from types import SimpleNamespace

        import modules.app_context as app_context
        import modules.common_utils as common_utils_real
        import modules.config_ui_getters as config_ui_getters_real

        ctx = SimpleNamespace(
            settings={'frame': {'width': 1896, 'height': 1900}},
            # The active objective is the runtime state's answer.
            scope=SimpleNamespace(
                runtime_state=SimpleNamespace(get_current_objective=lambda: objective)
            ),
            # The GUI getter resolves the scale off the LIVE scope.
            lumaview=SimpleNamespace(scope=SimpleNamespace(capabilities=scale_capabilities)),
        )
        monkeypatch.setattr(app_context, 'ctx', ctx)
        fn = _compile_ms_method(
            'refresh_fov_labels',
            {
                '_app_ctx': SimpleNamespace(ctx=ctx),
                'common_utils': common_utils_real,
                'config_ui_getters': config_ui_getters_real,
                'get_binning_from_ui': lambda: 1,
            },
        )
        ids = {
            'field_of_view_width_id': SimpleNamespace(text='stale'),
            'field_of_view_height_id': SimpleNamespace(text='stale'),
        }
        fn(SimpleNamespace(ids=ids))
        return ids['field_of_view_width_id'].text, ids['field_of_view_height_id'].text

    def testrefresh_fov_labels_computes_from_settings_frame(self, scale_capabilities, monkeypatch):
        import modules.common_utils as common_utils_real

        objective = {'focal_length': 9.0}
        width, height = self._refresh_fov(scale_capabilities, monkeypatch, objective)

        expected_fov = common_utils_real.get_field_of_view(
            focal_length=objective['focal_length'],
            frame_size={'width': 1896, 'height': 1900},
            binning_size=1,
            capabilities=scale_capabilities,
        )
        assert width == str(round(expected_fov['width'], 0))
        assert height == str(round(expected_fov['height'], 0))

    def test_an_unknown_objective_blanks_the_readout(self, scale_capabilities, monkeypatch):
        # No objective, no field of view: the readout says nothing rather
        # than keep a stale value or compute one from a guess.
        assert self._refresh_fov(scale_capabilities, monkeypatch, None) == ('', '')


class TestSelectBinningHandler:
    """select_binning_size reads the selector and hands the pick to the
    Session on the camera lane; the store is the Session's to write. During
    init it only redraws. Live, it shows the frame the new binning will give
    in the boxes before it submits, so an edit typed while the apply runs is
    read against the binning the selector shows."""

    def _make_select(self, initializing, settings):
        from types import SimpleNamespace

        import modules.binning as binning_real

        submits = []
        previews = []
        applies = []
        session = SimpleNamespace(
            frame_at_binning=lambda size: previews.append(size) or {'width': 960, 'height': 600},
            set_binning_size=lambda size: applies.append(size),
        )
        ctx = SimpleNamespace(
            settings=settings,
            initializing=initializing,
            session=session,
            camera_executor=object(),
        )
        ids = {
            'binning_spinner': SimpleNamespace(text='2x2'),
            'frame_width_id': SimpleNamespace(text='1920', focus=False),
            'frame_height_id': SimpleNamespace(text='1200', focus=False),
        }

        def _submit_reported(call, redraw, label, *, lane=None):
            submits.append(
                {
                    'call': call,
                    'redraw': redraw,
                    'label': label,
                    'lane': lane,
                    'boxes': (ids['frame_width_id'].text, ids['frame_height_id'].text),
                    'stored_binning': settings['binning']['size'],
                }
            )

        namespace = {
            '_app_ctx': SimpleNamespace(ctx=ctx),
            'binning': binning_real,
            'gui_logger': MagicMock(),
            'logger': MagicMock(),
            'submit_reported': _submit_reported,
        }
        fn = _compile_ms_method('select_binning_size', namespace)
        redraw = _compile_ms_method('_redraw_framing', namespace)
        fake_self = SimpleNamespace(
            ids=ids,
            _refresh_binning_depth_hint=lambda: None,
            refresh_fov_labels=lambda: None,
            _framing_applied=lambda: None,
        )
        fake_self._write_frame_text = _bind_write_frame_text(fake_self)
        fake_self._redraw_framing = lambda: redraw(fake_self)
        return fn, fake_self, ctx, submits, previews, applies

    def test_initializing_only_redraws_and_submits_nothing(self):
        # Bring-up applies the stored binning and frame; the selector is only
        # being set from the store, so no camera work is enqueued.
        settings = {'binning': {'size': '2x2'}, 'frame': {'width': 960, 'height': 600}}
        fn, fake_self, _ctx, submits, previews, applies = self._make_select(True, settings)

        fn(fake_self)

        assert submits == [], 'during init no camera work is enqueued'
        assert previews == [] and applies == []
        assert fake_self.ids['frame_width_id'].text == '960'  # the stored frame
        assert fake_self.ids['frame_height_id'].text == '600'

    def test_live_select_previews_the_frame_before_the_submit(self):
        settings = {'binning': {'size': '1x1'}, 'frame': {'width': 1920, 'height': 1200}}
        fn, fake_self, ctx, submits, previews, applies = self._make_select(False, settings)

        fn(fake_self)

        assert previews == [2]
        assert len(submits) == 1
        submit = submits[0]
        assert submit['boxes'] == ('960', '600'), (
            'the boxes must show the frame the new binning gives BEFORE the submit'
        )
        assert submit['stored_binning'] == '1x1', "the store is the Session's to write"
        assert submit['label'] == 'BINNING'
        assert submit['lane'] is ctx.camera_executor
        assert submit['redraw'] is fake_self._framing_applied
        submit['call']()
        assert applies == [2]


class TestImageModeMirrorAgreesWithTheStore:
    """The display mirror never disagrees with the stored mode.

    Config is assembled from the store, while the on-screen depth hints read
    the mirror; capture_depth rides the mode into saved output, so the two
    answering differently would change the file without changing the screen.
    Agreement holds because the one writer of the mirror, the image-mode
    redraw that runs after the camera answers, writes it from the stored
    mode, and the mirror's untouched default is the same mode the resolver
    reports for a store that has never been written.
    """

    def _make_redraw(self, stored_mode, shown_mode):
        from types import SimpleNamespace

        import modules.image_mode as image_mode_real

        settings = {'image_mode': stored_mode}
        scope_display = SimpleNamespace(image_mode=shown_mode)
        ids = {
            'image_mode_spinner': SimpleNamespace(
                text=image_mode_real.IMAGE_MODE_LABELS[shown_mode]
            ),
        }
        log = _RecordingGuiLogger(ids)
        ctx = SimpleNamespace(settings=settings, scope_display=scope_display)
        fn = _compile_ms_method(
            '_redraw_image_mode',
            {
                '_app_ctx': SimpleNamespace(ctx=ctx),
                'image_mode': image_mode_real,
                'gui_logger': log,
            },
        )
        fake_self = SimpleNamespace(ids=ids, _refresh_binning_depth_hint=lambda: None)
        return (lambda: fn(fake_self)), settings, scope_display, ids, log

    def test_every_mode_the_mirror_can_hold_resolves_to_itself(self):
        from types import SimpleNamespace

        import modules.image_mode as image_mode_real

        for mode in image_mode_real._MODE_TABLE:
            settings = {'image_mode': mode}
            scope_display = SimpleNamespace(image_mode=mode)
            assert image_mode_real.resolve_settings_image_mode(settings) == (
                scope_display.image_mode
            ), f'mirror and store disagree for {mode}'

    def test_the_untouched_mirror_matches_an_unwritten_store(self):
        # Before any selection the mirror carries its property default and the
        # store has no mode at all -- the state a hand-built settings dict is
        # also in. Both must name the same mode, or the GUI and a headless
        # caller would start out disagreeing.
        import modules.image_mode as image_mode_real

        assert image_mode_real.resolve_settings_image_mode({}) == (
            image_mode_real.DEFAULT_IMAGE_MODE
        )

    def test_the_redraw_leaves_the_mirror_and_the_store_in_agreement(self):
        # The redraw runs whatever the outcome: after an accepted change the
        # store moved and the mirror has not yet; after a refusal the store
        # stayed and the selector moved. Either way the mirror follows the
        # store.
        import modules.image_mode as image_mode_real

        for stored in image_mode_real._MODE_TABLE:
            redraw, settings, scope_display, ids, _log = self._make_redraw(
                stored, shown_mode='12bit_scaled' if stored != '12bit_scaled' else '8bit'
            )

            redraw()

            assert (
                image_mode_real.resolve_settings_image_mode(settings) == scope_display.image_mode
            ), f'the redraw to {stored} left the mirror and the store disagreeing'
            assert ids['image_mode_spinner'].text == image_mode_real.IMAGE_MODE_LABELS[stored]

    def test_a_refused_format_puts_the_selector_back_and_declares_it(self):
        # The Session refused 12-bit and stored nothing; the selector the
        # person moved goes back to the stored mode, declared as the app's
        # write before it dispatches.
        import modules.image_mode as image_mode_real

        redraw, _settings, scope_display, ids, log = self._make_redraw(
            '8bit', shown_mode='12bit_scientific'
        )
        scope_display.image_mode = '8bit'

        redraw()

        assert ids['image_mode_spinner'].text == image_mode_real.IMAGE_MODE_LABELS['8bit']
        assert scope_display.image_mode == '8bit'
        shown_12bit = image_mode_real.IMAGE_MODE_LABELS['12bit_scientific']
        assert ('note_write_back', 'IMAGE_MODE', '8bit', shown_12bit) in log.calls, (
            f'the selector restore was not declared before the write: {log.calls}'
        )


class TestTheFrameBoxesRecordWhatWasTyped:
    """A frame edit reports what the user entered, before anything acts on it.

    The boxes are an editor: until an apply lands they can hold a size no
    camera is at. Two things follow, and both are pinned here. The typed value
    is recorded FIRST, so a bundle reads in the order the user acted and a
    freeze cannot swallow the entry. And an entry that is not a pair of
    integers is a CORRECTION, not a request -- emptying a box used to log
    FRAME_SIZE for the size already in force, so the record claimed an edit
    that never happened while the box sat blank.
    """

    def _make(self, width_text, unparseable=False, camera_present=True):
        from types import SimpleNamespace

        records = []
        settings = {'frame': {'width': 768, 'height': 1200}}
        boxes = {
            'frame_width_id': SimpleNamespace(text=width_text, focus=False),
            'frame_height_id': SimpleNamespace(text='1200', focus=False),
        }

        def _typed():
            if unparseable:
                raise ValueError('Invalid value for frame width/height')
            return {
                'width': int(boxes['frame_width_id'].text),
                'height': int(boxes['frame_height_id'].text),
            }

        applied = []

        def _set_frame_size(width, height):
            applied.append({'width': width, 'height': height})
            return {'width': width, 'height': height} if camera_present else None

        ctx = SimpleNamespace(
            settings=settings,
            session=SimpleNamespace(set_frame_size=_set_frame_size),
            camera_executor=object(),
        )

        def _submit_reported(call, redraw, label, *, lane=None):
            # The camera lane, run inline: the apply is what is under test.
            call()

        fn = _compile_ms_method(
            'frame_size',
            {
                '_app_ctx': SimpleNamespace(ctx=ctx),
                'logger': MagicMock(),
                '_FRAME_BOXES': _extract_ms_constant('_FRAME_BOXES'),
                'gui_logger': SimpleNamespace(
                    text_input=lambda name, value: records.append((name, str(value)))
                ),
                'submit_reported': _submit_reported,
            },
        )
        fake_self = SimpleNamespace(
            ids=boxes,
            _typed_frame_dimensions=_typed,
            _frame_size_applier=TestCoalescingApplier()._make(),
            _framing_applied=lambda: None,
        )
        fake_self._write_frame_text = _bind_write_frame_text(fake_self)
        return fn, fake_self, records, applied, settings, boxes

    def test_the_typed_width_is_recorded_before_the_apply(self):
        fn, fake_self, records, applied, _settings, _boxes = self._make('800')

        fn(fake_self, 'frame_width_id')

        assert ('FRAME_WIDTH', '800') in records, (
            f'the typed width left no record of its own: {records}'
        )
        assert applied, 'a parseable entry must still reach the camera'
        assert records[0] == ('FRAME_WIDTH', '800'), (
            'the typed value must be the first thing recorded, so a bundle '
            f'reads in the order the user acted. Got {records}'
        )

    def test_the_committed_box_names_itself(self):
        fn, fake_self, records, _applied, _settings, boxes = self._make('800')
        boxes['frame_height_id'].text = '640'

        fn(fake_self, 'frame_height_id')

        assert records[0] == ('FRAME_HEIGHT', '640'), (
            f'the height box reported under the wrong name: {records}'
        )

    def test_a_blank_entry_is_reported_as_a_correction_and_applies_nothing(self):
        fn, fake_self, records, applied, _settings, _boxes = self._make('', unparseable=True)

        fn(fake_self, 'frame_width_id')

        assert ('FRAME_WIDTH', '') in records, (
            f'the blank entry itself was never recorded: {records}'
        )
        assert ('FRAME_WIDTH_APPLIED', '768') in records, (
            f'the correction was not reported under _APPLIED: {records}'
        )
        assert applied == [], (
            'an unparseable entry must not be applied -- substituting the '
            f'stored size reports a framing the user never asked for: {applied}'
        )

    def test_a_blank_entry_puts_both_boxes_back(self):
        fn, fake_self, _records, _applied, settings, boxes = self._make('', unparseable=True)

        fn(fake_self, 'frame_width_id')

        assert boxes['frame_width_id'].text == str(settings['frame']['width'])
        assert boxes['frame_height_id'].text == str(settings['frame']['height'])

    def test_a_disconnected_camera_still_records_the_entry(self):
        """The user typed it whether or not a camera was there to hear it.

        That an absent camera stores nothing is the Session's answer, pinned
        in tests/test_the_session_applies_and_stores_camera_settings.py.
        """
        fn, fake_self, records, _applied, _settings, _boxes = self._make(
            '800', camera_present=False
        )

        fn(fake_self, 'frame_width_id')

        assert ('FRAME_WIDTH', '800') in records, (
            f'a frame edit with no camera left no trace at all: {records}'
        )
