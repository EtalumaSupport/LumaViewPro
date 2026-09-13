"""Regression: the per-camera UI surface resyncs from the camera, uniformly.

Every per-camera UI cap and gate must come from the attached camera -- not
a drifting subset. Two confirmed gaps this pins:

  - The gain-slider cap (ctx.max_gain) was refreshed on only one of two
    bring-up paths, so a swap from a higher-cap camera to a lower one
    (e.g. LS850 -> LS620) left the gain slider over-ranged -- the user
    could drag gain past the usable range and black out the image.
  - A bring-up path re-applied settings only for a hardcoded 'BF', so a
    non-BF open layer's controls (e.g. gain/exposure sliders disabled while
    the prior camera's auto-gain was on) were never refreshed.

The fix:
  - config_helpers.camera_max_exposure_for_ui / camera_max_gain_for_ui are the
    single UI-facing cap resolvers: the live camera's cap, or the documented
    no-camera default (camera_max_* are None by design then; #616).
    load_settings resolves through them so the fallback can't be applied
    two different ways.
  - ImageSettings.sync_camera_capability_ranges groups the per-layer setters
    (exposure + gain ranges + autogain gate) AND clamp_layer_settings_to_caps,
    which reconciles each layer's stored gain_db/exposure_ms down to the new caps
    (the blackout fix, matching load_settings); _init_ui (connect) calls that
    grouping. A scope swap (the auto-reconnect item) re-runs it and re-applies
    the VISIBLE layer (ImageSettings.open_or_default_layer), never a hardcoded
    channel -- its acceptance list lives with that item.

The UI modules touch Kivy widgets and cannot be imported under the test mocks
(see test_ids_native_roi_sync_binning), so the wiring is pinned with AST
guards; the pure cap resolvers are exercised directly.
"""

import ast
import inspect
import pathlib
from types import SimpleNamespace

from modules.config_helpers import (
    BF_MAX_MANUAL_EXPOSURE_MS,
    DEFAULT_MAX_EXPOSURE_MS,
    DEFAULT_MAX_GAIN_DB,
    FLUORESCENCE_MAX_MANUAL_EXPOSURE_MS,
    TRANSMITTED_MAX_MANUAL_EXPOSURE_MS,
    camera_max_exposure_for_ui,
    camera_max_gain_for_ui,
    layer_max_exposure_ms_for_ui,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGE_SETTINGS_PATH = REPO_ROOT / 'ui' / 'image_settings.py'
LAYER_CONTROL_PATH = REPO_ROOT / 'ui' / 'layer_control.py'
MS_PATH = REPO_ROOT / 'ui' / 'microscope_settings.py'


def _method_node(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {path}')


def _name_calls(method: ast.FunctionDef, name: str):
    return [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    ]


def _attr_calls(method: ast.FunctionDef, attr: str):
    return [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
    ]


class TestCapResolvers:
    """The single UI-facing cap resolvers apply the documented #616 fallback."""

    def test_exposure_returns_live_cap_when_present(self):
        imaging = SimpleNamespace(max_exposure_ms_cached=500.0)
        assert camera_max_exposure_for_ui(imaging) == 500.0

    def test_exposure_falls_back_when_no_camera(self):
        imaging = SimpleNamespace(max_exposure_ms_cached=None)
        assert camera_max_exposure_for_ui(imaging) == DEFAULT_MAX_EXPOSURE_MS

    def test_gain_returns_live_cap_when_present(self):
        imaging = SimpleNamespace(max_gain_db_cached=24.0)
        assert camera_max_gain_for_ui(imaging) == 24.0

    def test_gain_falls_back_when_no_camera(self):
        imaging = SimpleNamespace(max_gain_db_cached=None)
        assert camera_max_gain_for_ui(imaging) == DEFAULT_MAX_GAIN_DB


class TestSyncGrouping:
    """sync_camera_capability_ranges groups all three per-layer setters."""

    def test_grouping_calls_all_setters_and_clamp(self):
        method = _method_node(IMAGE_SETTINGS_PATH, 'sync_camera_capability_ranges')
        for setter in (
            'set_layer_exposure_ranges',
            'set_layer_gain_ranges',
            'set_layer_autogain_support',
            'clamp_layer_settings_to_caps',
        ):
            assert _attr_calls(method, setter), f'sync_camera_capability_ranges must call {setter}.'

    def test_clamp_reconciles_stored_gain_and_exposure_to_caps(self):
        # The blackout fix: a stored gain_db/exposure_ms above the new camera's cap
        # must be brought down to the cap (and persisted) for every layer, so a
        # downshift swap can't push an over-cap value that blacks out.
        method = _method_node(IMAGE_SETTINGS_PATH, 'clamp_layer_settings_to_caps')
        clamped = {
            t.slice.value
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value in ('gain_db', 'exposure_ms')
        }
        assert clamped == {'gain_db', 'exposure_ms'}, (
            'clamp_layer_settings_to_caps must reconcile both stored gain_db and '
            'exposure_ms down to the camera caps.'
        )

    def test_init_ui_uses_the_grouping(self):
        method = _method_node(IMAGE_SETTINGS_PATH, '_init_ui')
        assert _attr_calls(method, 'sync_camera_capability_ranges'), (
            'ImageSettings._init_ui must resync via sync_camera_capability_ranges.'
        )

    def test_open_or_default_layer_reuses_helper_and_defaults_to_bf(self):
        method = _method_node(IMAGE_SETTINGS_PATH, 'open_or_default_layer')
        # Delegates the open-layer scan to the shared guarded helper, and
        # defaults to BF when none is open.
        assert _attr_calls(method, 'get_opened_layer') or any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == 'get_opened_layer'
            for n in ast.walk(method)
        ), 'open_or_default_layer must reuse common_utils.get_opened_layer.'
        returns_bf = any(
            isinstance(n, ast.Return)
            and isinstance(n.value, ast.Constant)
            and n.value.value == 'BF'
            for n in ast.walk(method)
        )
        assert returns_bf, 'open_or_default_layer must default to BF when no layer is open.'


class TestLoadSettingsResync:
    """load_settings resolves the caps and the clamp through the shared owners."""

    def test_load_settings_uses_the_cap_resolvers(self):
        # The de-fragmentation: load_settings resolves caps through the same
        # helpers, not its own inline `or DEFAULT`.
        method = _method_node(MS_PATH, 'load_settings')
        assert _name_calls(method, 'camera_max_exposure_for_ui'), (
            'load_settings must resolve the exposure cap via camera_max_exposure_for_ui.'
        )
        assert _name_calls(method, 'camera_max_gain_for_ui'), (
            'load_settings must resolve the gain cap via camera_max_gain_for_ui.'
        )

    def test_load_settings_delegates_clamp_not_inline(self):
        # De-dup: load_settings reconciles over-cap values via the single
        # clamp_layer_settings_to_caps owner, not a duplicate inline clamp.
        method = _method_node(MS_PATH, 'load_settings')
        assert _attr_calls(method, 'clamp_layer_settings_to_caps'), (
            'load_settings must delegate over-cap reconciliation to clamp_layer_settings_to_caps.'
        )
        inline = [
            t
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value in ('gain_db', 'exposure_ms')
        ]
        assert not inline, (
            'load_settings must not carry an inline gain_db/exposure_ms clamp-persist '
            '(it duplicates clamp_layer_settings_to_caps).'
        )


class TestCapabilitySyncDoesNotImpersonateTheUser:
    """The app's own bound-application must not read back as a user drag.

    Narrowing a slider's max makes Kivy clamp its value, which fires on_value
    into LayerControl's handler. By the time sync_camera_capability_ranges
    runs, load_settings has already cleared the per-layer _initializing flag
    (it clears it in sync_widgets_from_settings' finally), so the handler is
    live: it rewrote settings[layer][...] with the bound and emitted a SLIDER
    record crediting the user. Measured in the simulator with nobody touching
    the app -- a stored BF illumination of 500 became 50 in current.json, and
    a stored DF exposure of 500 became 200, each with a matching SLIDER line.

    clamp_layer_settings_to_caps must stay outside the flag: it reconciles a
    value the hardware cannot honor, which is a real settings change.
    """

    def _sync_method(self):
        return _method_node(IMAGE_SETTINGS_PATH, 'sync_camera_capability_ranges')

    def test_setters_run_under_the_initializing_flag(self):
        method = self._sync_method()
        try_nodes = [n for n in ast.walk(method) if isinstance(n, ast.Try)]
        assert try_nodes, (
            'sync_camera_capability_ranges must run its setters inside try/finally '
            'so the _initializing flag is cleared even if a setter raises.'
        )
        guarded = {
            n.func.attr
            for try_node in try_nodes
            for n in ast.walk(try_node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr.startswith('set_layer_')
        }
        for setter in (
            'set_layer_exposure_ranges',
            'set_layer_gain_ranges',
            'set_layer_illumination_ranges',
            'set_layer_autogain_support',
        ):
            assert setter in guarded, (
                f'{setter} must run inside the _initializing guard -- outside it, '
                'the Kivy max-clamp reaches the layer handler and is recorded as a drag.'
            )

    def test_the_flag_is_set_and_cleared(self):
        method = self._sync_method()
        writes = [
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Attribute) and t.attr == '_initializing'
        ]
        values = {n.value.value for n in writes if isinstance(n.value, ast.Constant)}
        assert values == {True, False}, (
            'sync_camera_capability_ranges must both set and clear _initializing; '
            f'found {values or "no writes"}.'
        )
        finallys = [n for n in ast.walk(method) if isinstance(n, ast.Try) and n.finalbody]
        cleared_in_finally = any(
            isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Constant)
            and n.value.value is False
            for try_node in finallys
            for stmt in try_node.finalbody
            for n in ast.walk(stmt)
        )
        assert cleared_in_finally, 'The flag must be cleared in a finally, not on the happy path.'

    def test_clamp_stays_outside_the_guard(self):
        method = self._sync_method()
        in_try = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'clamp_layer_settings_to_caps'
            for try_node in [x for x in ast.walk(method) if isinstance(x, ast.Try)]
            for n in ast.walk(try_node)
        )
        assert not in_try, (
            'clamp_layer_settings_to_caps must stay OUTSIDE the _initializing guard -- '
            'it reconciles a value the hardware cannot honor, which is a real change.'
        )
        assert _attr_calls(method, 'clamp_layer_settings_to_caps'), (
            'sync_camera_capability_ranges must still run the clamp.'
        )


class TestManualExposurePolicy:
    """The transmitted layers' manual exposure ceilings, which nothing asserted.

    BF / PC / DF light is bright enough that the useful manual range sits far
    under the sensor's maximum, so their sliders stop below the camera cap.
    The numbers lived as literals in ui/image_settings.py with no home below
    the GUI and no test, so a non-GUI caller was held to a different bound
    than the slider showed -- and nothing would have caught a change to either.
    """

    def test_bf_stops_at_its_own_ceiling(self):
        assert layer_max_exposure_ms_for_ui(10_000.0, 'BF') == BF_MAX_MANUAL_EXPOSURE_MS

    def test_the_other_transmitted_layers_stop_higher(self):
        for layer in ('PC', 'DF'):
            assert (
                layer_max_exposure_ms_for_ui(10_000.0, layer) == TRANSMITTED_MAX_MANUAL_EXPOSURE_MS
            ), layer

    def test_a_lower_camera_cap_wins(self):
        # The policy narrows the camera; it never widens past what the body can do.
        assert layer_max_exposure_ms_for_ui(30.0, 'BF') == 30.0
        assert layer_max_exposure_ms_for_ui(120.0, 'DF') == 120.0

    def test_fluorescence_stops_at_its_own_ceiling(self):
        for layer in ('Blue', 'Green', 'Red'):
            assert (
                layer_max_exposure_ms_for_ui(10_000.0, layer) == FLUORESCENCE_MAX_MANUAL_EXPOSURE_MS
            ), layer

    def test_luminescence_alone_gets_the_camera_cap(self):
        """The exemption, asserted on its own so it cannot be folded away.

        Integrating as long as the sensor allows is what the channel is for, so
        Lumi is the one layer with no manual ceiling. Every other class narrows.
        """
        assert layer_max_exposure_ms_for_ui(10_000.0, 'Lumi') == 10_000.0

    def test_a_body_that_caps_low_narrows_every_class(self):
        """A camera whose own cap sits under the policy ceilings.

        The FX2 boards cap exposure at 178 ms -- above the per-frame readout
        time the sensor inserts blanking rows, which changes the byte rate
        mid-stream and desyncs the frame parser. Nothing may hand any layer a
        bound above what the attached body will honor.
        """
        for layer in ('BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi'):
            assert layer_max_exposure_ms_for_ui(178.0, layer) <= 178.0, layer
        assert layer_max_exposure_ms_for_ui(178.0, 'BF') == BF_MAX_MANUAL_EXPOSURE_MS
        assert layer_max_exposure_ms_for_ui(178.0, 'PC') == 178.0
        assert layer_max_exposure_ms_for_ui(178.0, 'Red') == 178.0
        assert layer_max_exposure_ms_for_ui(178.0, 'Lumi') == 178.0

    def test_the_manual_ceiling_is_not_the_auto_ceiling(self):
        # DEFAULT_AG_AE_MAX_EXPOSURE_MS bounds what the AUTO loop may drive to
        # and is overridable per install; reusing it here would let auto-exposure
        # tuning silently resize the manual slider. The resolver takes no settings
        # argument, so no install override can reach it.
        assert 'settings' not in inspect.signature(layer_max_exposure_ms_for_ui).parameters
        assert 'overrides' not in inspect.signature(layer_max_exposure_ms_for_ui).parameters


class TestTypedExposureCeiling:
    """The exposure TEXT box is bounded by the camera, not by its slider.

    The slider's range is a manual convenience range, deliberately narrower
    than the sensor on most classes. The box is the physical limit, so the
    two bounds cannot share a source: reading the slider's max made the box
    inherit a policy number, and brightfield read a GUI constant that matched
    no camera at all -- on a body whose real cap is 178 ms that constant let a
    user store an exposure the sensor silently clamped away.
    """

    def _exp_text(self) -> ast.FunctionDef:
        return _method_node(LAYER_CONTROL_PATH, 'exp_text')

    def test_the_bound_comes_from_the_camera_resolver(self):
        assert _name_calls(self._exp_text(), 'get_exposure_text_max'), (
            'exp_text must resolve its upper bound through get_exposure_text_max.'
        )

    def test_no_layer_is_special_cased(self):
        """One rule for every layer: the slider carries all the narrowing.

        A layer-name comparison here is how the brightfield constant survived.
        """
        compares = [
            n
            for n in ast.walk(self._exp_text())
            if isinstance(n, ast.Compare)
            for c in n.comparators
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        ]
        assert not compares, (
            f'exp_text must not branch on a layer name; found {len(compares)} string compare(s).'
        )

    def test_the_gui_constant_is_gone(self):
        src = LAYER_CONTROL_PATH.read_text(encoding='utf-8')
        assert 'BF_MAX_EXPOSURE_MS' not in src, (
            'A GUI-side exposure ceiling constant matches no camera; the bound '
            'belongs to the attached body.'
        )

    def test_no_camera_means_no_ceiling(self):
        """None, not a substituted default.

        camera_max_exposure_for_ui answers the no-camera case with
        DEFAULT_MAX_EXPOSURE_MS so a slider always has some range to draw.
        Reusing it here would raise the typed ceiling on a camera drop and
        re-open exactly the divergence this resolver closes.
        """
        import modules.app_context as _app_ctx
        from modules.config_ui_getters import get_exposure_text_max

        def _ctx_with(lumaview):
            return SimpleNamespace(lumaview=lumaview)

        def _scope_reporting(cap):
            return SimpleNamespace(
                scope=SimpleNamespace(imaging=SimpleNamespace(max_exposure_ms_cached=cap))
            )

        prior = _app_ctx.ctx
        try:
            # No scope built yet.
            _app_ctx.ctx = _ctx_with(None)
            assert get_exposure_text_max() is None

            # A scope whose camera reports no cap still has no honest ceiling.
            _app_ctx.ctx = _ctx_with(_scope_reporting(None))
            assert get_exposure_text_max() is None

            # A body that caps low is reported at its real cap, not a default.
            _app_ctx.ctx = _ctx_with(_scope_reporting(178.0))
            assert get_exposure_text_max() == 178.0
        finally:
            _app_ctx.ctx = prior

    def test_the_resolver_does_not_substitute_the_slider_default(self):
        """Asserted on the BODY, not the source text.

        The docstring names the resolver it rejects, and must keep naming it --
        that is the whole reason the two look interchangeable.
        """
        import modules.config_ui_getters as getters

        fn = ast.parse(inspect.getsource(getters.get_exposure_text_max)).body[0]
        called = {
            n.func.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        } | {
            n.func.attr
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert 'camera_max_exposure_for_ui' not in called, (
            'get_exposure_text_max must read the cap directly; that resolver '
            'substitutes the no-camera default and would hide a low-capping body.'
        )
