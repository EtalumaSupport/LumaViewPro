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
  - The gain and exposure sliders' maxima are the session's answer
    (ScopeSession.channel_slider_maxima, its gain maximum resolved by
    config_helpers.camera_max_gain_for_ui: the camera's cap, or None with
    none known, where the slider keeps its own), read through
    set_channel_gain_ranges and set_channel_exposure_ranges.
  - ImageSettings.sync_camera_capability_ranges groups the per-layer setters
    (exposure, gain and illumination ranges, the LED, camera and autogain
    gates) AND reconcile_channels_to_camera_caps, which re-renders each layer
    the attached camera cannot fully reach. The blackout fix lives on the
    APPLY -- the API caps what it writes to hardware -- so the stored value is
    never narrowed. load_settings runs that grouping once at start, before the
    values are shown, and _init_ui runs none of it, so the start sets each
    control in one place. A scope swap (the auto-reconnect item) re-runs it
    and re-applies the shown layer (the Session's shown channel), never a
    hardcoded channel -- its acceptance list lives with that item.

The UI modules touch Kivy widgets and cannot be imported under the test mocks
(see test_ids_native_roi_sync_binning), so the wiring is pinned with AST
guards; the pure cap resolvers are exercised directly.
"""

import ast
import inspect
import pathlib
from types import SimpleNamespace

from _lumascope.modules.config_helpers import (
    camera_max_gain_for_ui,
    channel_slider_maxima,
)
from _lumascope.modules.channel_record import release_catalogue
from tests.stock_channels import LED0, LED1, LED2

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGE_SETTINGS_PATH = REPO_ROOT / 'ui' / 'image_settings.py'
LAYER_CONTROL_PATH = REPO_ROOT / 'ui' / 'channel_control.py'
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
    """The single UI-facing gain cap resolver answers the camera's cap, or none."""

    def test_gain_returns_live_cap_when_present(self):
        imaging = SimpleNamespace(gain_max_db=24.0)
        assert camera_max_gain_for_ui(imaging) == 24.0

    def test_with_no_camera_there_is_no_gain_maximum(self):
        # No stand-in: the slider keeps its own (set_channel_gain_ranges).
        assert camera_max_gain_for_ui(None) is None


class TestSyncGrouping:
    """sync_camera_capability_ranges groups every per-layer setter and the reconcile."""

    def test_grouping_calls_all_setters_and_reconcile(self):
        method = _method_node(IMAGE_SETTINGS_PATH, 'sync_camera_capability_ranges')
        for setter in (
            'set_channel_exposure_ranges',
            'set_channel_gain_ranges',
            'set_channel_illumination_ranges',
            'set_channel_led_controller_support',
            'set_camera_controls_support',
            'set_channel_autogain_support',
            'reconcile_channels_to_camera_caps',
        ):
            assert _attr_calls(method, setter), f'sync_camera_capability_ranges must call {setter}.'

    def test_the_reconcile_never_writes_the_stored_gain_or_exposure(self):
        # A stored gain_db/exposure_ms above the attached camera's cap is the
        # user's committed intent, not an error to correct: the API caps what
        # it writes to hardware and the store is left alone, so putting a
        # capable camera back applies the intent again. Overwriting it here
        # destroyed the setting silently -- the periodic current.json flush
        # persisted the shrunken value with no record of the original.
        method = _method_node(IMAGE_SETTINGS_PATH, 'reconcile_channels_to_camera_caps')
        written = {
            t.slice.value
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Subscript)
            and isinstance(t.slice, ast.Constant)
            and t.slice.value in ('gain_db', 'exposure_ms')
        }
        assert written == set(), (
            'reconcile_channels_to_camera_caps must not write gain_db or exposure_ms '
            f'into the store; found writes to {sorted(written)}. The cap belongs on '
            'the value sent to the camera, not on the value kept for the user.'
        )


class TestTheStartSetsTheControlsOnce:
    """The start sets each layer's camera controls in one place: load_settings.

    load_settings runs the grouping before the values are shown (the guard
    ``test_load_settings_sets_the_controls_before_anything_renders`` pins the
    order); _init_ui, a Clock frame later, ran the gain and exposure sizing
    and the reconcile a second time, so a change to one pass and not the
    other went unseen.
    """

    _START_SETTERS = frozenset(
        {
            'sync_camera_capability_ranges',
            'set_channel_exposure_ranges',
            'set_channel_gain_ranges',
            'set_channel_illumination_ranges',
            'set_channel_led_controller_support',
            'set_camera_controls_support',
            'set_channel_autogain_support',
            'reconcile_channels_to_camera_caps',
        }
    )

    def test_load_settings_runs_the_grouping(self):
        method = _method_node(MS_PATH, 'load_settings')
        assert _attr_calls(method, 'sync_camera_capability_ranges'), (
            'load_settings must set the controls through sync_camera_capability_ranges.'
        )

    def test_init_ui_sets_none_of_them(self):
        method = _method_node(IMAGE_SETTINGS_PATH, '_init_ui')
        called = {
            n.func.attr
            for n in ast.walk(method)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert not called & self._START_SETTERS, (
            'ImageSettings._init_ui must not set the camera controls a second time; '
            f'it calls {sorted(called & self._START_SETTERS)}.'
        )

    def test_load_settings_writes_no_stored_gain_or_exposure(self):
        # The store is the user's intent and no load path narrows it.
        method = _method_node(MS_PATH, 'load_settings')
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
            'load_settings must not carry an inline gain_db/exposure_ms write -- the '
            'store is the user intent and no load path narrows it.'
        )


class TestCapabilitySyncDoesNotImpersonateTheUser:
    """The app's own bound-application must not read back as a user drag.

    Narrowing a slider's max makes Kivy clamp its value, which fires on_value
    into ChannelControl's handler. Unguarded, the handler rewrote
    settings[layer][...] with the bound and emitted a SLIDER record crediting
    the user. Measured in the simulator with nobody touching the app -- a
    stored BF illumination of 500 became 50 in current.json, and a stored DF
    exposure of 500 became 200, each with a matching SLIDER line.

    The grouping marks every layer initializing around its setters and
    restores each flag as it found it: load_settings runs it while the layers
    are still initializing from their construction, and clearing the flag
    there would end that guard before the values are shown. The reconcile
    runs after the setters, once every slider carries its maximum.
    """

    def _sync_method(self):
        return _method_node(IMAGE_SETTINGS_PATH, 'sync_camera_capability_ranges')

    def _the_try(self):
        try_nodes = [n for n in ast.walk(self._sync_method()) if isinstance(n, ast.Try)]
        assert len(try_nodes) == 1, (
            'sync_camera_capability_ranges must run its setters inside one try/finally '
            'so the _initializing flag is restored even if a setter raises.'
        )
        return try_nodes[0]

    def test_setters_run_under_the_initializing_flag(self):
        guarded = {
            n.func.attr
            for n in ast.walk(self._the_try())
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr.startswith('set_')
        }
        for setter in (
            'set_channel_exposure_ranges',
            'set_channel_gain_ranges',
            'set_channel_illumination_ranges',
            'set_channel_autogain_support',
        ):
            assert setter in guarded, (
                f'{setter} must run inside the _initializing guard -- outside it, '
                'the Kivy max-clamp reaches the layer handler and is recorded as a drag.'
            )

    def test_the_flag_is_set_and_restored_as_found(self):
        writes = [
            n
            for n in ast.walk(self._sync_method())
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Attribute) and t.attr == '_initializing'
        ]
        assert any(isinstance(n.value, ast.Constant) and n.value.value is True for n in writes), (
            'sync_camera_capability_ranges must mark every layer initializing.'
        )
        restores = [
            n
            for stmt in self._the_try().finalbody
            for n in ast.walk(stmt)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == '_initializing' for t in n.targets)
        ]
        assert restores, 'The flag must be put back in the finally, not on the happy path.'
        assert not [n for n in restores if isinstance(n.value, ast.Constant)], (
            'The finally must restore each flag as it was found, not set a constant: '
            'under load_settings the layers are still initializing from construction.'
        )

    def test_the_reconcile_runs_after_the_setters(self):
        the_try = self._the_try()
        reconciles = [
            n
            for n in ast.walk(self._sync_method())
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'reconcile_channels_to_camera_caps'
        ]
        assert reconciles, 'sync_camera_capability_ranges must still run the reconcile.'
        assert all(n.lineno > the_try.end_lineno for n in reconciles), (
            'reconcile_channels_to_camera_caps must run after the setters, once every '
            'slider carries its maximum.'
        )


def layer_max_exposure_ms(camera_max_ms: float, layer: str) -> float:
    """The exposure slider maximum the session publishes for ``layer`` under a
    camera capped at ``camera_max_ms``."""
    return channel_slider_maxima(
        layer, exposure_cap_ms=camera_max_ms, illumination_cap_ma=None, gain_limits=None
    ).slider_max_exposure_ms


class TestManualExposurePolicy:
    """The transmitted layers' manual exposure ceilings, which nothing asserted.

    BF / PC / DF light is bright enough that the useful manual range sits far
    under the sensor's maximum, so their sliders stop below the camera cap.
    The numbers lived as literals in ui/image_settings.py with no home below
    the GUI and no test, so a non-GUI caller was held to a different bound
    than the slider showed -- and nothing would have caught a change to either.
    """

    def test_bf_stops_at_its_own_ceiling(self):
        assert layer_max_exposure_ms(10_000.0, 'BF') == 50.0

    def test_the_other_transmitted_layers_stop_higher(self):
        for layer in ('PC', 'DF'):
            assert layer_max_exposure_ms(10_000.0, layer) == 200.0, layer

    def test_a_lower_camera_cap_wins(self):
        # The policy narrows the camera; it never widens past what the body can do.
        assert layer_max_exposure_ms(30.0, 'BF') == 30.0
        assert layer_max_exposure_ms(120.0, 'DF') == 120.0

    def test_fluorescence_stops_at_its_own_ceiling(self):
        for layer in (LED0, LED1, LED2):
            assert layer_max_exposure_ms(10_000.0, layer) == 1000.0, layer

    def test_luminescence_alone_gets_the_camera_cap(self):
        """The exemption, asserted on its own so it cannot be folded away.

        Integrating as long as the sensor allows is what the channel is for, so
        Lumi is the one layer with no manual ceiling. Every other class narrows.
        """
        assert layer_max_exposure_ms(10_000.0, 'Lumi') == 10_000.0

    def test_a_body_that_caps_low_narrows_every_class(self):
        """A camera whose own cap sits under the policy ceilings.

        A body may cap exposure under a layer's policy ceiling (the FX2
        boards once capped at 178 ms). Nothing may hand any layer a bound
        above what the attached body will honor.
        """
        for layer in release_catalogue():
            assert layer_max_exposure_ms(178.0, layer) <= 178.0, layer
        assert layer_max_exposure_ms(178.0, 'BF') == 50.0
        assert layer_max_exposure_ms(178.0, 'PC') == 178.0
        assert layer_max_exposure_ms(178.0, LED2) == 178.0
        assert layer_max_exposure_ms(178.0, 'Lumi') == 178.0

    def test_the_manual_ceiling_is_not_the_auto_ceiling(self):
        # DEFAULT_AG_AE_MAX_EXPOSURE_MS bounds what the AUTO loop may drive to
        # and is overridable per install; reusing it here would let auto-exposure
        # tuning silently resize the manual slider. The resolver takes no settings
        # argument, so no install override can reach it.
        assert 'settings' not in inspect.signature(channel_slider_maxima).parameters
        assert 'overrides' not in inspect.signature(channel_slider_maxima).parameters


class TestTheTypedExposureHasNoWidgetCeiling:
    """The exposure TEXT box carries no ceiling of its own; the camera's is applied at the apply.

    The slider's range is a manual convenience range, deliberately narrower
    than the sensor on most classes, and it narrows nothing typed. The box
    used to read a ceiling from a GUI constant that matched no camera, then
    from a resolver of the live camera's cap, and clip the typed value to
    it; both put a second copy of the sensor's limit in the widget. The
    typed value is now stored as the layer's intent and the camera is
    driven at its own limit when the layer is applied
    (``ImagingAPI.applied_exposure_ms_for``), the same as a stored value
    loaded from the file (Eric, 2026-10-08).
    """

    def _exp_text(self) -> ast.FunctionDef:
        return _method_node(LAYER_CONTROL_PATH, 'exp_text')

    def test_the_box_passes_no_ceiling_to_the_handler(self):
        keywords = {
            kw.arg
            for n in ast.walk(self._exp_text())
            if isinstance(n, ast.Call)
            for kw in n.keywords
        }
        assert 'value_max' not in keywords, 'exp_text must not hand the handler a ceiling.'
        from tests.ast_seams import parse_module

        named = {
            getattr(n, 'id', None) or getattr(n, 'attr', None) or getattr(n, 'name', None)
            for n in ast.walk(parse_module('ui/channel_control.py'))
        }
        assert 'get_exposure_text_max' not in named, (
            'ui/channel_control.py: the typed-exposure ceiling resolver is retired; the apply caps.'
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
