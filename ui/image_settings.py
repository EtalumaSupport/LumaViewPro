# Copyright Etaluma, Inc.
import functools
import logging

from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import ColorProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.layer_record as layer_record
from modules import gui_logger
from modules.config_ui_getters import (
    camera_autogain_supported,
    get_layer_exposure_slider_max,
    get_layer_illumination_slider_max,
)
from ui.ui_helpers import LoggedAccordionItem, resort_accordion

logger = logging.getLogger('LVP.ui.image_settings')


# ============================================================================
# Accordion Item Widgets (Layer/Channel Selection)
# ============================================================================


class AccordionItemXyStageControl(LoggedAccordionItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_gui(self, full_redraw: bool = False):
        self.ids['xy_stagecontrol_id'].update_gui(full_redraw=full_redraw)


class ChannelDrawer(LoggedAccordionItem):
    """One layer's drawer: its controls, titled by the layer's label beside a
    swatch of its colour.

    One class for every catalogue key. What differs between layers is the
    layer's record, applied by ``ImageSettings.show_drawers``; a key added to
    the catalogue gets a drawer with no code naming it.
    """

    layer = StringProperty('')
    # The layer's display colour as RGBA, the title's swatch.
    color_rgba = ColorProperty((0.5, 0.5, 0.5, 1.0))
    # The controls' chrome: the display colour lightened, so a dark colour's
    # headings stay legible on the panel's dark ground.
    chrome_rgba = ColorProperty((0.5, 0.5, 0.5, 1.0))


# How far a drawer's chrome moves from its layer's colour toward white:
# pure blue becomes (0.4, 0.4, 1.0), readable as text on the dark panel.
_CHROME_LIGHTENING = 0.4


def _lightened(color: layer_record.Color) -> layer_record.Color:
    return tuple(weight + (1.0 - weight) * _CHROME_LIGHTENING for weight in color)


# ============================================================================
# ImageSettings -- Right Sidebar Panel (Channel Controls, LED, Exposure)
# ============================================================================


class ImageSettings(BoxLayout):
    settings_width = dp(300)
    tab_width = dp(30)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug('[LVP Main  ] ImageSettings.__init__()')
        # One drawer per catalogue key, in display order; a drawer is in the
        # accordion while the unit has its layer (show_drawers).
        self._drawers = {layer: ChannelDrawer(layer=layer) for layer in common_utils.get_layers()}
        self._shown_layers: frozenset[str] = frozenset()
        self._init_ui_retries = 0
        Clock.schedule_once(self._init_ui, 0)

    def layer_lookup(self, layer: str):
        return self._drawers[layer].ids['control']

    def accordion_item_lookup(self, layer: str):
        return self._drawers[layer]

    def draw_shown_channel(self, *largs) -> None:
        """Open the drawer of the Session's shown channel and close the others.

        The drawers display the shown channel, in a run too: this is their
        one writer, called at start-up and by the listener bridge on each
        change, and after a person's open in case the Session refused it.
        Cleans up ScrollView viewport textures on collapse to prevent memory
        accumulation.
        """
        layer = _app_ctx.ctx.session.shown_channel.channel
        for a_layer in common_utils.get_layers():
            accordion_item_obj = self.accordion_item_lookup(layer=a_layer)
            if layer == a_layer:
                accordion_item_obj.collapse = False
                continue
            if not accordion_item_obj.collapse:
                layer_control = self.layer_lookup(layer=a_layer)
                for child in layer_control.walk():
                    if isinstance(child, ScrollView):
                        # Schedule cleanup after collapse animation completes
                        from ui.ui_helpers import cleanup_scrollview_viewport

                        Clock.schedule_once(
                            lambda dt, sv=child: cleanup_scrollview_viewport(sv), 0.2
                        )
            accordion_item_obj.collapse = True
        gui_logger.display('OPEN_LAYER', layer)
        self.layer_lookup(layer=layer).show_as_shown()

    def _show_channel_for_person(self, layer: str) -> None:
        """A person opened ``layer``'s drawer: the Session shows that layer.

        The drawer opens at once; the Session's answer redraws it, so a
        refused show puts the drawer back on the channel the Session shows.
        """
        session = _app_ctx.ctx.session
        from ui.ui_helpers import submit_reported

        submit_reported(
            lambda: session.show_channel(layer),
            self.draw_shown_channel,
            f'SHOW_CHANNEL_{layer}',
        )

    def show_drawers(self, identity) -> None:
        """Show a drawer for each layer the unit has, drawn from its record,
        and none for the others.

        Visibility, titles and controls are per layer from the identity, so a
        filterset carrying only some channels (an Ex488-only unit) shows
        exactly what the unit has. A drawer is titled by its layer's label
        beside a swatch of its colour, and has illumination controls where
        the unit's row drives an LED. A layer the unit lacks stops being
        acquired.
        """
        records = {record.key_name: record for record in identity.channels}
        colors = _app_ctx.ctx.session.channel_colors()
        for layer, drawer in self._drawers.items():
            record = records.get(layer)
            if record is None:
                _app_ctx.ctx.session.set_channel_acquire(layer, None)
                drawer.collapse = True
                continue
            color = colors[layer]
            drawer.color_rgba = (*color, 1.0)
            drawer.chrome_rgba = (*_lightened(color), 1.0)
            drawer.title = record.label
            control = drawer.ids['control']
            control.illumination_support = bool(record.led_addresses)
            # Luminescence is long and dim: its exposure sums frames, and
            # the camera's auto gain is not offered for it.
            luminescence = record.modality == layer_record.Modality.LUMINESCENCE
            control.exposure_summing_support = luminescence
            control.autogain_support = not luminescence
        self._shown_layers = frozenset(records)
        self._resort_accordion()

    def set_layer_focus_visibility(self, visible: bool) -> None:
        for layer in common_utils.get_layers():
            self.layer_lookup(layer=layer).focus_support = visible

    def _resort_accordion(self):
        """Put the layer accordion back in catalogue order after a model switch.

        Live scope-model transitions (LS620 -> LS850, etc.) change which
        drawers are shown; each change puts them back in order. Display order
        IS the catalogue order (a layer's id is its position there), so no
        second order is authored here.
        """
        accordion = self.ids.get('accordion_id') if hasattr(self, 'ids') else None
        if accordion is None:
            return
        resort_accordion(
            accordion,
            [(drawer, layer in self._shown_layers) for layer, drawer in self._drawers.items()],
        )

    def _init_ui(self, dt=0):
        ctx = _app_ctx.ctx
        if ctx is None:
            self._init_ui_retries += 1
            if self._init_ui_retries > 50:
                logger.error(
                    '[LVP Main  ] ImageSettings._init_ui: ctx still None after 50 retries, giving up'
                )
                return
            Clock.schedule_once(self._init_ui, 0.1)
            return
        for layer in common_utils.get_layers():
            self.accordion_item_lookup(layer=layer).person_opened = functools.partial(
                self._show_channel_for_person, layer
            )
        self.sync_camera_capability_ranges()
        self.enable_image_stats_if_needed()

    def enable_image_stats_if_needed(self):
        if _app_ctx.ctx.session.engineering_mode:
            for layer in common_utils.get_layers():
                layer_obj = self.layer_lookup(layer=layer)
                layer_obj.ids['image_stats_mean_id'].height = '30dp'
                layer_obj.ids['image_stats_stddev_id'].height = '30dp'
                layer_obj.ids['image_af_score_id'].height = '30dp'

    def set_layer_exposure_ranges(self):
        """Size each layer's exposure slider from the connected camera's cap,
        through the getter that also applies the transmitted-layer policy.

        The one owner of exp_slider.max. The policy numbers live beside the
        illumination policy below the GUI rather than as literals here, so the
        bound a non-GUI caller is held to and the bound the slider shows come
        from the same place.
        """
        camera_max_ms = _app_ctx.ctx.max_exposure
        for layer in common_utils.get_fluorescence_layers():
            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.ids[
                'exp_slider'
            ].min = 1.0  # 1ms floor -- sub-ms never realistic for fluorescence
            layer_obj.ids['exp_slider'].max = get_layer_exposure_slider_max(camera_max_ms, layer)
            layer_obj.ids['exp_slider'].step = 1.0  # Integer steps only

        for layer in common_utils.get_transmitted_layers():
            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.ids['exp_slider'].max = get_layer_exposure_slider_max(camera_max_ms, layer)

        for layer in common_utils.get_luminescence_layers():
            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.ids['exp_slider'].min = 1.0  # 1ms floor
            layer_obj.ids['exp_slider'].max = get_layer_exposure_slider_max(camera_max_ms, layer)
            layer_obj.ids['exp_slider'].step = 1.0  # Integer steps only

    def set_layer_gain_ranges(self):
        """Size each layer's gain slider to the connected camera's cap.

        Parallel to set_layer_exposure_ranges. Pre-fix the gain slider
        was hardcoded 0-48 dB in the kv regardless of camera -- that
        let LS620 users drag past the usable range and black out the
        image. `ctx.max_gain` is populated from
        capabilities.camera_limits in load_settings (and the same
        default-fallback pattern as exposure).
        """
        ctx = _app_ctx.ctx
        for layer in common_utils.get_layers():
            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.ids['gain_slider'].max = ctx.max_gain

    def set_layer_illumination_ranges(self):
        """Size each layer's illumination slider from the connected LED
        driver's cap, through the getter that also applies the transmitted-
        layer policy. The one owner of ill_slider.max; the .kv value is the
        placeholder until the scope is built. A scope with no LED board has
        no cap to size from, and its illumination controls are hidden
        (set_layer_led_controller_support).
        """
        for layer in common_utils.get_layers():
            layer_obj = self.layer_lookup(layer=layer)
            if not layer_obj.led_controller_support:
                continue
            bound = get_layer_illumination_slider_max(layer)
            if bound is None:
                continue
            layer_obj.ids['ill_slider'].max = bound

    def set_layer_autogain_support(self):
        """Gate the Auto Gain/Exp control on the camera's hardware AG/AE support.

        The checkbox drives hardware auto-gain/exposure (camera.set_auto_gain
        -> driver.auto_gain), so on a camera without either (IDS U3-34Lx, FX2
        LS620) the control is a no-op and is hidden. Parallel to
        set_layer_gain_ranges: runs at every capability-sync point (connect /
        scope-change). The per-layer static rule (Lumi has no autogain) is
        preserved -- this only drives the orthogonal camera_autogain_support
        gate, AND-ed with autogain_support in the kv. Capability resolution +
        fail-safe live in camera_autogain_supported() (the single gate).

        Only the visibility gate is set here; the persisted per-layer auto_gain
        is NOT mutated. The enable actually in force is the API's answer
        (camera.applied_auto_gain_for), read at the consumption point
        (LayerControl.effective_auto_gain), so a capable camera's saved
        preference survives a swap to an AG-less body and back.
        """
        supported = camera_autogain_supported()
        for layer in common_utils.get_layers():
            layer_obj = self.layer_lookup(layer=layer)
            layer_obj.camera_autogain_support = supported

    def set_layer_led_controller_support(self):
        """Hide every layer's LED toggle and illumination current when the
        scope came up without its LED board: its capabilities carry no LED
        channels. Display of the API's answer, set where the capabilities
        are synced.
        """
        present = _app_ctx.ctx.scope.capabilities.led_addresses is not None
        for layer in common_utils.get_layers():
            self.layer_lookup(layer=layer).led_controller_support = present

    def set_camera_controls_support(self):
        """Disable the camera controls -- each layer's gain, exposure and
        auto-gain, and the frame size and binning -- when the scope has no
        camera connected. Display of the API's answer, set where the
        capabilities are synced.
        """
        ctx = _app_ctx.ctx
        connected = ctx.scope.camera_connected
        for layer in common_utils.get_layers():
            self.layer_lookup(layer=layer).camera_connected = connected
        ctx.motion_settings.ids['microscope_settings_id'].camera_connected = connected

    def sync_camera_capability_ranges(self):
        """Resync every per-layer camera control from the live camera caps.

        The single grouping of the per-layer capability setters, so the connect
        path (_init_ui) and the reconnect path apply the SAME set and can't
        drift to a subset (reconnect previously refreshed only exposure ranges).
        Callers that change the camera (reconnect) must refresh ctx.max_gain /
        ctx.max_exposure first -- these setters read those caps.

        The setters run with every layer marked initializing. Narrowing a
        slider's max makes Kivy clamp its value, which fires on_value into the
        layer's handler; by the time this runs, load_settings has already
        cleared the flag, so without this the app's own bound-application would
        be recorded as a user drag and would overwrite the stored value with
        the bound. Every other programmatic widget write in LayerControl takes
        the same flag for the same reason.

        reconcile_layers_to_camera_caps stays OUTSIDE the flag: it delivers to
        the camera through each layer's apply, and that apply returns early on
        exactly this flag -- inside, it would render and push nothing.
        """
        layer_objs = [self.layer_lookup(layer=layer) for layer in common_utils.get_layers()]
        for layer_obj in layer_objs:
            layer_obj._initializing = True
        try:
            self.set_layer_exposure_ranges()
            self.set_layer_gain_ranges()
            self.set_layer_led_controller_support()
            self.set_camera_controls_support()
            self.set_layer_illumination_ranges()
            self.set_layer_autogain_support()
        finally:
            for layer_obj in layer_objs:
                layer_obj._initializing = False
        shown = layer_objs[0]
        ctx = _app_ctx.ctx
        gui_logger.display(
            'CAMERA_CONTROLS',
            f'camera={shown.camera_connected} led_board={shown.led_controller_support} '
            f'autogain={shown.camera_autogain_support} '
            f'gain_max={ctx.max_gain} exposure_max={ctx.max_exposure}',
        )
        self.reconcile_layers_to_camera_caps()

    def reconcile_layers_to_camera_caps(self):
        """Render every layer the attached camera cannot fully reach, and re-apply the open one.

        A camera swap can leave a layer's persisted gain_db/exposure_ms above
        the new body's maximum. The stored value is the user's committed
        intent, so it is NOT rewritten: the API caps what it writes to
        hardware, the channel runs at the most that body can do, and putting a
        capable camera back applies the intent again. Overwriting the store
        instead destroyed the setting silently -- the periodic current.json
        flush persisted the shrunken value and nothing recorded the original.

        The divergence itself is the trigger. Nothing is applied here: the
        Session caps each value as it puts a layer on the camera, and bring-up
        has already put the shown layer there; another layer reaches the
        camera, capped, when it is next shown.

        Runs BEFORE anything renders the store, on every path that reaches it,
        so the slider's pinned position and the box's stored value are in
        agreement the first time they are drawn.

        The re-render is explicit. It used to arrive as a side effect of
        writing the slider -- the layer's handler re-committed the value
        (crediting the user with a drag it never made).
        """
        ctx = _app_ctx.ctx
        settings = ctx.settings
        camera = ctx.lumaview.scope.camera
        for layer in common_utils.get_layers():
            gain = camera.applied_gain_db_for(settings[layer]['gain_db'])
            exposure = camera.applied_exposure_ms_for(settings[layer]['exposure_ms'])
            if gain.capped or exposure.capped:
                self.layer_lookup(layer=layer).render_layer_values_from_settings()

    # Hide (and unhide) main settings
    def toggle_settings(self):
        """A person pressed the panel's tab: record it, then place the panel."""
        # State after toggle reflects target visibility -- 'normal' = settings
        # tab going invisible (panel collapsing to side), 'down' = expanding.
        gui_logger.toggle('IMAGE_SETTINGS_PANEL', self.ids['toggle_imagesettings'].state == 'down')
        self.place_settings()

    def place_settings(self):
        """Place the panel where its tab says: open beside the view, or folded to the side."""
        if _app_ctx.ctx.session.controls_lock.run_named is None:
            self.update_transmitted()
        logger.info('[LVP Main  ] ImageSettings.place_settings()')
        ctx = _app_ctx.ctx
        lumaview = ctx.lumaview

        # move position of settings and stop histogram if main settings are collapsed
        if self.ids['toggle_imagesettings'].state == 'normal':
            self.x = lumaview.width - self.tab_width

            for layer in common_utils.get_layers():
                layer_obj = ctx.image_settings.layer_lookup(layer=layer)
                Clock.unschedule(layer_obj.ids['histo_id'].histogram)
                logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')
        else:
            self.x = lumaview.width - self.settings_width

    def update_transmitted(self):
        for layer in common_utils.get_transmitted_layers():
            layer_obj = self.layer_lookup(layer=layer)

            # Remove 'Colorize' option in transmitted channels control
            # -----------------------------------------------------
            # Remove CBT from transmitted channel control
            layer_obj.show_cbt = False
            label = layer_obj.ids['composite_threshold_label']
            slider = layer_obj.ids['composite_threshold_slider']
            text = layer_obj.ids['composite_threshold_text']
            label.text = ''
            label.visible = False
            label.opacity = 0
            slider.disabled = True
            slider.visible = False
            slider.cursor_size = '0dp', '0dp'
            slider.opacity = 0
            slider.value_track_color = (0.0,) * 4
            text.disabled = True
            text.visible = False
            text.width = '0dp'
            text.text = ''
            text.opacity = 0
            layer_obj.ids['false_color_label'].text = ''
            layer_obj.ids['false_color'].color = (0.0,) * 4

            layer_obj.ids['ill_slider'].step = 1

    def check_settings(self, *args):
        logger.info('[LVP Main  ] ImageSettings.check_settings()')
        lumaview = _app_ctx.ctx.lumaview
        if self.ids['toggle_imagesettings'].state == 'normal':
            self.x = lumaview.width - self.tab_width
        else:
            self.x = lumaview.width - self.settings_width


def set_histogram_layer(active_layer):
    for layer in common_utils.get_layers():
        layer_ref = _app_ctx.ctx.image_settings.layer_lookup(layer=layer)
        Clock.unschedule(layer_ref.ids['histo_id'].histogram)

        if layer == active_layer:
            Clock.schedule_interval(layer_ref.ids['histo_id'].histogram, 0.5)
            logger.info(
                f'[LVP Main  ] Clock.schedule_interval(...[{active_layer}]...histogram, 0.5)'
            )
