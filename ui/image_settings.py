# Copyright Etaluma, Inc.
import functools
import logging

from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import ColorProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

import _lumascope.modules.app_context as _app_ctx
import _lumascope.modules.common_utils as common_utils
import _lumascope.modules.channel_record as channel_record
from _lumascope.modules import gui_logger
from ui.ui_helpers import LoggedAccordionItem, resort_accordion

logger = logging.getLogger('LVP.ui.image_settings')


# ============================================================================
# Accordion Item Widgets (Channel Selection)
# ============================================================================


class AccordionItemXyStageControl(LoggedAccordionItem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_gui(self, full_redraw: bool = False):
        self.ids['xy_stagecontrol_id'].update_gui(full_redraw=full_redraw)


class ChannelDrawer(LoggedAccordionItem):
    """One channel's drawer: its controls, titled by the channel's label (a
    fluorescence channel's as its excitation) beside a swatch of its colour.

    One class for every catalogue key. What differs between channels is the
    channel's record, applied by ``ImageSettings.show_drawers``; a key added to
    the catalogue gets a drawer with no code naming it.
    """

    channel = StringProperty('')
    # The channel's display colour as RGBA, the title's swatch.
    color_rgba = ColorProperty((0.5, 0.5, 0.5, 1.0))
    # The controls' chrome: the display colour lightened, so a dark colour's
    # headings stay legible on the panel's dark ground.
    chrome_rgba = ColorProperty((0.5, 0.5, 0.5, 1.0))


# How far a drawer's chrome moves from its channel's colour toward white:
# pure blue becomes (0.4, 0.4, 1.0), readable as text on the dark panel.
_CHROME_LIGHTENING = 0.4


def _lightened(color: channel_record.Color) -> channel_record.Color:
    return tuple(weight + (1.0 - weight) * _CHROME_LIGHTENING for weight in color)


def _drawer_title(record: channel_record.ChannelRecord) -> str:
    """A drawer's title: a fluorescence channel's label is a bare wavelength
    (``'488 nm'``), so its drawer says what that wavelength is; any other
    channel's label is already its name. The label itself is what files
    record, so only the title says more."""
    if record.modality == channel_record.Modality.FLUORESCENCE:
        return f'{record.label} Excitation'
    return record.label


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
        # accordion while the unit has its channel (show_drawers).
        self._drawers = {
            channel: ChannelDrawer(channel=channel) for channel in common_utils.get_channels()
        }
        self._shown_channels: frozenset[str] = frozenset()
        self._init_ui_retries = 0
        Clock.schedule_once(self._init_ui, 0)

    def channel_control_lookup(self, channel: str):
        return self._drawers[channel].ids['control']

    def accordion_item_lookup(self, channel: str):
        return self._drawers[channel]

    def draw_shown_channel(self, *largs) -> None:
        """Open the drawer of the Session's shown channel and close the others.

        The drawers display the shown channel, in a run too: this is their
        one writer, called at start-up and by the listener bridge on each
        change, and after a person's open in case the Session refused it.
        """
        channel = _app_ctx.ctx.session.shown_channel.channel
        for a_channel in common_utils.get_channels():
            accordion_item_obj = self.accordion_item_lookup(channel=a_channel)
            if channel == a_channel:
                accordion_item_obj.collapse = False
                continue
            accordion_item_obj.collapse = True
        gui_logger.display('OPEN_CHANNEL', channel)
        self.channel_control_lookup(channel=channel).show_as_shown()

    def _show_channel_for_person(self, channel: str) -> None:
        """A person opened ``channel``'s drawer: the Session shows that channel.

        The drawer opens at once; the Session's answer redraws it, so a
        refused show puts the drawer back on the channel the Session shows.
        """
        session = _app_ctx.ctx.session
        from ui.ui_helpers import submit_reported

        submit_reported(
            lambda: session.show_channel(channel),
            self.draw_shown_channel,
            f'SHOW_CHANNEL_{channel}',
        )

    def show_drawers(self, identity) -> None:
        """Show a drawer for each channel the unit has, drawn from its record,
        and none for the others.

        Visibility, titles and controls are per channel from the identity, so a
        filterset carrying only some channels (an Ex488-only unit) shows
        exactly what the unit has. A drawer is titled by its channel's label
        (``_drawer_title``) beside a swatch of its colour, and has illumination controls where
        the unit's row drives an LED. A channel the unit lacks stops being
        acquired.
        """
        records = {record.key_name: record for record in identity.channels}
        colors = _app_ctx.ctx.session.channel_colors()
        for channel, drawer in self._drawers.items():
            record = records.get(channel)
            if record is None:
                _app_ctx.ctx.session.set_channel_acquire(channel, None)
                drawer.collapse = True
                continue
            color = colors[channel]
            drawer.color_rgba = (*color, 1.0)
            drawer.chrome_rgba = (*_lightened(color), 1.0)
            drawer.title = _drawer_title(record)
            control = drawer.ids['control']
            control.illumination_support = bool(record.led_addresses)
            # Luminescence is long and dim: its exposure sums frames, and
            # the camera's auto gain is not offered for it.
            luminescence = record.modality == channel_record.Modality.LUMINESCENCE
            control.exposure_summing_support = luminescence
            control.autogain_support = not luminescence
        self._shown_channels = frozenset(records)
        self._resort_accordion()

    def set_channel_focus_visibility(self, visible: bool) -> None:
        for channel in common_utils.get_channels():
            self.channel_control_lookup(channel=channel).focus_support = visible

    def _resort_accordion(self):
        """Put the channel accordion back in catalogue order after a model switch.

        Live scope-model transitions (LS620 -> LS850, etc.) change which
        drawers are shown; each change puts them back in order. Display order
        IS the catalogue order (a channel's id is its position there), so no
        second order is authored here.
        """
        accordion = self.ids.get('accordion_id') if hasattr(self, 'ids') else None
        if accordion is None:
            return
        resort_accordion(
            accordion,
            [
                (drawer, channel in self._shown_channels)
                for channel, drawer in self._drawers.items()
            ],
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
        for channel in common_utils.get_channels():
            self.accordion_item_lookup(channel=channel).person_opened = functools.partial(
                self._show_channel_for_person, channel
            )
        self.sync_camera_capability_ranges()
        self.enable_image_stats_if_needed()

    def enable_image_stats_if_needed(self):
        if _app_ctx.ctx.session.engineering_mode:
            for channel in common_utils.get_channels():
                channel_control = self.channel_control_lookup(channel=channel)
                channel_control.ids['image_stats_mean_id'].height = '30dp'
                channel_control.ids['image_stats_stddev_id'].height = '30dp'
                channel_control.ids['image_af_score_id'].height = '30dp'

    def set_channel_exposure_ranges(self):
        """Size each channel's exposure slider to the session's answer
        (``channel_slider_maxima``), the one owner of exp_slider.max. With no
        camera there is no maximum to show, and the camera controls are
        disabled (set_camera_controls_support), so the slider keeps its own.
        """
        session = _app_ctx.ctx.session
        for channel in common_utils.get_fluorescence_channels():
            channel_control = self.channel_control_lookup(channel=channel)
            channel_control.ids[
                'exp_slider'
            ].min = 1.0  # 1ms floor -- sub-ms never realistic for fluorescence
            channel_control.ids['exp_slider'].step = 1.0  # Integer steps only

        for channel in common_utils.get_luminescence_channels():
            channel_control = self.channel_control_lookup(channel=channel)
            channel_control.ids['exp_slider'].min = 1.0  # 1ms floor
            channel_control.ids['exp_slider'].step = 1.0  # Integer steps only

        for channel in common_utils.get_channels():
            maximum = session.channel_slider_maxima(channel).slider_max_exposure_ms
            if maximum is not None:
                self.channel_control_lookup(channel=channel).ids['exp_slider'].max = maximum

    def set_channel_gain_ranges(self):
        """Size each channel's gain slider to the connected camera's cap.

        Parallel to set_channel_exposure_ranges. Pre-fix the gain slider
        was hardcoded 0-48 dB in the kv regardless of camera -- that
        let LS620 users drag past the usable range and black out the
        image. `ctx.max_gain` is populated from
        capabilities.camera_limits in load_settings (and the same
        default-fallback pattern as exposure).
        """
        ctx = _app_ctx.ctx
        for channel in common_utils.get_channels():
            channel_control = self.channel_control_lookup(channel=channel)
            channel_control.ids['gain_slider'].max = ctx.max_gain

    def set_channel_illumination_ranges(self):
        """Size each channel's illumination slider to the session's answer
        (``channel_slider_maxima``), the one owner of ill_slider.max; the .kv
        value is the placeholder until this runs. A scope with no LED board
        has no maximum to show, and its illumination controls are hidden
        (set_channel_led_controller_support), so the slider keeps its own.
        """
        session = _app_ctx.ctx.session
        for channel in common_utils.get_channels():
            maximum = session.channel_slider_maxima(channel).slider_max_illumination_ma
            if maximum is not None:
                self.channel_control_lookup(channel=channel).ids['ill_slider'].max = maximum

    def set_channel_autogain_support(self):
        """Gate the Auto Gain/Exp control on the camera's hardware auto-gain.

        The checkbox drives hardware auto-gain (camera.set_auto_gain ->
        driver.auto_gain), which CameraAPI refuses on a camera without it
        (IDS U3-34Lx, FX2 LS620), so there the control is hidden. Auto-gain
        and auto-exposure come together on every shipped camera. Parallel to
        set_channel_gain_ranges: runs at every capability-sync point (connect /
        scope-change). The per-channel static rule (Lumi has no autogain) is
        preserved -- this only drives the orthogonal camera_autogain_support
        gate, AND-ed with autogain_support in the kv. The answer is the
        scope's capability, camera_supports_auto_gain.

        Only the visibility gate is set here; the persisted per-channel auto_gain
        is NOT mutated. The enable actually in force is the API's answer
        (camera.applied_auto_gain_for), read at the consumption point
        (ChannelControl.effective_auto_gain), so a capable camera's saved
        preference survives a swap to an AG-less body and back.
        """
        supported = _app_ctx.ctx.scope.capabilities.camera_supports_auto_gain
        for channel in common_utils.get_channels():
            channel_control = self.channel_control_lookup(channel=channel)
            channel_control.camera_autogain_support = supported

    def set_channel_led_controller_support(self):
        """Hide every channel's LED toggle and illumination current when the
        scope came up without its LED board: its capabilities carry no LED
        channels. Display of the API's answer, set where the capabilities
        are synced.
        """
        present = _app_ctx.ctx.scope.capabilities.led_addresses is not None
        for channel in common_utils.get_channels():
            self.channel_control_lookup(channel=channel).led_controller_support = present

    def set_camera_controls_support(self):
        """Disable the camera controls -- each channel's gain, exposure and
        auto-gain, and the frame size and binning -- when the scope has no
        camera connected. Display of the API's answer, set where the
        capabilities are synced.
        """
        ctx = _app_ctx.ctx
        connected = ctx.scope.camera_connected
        for channel in common_utils.get_channels():
            self.channel_control_lookup(channel=channel).camera_connected = connected
        ctx.motion_settings.ids['microscope_settings_id'].camera_connected = connected

    def sync_camera_capability_ranges(self):
        """Resync every per-channel camera control from the live camera caps.

        The single grouping of the per-channel capability setters, so the connect
        path (_init_ui) and the reconnect path apply the SAME set and can't
        drift to a subset (reconnect previously refreshed only exposure ranges).
        Callers that change the camera (reconnect) must refresh ctx.max_gain
        first -- the gain setter reads that cap.

        The setters run with every channel marked initializing. Narrowing a
        slider's max makes Kivy clamp its value, which fires on_value into the
        channel's handler; by the time this runs, load_settings has already
        cleared the flag, so without this the app's own bound-application would
        be recorded as a user drag and would overwrite the stored value with
        the bound. Every other programmatic widget write in ChannelControl takes
        the same flag for the same reason.

        reconcile_channels_to_camera_caps stays OUTSIDE the flag: it delivers to
        the camera through each channel's apply, and that apply returns early on
        exactly this flag -- inside, it would render and push nothing.
        """
        channel_controls = [
            self.channel_control_lookup(channel=channel) for channel in common_utils.get_channels()
        ]
        for channel_control in channel_controls:
            channel_control._initializing = True
        try:
            self.set_channel_exposure_ranges()
            self.set_channel_gain_ranges()
            self.set_channel_led_controller_support()
            self.set_camera_controls_support()
            self.set_channel_illumination_ranges()
            self.set_channel_autogain_support()
        finally:
            for channel_control in channel_controls:
                channel_control._initializing = False
        shown = channel_controls[0]
        ctx = _app_ctx.ctx
        limits = ctx.scope.capabilities.camera_limits
        exposure_max = None if limits is None else limits.exposure_max_ms
        gui_logger.display(
            'CAMERA_CONTROLS',
            f'camera={shown.camera_connected} led_board={shown.led_controller_support} '
            f'autogain={shown.camera_autogain_support} '
            f'gain_max={ctx.max_gain} exposure_max={exposure_max}',
        )
        self.reconcile_channels_to_camera_caps()

    def reconcile_channels_to_camera_caps(self):
        """Render every channel the attached camera cannot fully reach, and re-apply the open one.

        A camera swap can leave a channel's persisted gain_db/exposure_ms above
        the new body's maximum. The stored value is the user's committed
        intent, so it is NOT rewritten: the API caps what it writes to
        hardware, the channel runs at the most that body can do, and putting a
        capable camera back applies the intent again. Overwriting the store
        instead destroyed the setting silently -- the periodic current.json
        flush persisted the shrunken value and nothing recorded the original.

        The divergence itself is the trigger. Nothing is applied here: the
        Session caps each value as it puts a channel on the camera, and bring-up
        has already put the shown channel there; another channel reaches the
        camera, capped, when it is next shown.

        Runs BEFORE anything renders the store, on every path that reaches it,
        so the slider's pinned position and the box's stored value are in
        agreement the first time they are drawn.

        The re-render is explicit. It used to arrive as a side effect of
        writing the slider -- the channel's handler re-committed the value
        (crediting the user with a drag it never made).
        """
        ctx = _app_ctx.ctx
        settings = ctx.settings
        camera = ctx.lumaview.scope.camera
        for channel in common_utils.get_channels():
            gain = camera.applied_gain_db_for(settings[channel]['gain_db'])
            exposure = camera.applied_exposure_ms_for(settings[channel]['exposure_ms'])
            if gain.capped or exposure.capped:
                self.channel_control_lookup(channel=channel).render_channel_values_from_settings()

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

            for channel in common_utils.get_channels():
                channel_control = ctx.image_settings.channel_control_lookup(channel=channel)
                Clock.unschedule(channel_control.ids['histo_id'].histogram)
                logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')
        else:
            self.x = lumaview.width - self.settings_width

    def update_transmitted(self):
        for channel in common_utils.get_transmitted_channels():
            channel_control = self.channel_control_lookup(channel=channel)

            # Remove 'Colorize' option in transmitted channels control
            # -----------------------------------------------------
            # Remove CBT from transmitted channel control
            channel_control.show_cbt = False
            label = channel_control.ids['composite_threshold_label']
            slider = channel_control.ids['composite_threshold_slider']
            text = channel_control.ids['composite_threshold_text']
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
            channel_control.ids['false_color_label'].text = ''
            channel_control.ids['false_color'].color = (0.0,) * 4

            channel_control.ids['ill_slider'].step = 1

    def check_settings(self, *args):
        logger.info('[LVP Main  ] ImageSettings.check_settings()')
        lumaview = _app_ctx.ctx.lumaview
        if self.ids['toggle_imagesettings'].state == 'normal':
            self.x = lumaview.width - self.tab_width
        else:
            self.x = lumaview.width - self.settings_width


def set_histogram_channel(active_channel):
    for channel in common_utils.get_channels():
        channel_control = _app_ctx.ctx.image_settings.channel_control_lookup(channel=channel)
        Clock.unschedule(channel_control.ids['histo_id'].histogram)

        if channel == active_channel:
            Clock.schedule_interval(channel_control.ids['histo_id'].histogram, 0.5)
            logger.debug(
                f'[LVP Main  ] Clock.schedule_interval(...[{active_channel}]...histogram, 0.5)'
            )
