# Copyright Etaluma, Inc.
import copy
import logging

import numpy as np

from kivy.clock import Clock
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout

import _lumascope.modules.app_context as _app_ctx
import _lumascope.modules.common_utils as common_utils
import _lumascope.modules.channel_record as channel_record
import _lumascope.modules.image_mode as image_mode
from _lumascope.modules import gui_logger
from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.protocol import CHANNEL_SETTING_KEYS
from ui.ui_helpers import run_reported, show_on_slider, submit_reported, typed_number

logger = logging.getLogger('LVP.ui.channel_control')

# The channel-settings field each stored key is written as.
_FIELD_OF_KEY = {key: field for field, key in CHANNEL_SETTING_KEYS}

# Every channel's slider maxima, brightfield's included, are the session's
# answer (ScopeSession.channel_slider_maxima), set in image_settings.
INIT_MAX_RETRIES = 50

# The three settings a channel shows twice -- once on a slider, once in a text
# box -- paired with the widgets that show them. Illumination is absent from
# the channels with no LED, so a renderer skips a key the channel does not carry.
_CHANNEL_VALUE_WIDGETS = (
    ('ill_slider', 'ill_text', 'illumination_ma'),
    ('gain_slider', 'gain_text', 'gain_db'),
    ('exp_slider', 'exp_text', 'exposure_ms'),
)

# ------------------------------------------------------------------
# Diagnostic toggle for the "illumination slider > ~150 mA silently
# fails to light LED on LS620 FX2" bench investigation
# (2026-04-16). Logs type + value at the slider vs text entry points
# so the bench trace can show whether the two code paths diverge
# here (int vs float) or further downstream. Companion gates live
# in src/_lumascope/drivers/fx2driver.py (byte-level wire trace) and
# src/_lumascope/modules/lumascope_api/illumination.py (cache-equality check).
# Toggle by either:
#   * set fx2_debug_wire_enabled: true in the settings
#   * flip _FX2_DEBUG_WIRE = True  below
# ------------------------------------------------------------------
_FX2_DEBUG_WIRE = False


def _fx2_wire_debug_enabled(settings: dict) -> bool:
    return _FX2_DEBUG_WIRE or settings['fx2_debug_wire_enabled']


class ChannelControl(BoxLayout):
    channel = StringProperty(None)
    bg_color = ObjectProperty(None)
    illumination_support = BooleanProperty(True)
    stimulation_support = BooleanProperty(False)
    show_stim_controls = BooleanProperty(False)
    autogain_support = BooleanProperty(True)
    # Orthogonal runtime gate (like show_camera_controls): the Auto Gain/Exp
    # checkbox drives the camera's hardware auto-gain, so it is hidden on a
    # camera whose profile reports no hardware AG (IDS U3-34Lx, FX2 LS620).
    # Set from scope.capabilities.camera_supports_auto_gain on connect /
    # scope-change; AND-ed with the per-channel static autogain_support in the kv.
    camera_autogain_support = BooleanProperty(True)
    exposure_summing_support = BooleanProperty(False)
    # Hides the focus and autofocus rows on a scope with no Z axis; set from
    # scope.capabilities.has_focus when the scope's features are applied.
    focus_support = BooleanProperty(True)
    # Orthogonal runtime gate (like camera_autogain_support): hides the LED
    # toggle and the illumination current on a scope that came up without
    # its LED board; set from scope.capabilities when the camera's
    # capabilities are synced. AND-ed with the static illumination_support.
    led_controller_support = BooleanProperty(True)
    # Disables the gain, exposure and auto-gain controls on a scope with no
    # camera connected; set from scope.camera_connected when the camera's
    # capabilities are synced.
    camera_connected = BooleanProperty(True)
    show_camera_controls = BooleanProperty(True)
    # Drives the 8-bit summing depth-loss hint row; the row height follows the
    # label's wrapped texture so the multi-line warning is not clipped.
    sum_depth_hint_active = BooleanProperty(False)
    # The False Color box and the CBT row: shown on a channel with a colour to
    # tint or composite, hidden on a transmitted one; set from the channel's
    # record when the drawers are drawn.
    false_color_support = BooleanProperty(True)
    show_cbt = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logger.debug('[LVP Main  ] ChannelControl.__init__()')
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)

        # Flag to prevent apply_settings during initialization
        self._initializing = True

        self._init_ui_retries = 0
        Clock.schedule_once(self._init_ui, 0)
        # Defer the depth-hint binding: scope_display (the image_mode owner) is
        # built later in the app startup, so bind on the next frame.
        Clock.schedule_once(self._bind_depth_hint, 0)

    def _bind_depth_hint(self, *args):
        """Observe image_mode changes so the summing depth-loss hint stays
        current when the user switches mode in the other settings panel.

        scope_display (the image_mode owner) is built later in app startup, so
        the first frame can fire before it exists; retry rather than drop the
        binding for the session, and guard against binding twice across retries.
        """
        if getattr(self, '_depth_hint_bound', False):
            return
        scope_display = getattr(_app_ctx.ctx, 'scope_display', None)
        if scope_display is None:
            self._depth_hint_bind_retries = getattr(self, '_depth_hint_bind_retries', 0) + 1
            if self._depth_hint_bind_retries <= INIT_MAX_RETRIES:
                Clock.schedule_once(self._bind_depth_hint, 0.1)
            return
        scope_display.bind(image_mode=lambda *a: self._refresh_sum_depth_hint())
        self._depth_hint_bound = True
        self._refresh_sum_depth_hint()

    def _refresh_sum_depth_hint(self):
        """Show the depth-loss hint below the Sum control only when this channel
        sums in an 8-bit mode (the summed range is truncated on save).
        """
        if 'sum_depth_hint_row' not in self.ids:
            return
        scope_display = getattr(_app_ctx.ctx, 'scope_display', None)
        if scope_display is None:
            return
        sum_count = _app_ctx.ctx.settings[self.channel]['sum']
        self.sum_depth_hint_active = image_mode.depth_truncation_warning_active(
            sum_count, scope_display.image_mode
        )
        gui_logger.display(f'SUM_DEPTH_HINT_{self.channel}', self.sum_depth_hint_active)

    def _validate_and_apply_text_input(
        self,
        text_id: str,
        slider_id: str,
        settings_key: str,
        cast=float,
        settings_path: str | None = None,
        lane=None,
    ) -> bool:
        """Hand a typed channel value to the settings writer; True when the store changed.

        Parses the text and writes it through ``set_channel_settings``. The Session
        decides what is stored: it refuses a value outside the setting's range
        (an illumination past the LED board's maximum among them), telling
        the person, and stores a gain or exposure past the camera's limit as
        the intent, which the apply drives at the limit. The box and its
        slider then show what is stored, whatever the call did; the slider's
        own range is display and never narrows what was typed.

        Args:
            text_id: Kivy widget id for the text input (e.g., 'gain_text')
            slider_id: Kivy widget id for the slider (e.g., 'gain_slider')
            settings_key: Key in settings[self.channel] (e.g., 'gain_db')
            cast: Type to cast the text value (float or int)
            settings_path: Dot-separated sub-path for nested settings
                          (e.g., 'video_config.duration' or 'stim_config.frequency')
            lane: For a setting the Session puts on the hardware (gain and
                exposure on the camera lane, illumination on the IO lane),
                that lane: the write runs there and the box shows the store
                once it has, and this answers False, the store not yet read.
                None for a setting that is stored only, written here.
        """
        settings = _app_ctx.ctx.settings

        # The log name is derived from the widget id rather than passed in:
        # every text box here is '<name>_text' and its slider twin already logs
        # '<NAME>_<channel>', so the id IS the name and a separate parameter can
        # only drift from it or go unpassed. A non-conforming id is a caller
        # bug and says so rather than logging under a wrong name. Checked here,
        # before the settings write, so a bad caller cannot half-apply.
        if not text_id.endswith('_text'):
            raise ValueError(
                f"text_id {text_id!r} must end in '_text' -- the log name is derived from it"
            )
        record_name = f'{text_id.removesuffix("_text").upper()}_{self.channel}'

        # Captured before any path below rewrites the widget, so the record can
        # carry what the user actually typed rather than what we made of it.
        typed_text = self.ids[text_id].text

        field = _FIELD_OF_KEY[settings_path or settings_key]

        def stored_value():
            val = settings[self.channel]
            for p in (settings_path or settings_key).split('.'):
                val = val[p]
            return val

        def show_stored():
            # The display half, read back from the store: what the writer
            # kept, or left in place when it refused, is what the box and
            # slider show.
            self._show_value_on_widgets(slider_id, text_id, stored_value(), cast=cast)

        val = stored_value()
        raw = typed_number(typed_text, cast, show_stored)
        if raw is None:
            # An unparseable entry is still a user action, and the reset above
            # would otherwise leave no trace of it. Both halves are recorded:
            # what was typed, and what the box was put back to. Separate names
            # because both lines have to survive -- the pair is what says the
            # entry was refused rather than accepted.
            gui_logger.text_input(record_name, typed_text)
            gui_logger.text_input(f'{record_name}_APPLIED', val)
            return False

        # An entry equal to the stored value changes nothing: the box arrives
        # here on an edit or an Enter (`on_commit`), and an Enter on the shown
        # value, or a retyped one, is that case. No change, no commit; the box
        # already shows the stored value.
        if raw == val:
            return False

        if lane is not None:
            gui_logger.text_input(record_name, typed_text)

            def shown_once_written():
                show_stored()
                if raw != stored_value():
                    gui_logger.text_input(f'{record_name}_APPLIED', stored_value())

            session = _app_ctx.ctx.session
            submit_reported(
                lambda: session.set_channel_settings(self.channel, **{field: raw}),
                shown_once_written,
                record_name,
                lane=lane,
            )
            return False

        run_reported(
            lambda: _app_ctx.ctx.session.set_channel_settings(self.channel, **{field: raw}),
            show_stored,
            record_name,
        )
        stored = stored_value()

        # text_input (not slider): this is a typed commit, and the twin slider
        # emits SLIDER for the same setting, so sharing the verb would make a
        # drag and a keystroke indistinguishable in the bundle.
        gui_logger.text_input(record_name, typed_text)

        # Only when the writer refused and left the stored value in place.
        # The comparison is on the PARSED number, not the strings: '5' typed
        # into a float box becomes 5.0, which is the same value and must not
        # look like a correction.
        if raw != stored:
            gui_logger.text_input(f'{record_name}_APPLIED', stored)

        return stored != val

    def _init_ui(self, dt=0):
        ctx = _app_ctx.ctx
        if ctx is None:
            self._init_ui_retries += 1
            if self._init_ui_retries > INIT_MAX_RETRIES:
                logger.error(
                    '[LVP Main  ] ChannelControl._init_ui: ctx still None after 50 retries, giving up'
                )
                return
            Clock.schedule_once(self._init_ui, 0.1)
            return
        settings = ctx.settings

        if (
            self.channel in common_utils.get_fluorescence_channels()
            and settings['stimulation_enabled']
            and ctx.scope.capabilities.has_firmware_stim
        ):
            self.stimulation_support = True
            self.show_stim_controls = True
        else:
            self.stimulation_support = False
            self.show_stim_controls = False

        self.update_stim_controls_visibility()

        # Don't apply settings during initial UI setup - will be done after load_settings
        # Skip initialization of autogain and apply_settings here

    def update_stim_controls_visibility(self):
        if self.ids['stim_enable_btn'].active:
            self.show_stim_controls = True
            self.show_camera_controls = False
            self.hide_camera_controls()
        else:
            self.show_stim_controls = False
            self.show_camera_controls = True

    def hide_camera_controls(self):
        """Display only: the acquire setting itself is written by the user's
        action (enabling stim clears it), and this also runs from the
        settings-to-widgets sync, which must not write settings."""
        self.show_camera_controls = False
        self.ids['acquire_none'].active = True

    def _write_on_lane(self, key: str, value, name: str, lane) -> None:
        """Write this channel's ``key`` on the lane its hardware is on.

        The Session puts a write to the shown channel's gain, exposure or
        illumination on the hardware before it returns, so a drag is live,
        each tick in order on that one lane. A stored write reaches the
        widgets through the store and the listeners; a refused one stores
        nothing and tells no listener, so the setting's own slider and box
        are drawn from the store once the write has answered, whatever it
        answered, and a refused drag shows the setting it left in place.
        The other settings' widgets are left to their own writes.
        """
        ctx = _app_ctx.ctx
        session = ctx.session
        (slider_id, text_id) = next((s, t) for s, t, k in _CHANNEL_VALUE_WIDGETS if k == key)

        def show_stored():
            with ctx.settings_lock:
                stored = ctx.settings[self.channel][key]
            self._show_value_on_widgets(slider_id, text_id, stored)

        submit_reported(
            lambda: session.set_channel_settings(self.channel, **{_FIELD_OF_KEY[key]: value}),
            show_stored,
            f'{name}_{self.channel}',
            lane=lane,
        )

    def ill_slider(self):
        settings = _app_ctx.ctx.settings
        # Early return on programmatic updates (#617): when another code
        # path sets ill_slider.value directly (load_settings, ill_text,
        # set_step_state), on_value fires and re-enters
        # here. Without this guard, the handler overwrites the caller's
        # settings write and schedules a redundant apply_settings. Callers
        # are responsible for writing settings explicitly when they use
        # _initializing=True.
        if self._initializing:
            return
        logger.info('[LVP Main  ] ChannelControl.ill_slider()')
        illumination = round(self.ids['ill_slider'].value)  # Round to integer (step=1)
        # Slider-vs-text divergence trace for the > ~150 mA silent-
        # fail bench investigation. See _FX2_DEBUG_WIRE block at top
        # of this file. INFO level -- this is a key divergence point
        # (int from slider vs float from text).
        if _fx2_wire_debug_enabled(settings):
            logger.info(
                '[FX2 LED diag] ill_slider ENTRY channel=%s raw_value=%r '
                'raw_type=%s -> illumination=%r type=%s source=slider',
                self.channel,
                self.ids['ill_slider'].value,
                type(self.ids['ill_slider'].value).__name__,
                illumination,
                type(illumination).__name__,
            )
        gui_logger.slider(f'ILLUMINATION_{self.channel}', illumination)
        self._write_on_lane(
            'illumination_ma', illumination, 'ILLUMINATION', _app_ctx.ctx.io_executor
        )

        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(illumination)
        if self.ids['ill_text'].text != new_text:
            self.ids['ill_text'].text = new_text

    def ill_text(self) -> None:
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] ChannelControl.ill_text()')
        typed_text = self.ids['ill_text'].text
        # Text-entry divergence trace for the > ~150 mA silent-fail
        # bench investigation. See _FX2_DEBUG_WIRE block at top of
        # this file. INFO level -- this is the other key divergence
        # point (float from text vs int from slider).
        if _fx2_wire_debug_enabled(settings):
            logger.info(
                '[FX2 LED diag] ill_text ENTRY channel=%s raw_text=%r source=text',
                self.channel,
                typed_text,
            )
        self._validate_and_apply_text_input(
            'ill_text',
            'ill_slider',
            'illumination_ma',
            lane=_app_ctx.ctx.io_executor,
        )

    def sum_slider(self):
        logger.info('[LVP Main  ] ChannelControl.sum_slider()')
        total = int(self.ids['sum_slider'].value)
        gui_logger.slider(f'SUM_{self.channel}', total)
        _app_ctx.ctx.session.set_channel_settings(self.channel, frames_summed=total)
        self._refresh_sum_depth_hint()

    def sum_text(self):
        logger.info('[LVP Main  ] ChannelControl.sum_text()')
        if self._validate_and_apply_text_input('sum_text', 'sum_slider', 'sum', cast=int):
            self._refresh_sum_depth_hint()

    def video_duration_slider(self):
        logger.info('[LVP Main  ] ChannelControl.video_duration_slider()')
        duration = self.ids['video_duration_slider'].value
        gui_logger.slider(f'VIDEO_DURATION_{self.channel}', duration)
        _app_ctx.ctx.session.set_channel_settings(self.channel, video_duration_s=duration)

    def video_duration_text(self):
        logger.info('[LVP Main  ] ChannelControl.video_duration_text()')
        self._validate_and_apply_text_input(
            'video_duration_text',
            'video_duration_slider',
            'duration',
            cast=int,
            settings_path='video_config.duration',
        )

    def update_auto_gain(self):
        logger.info('[LVP Main  ] ChannelControl.update_auto_gain()')
        enabled = self.ids['auto_gain'].state == 'down'
        gui_logger.toggle(f'AUTO_GAIN_{self.channel}', enabled)

        # The Session arms the camera, or locks its arm and stores what it
        # reached, so a script turning auto-gain on or off does the same. It
        # waits on the camera, so it runs on the camera lane. The redraw shows
        # whatever the store holds afterwards -- the reached values, or the
        # old ones if the lock was refused -- and re-syncs the box.
        ctx = _app_ctx.ctx
        channel = self.channel

        def redraw():
            self.render_channel_values_from_settings()
            self.show_as_shown()

        submit_reported(
            lambda: ctx.session.set_channel_settings(channel, auto_gain=enabled),
            redraw,
            f'AUTO_GAIN_{channel}',
            lane=ctx.camera_executor,
        )

    def gain_slider(self):
        # See ill_slider -- programmatic updates must not re-enter (#617).
        if self._initializing:
            return
        logger.info('[LVP Main  ] ChannelControl.gain_slider()')
        gain = round(self.ids['gain_slider'].value, 1)  # Round to 1 decimal (step=0.1)
        gui_logger.slider(f'GAIN_{self.channel}', gain)
        self._write_on_lane('gain_db', gain, 'GAIN', _app_ctx.ctx.camera_executor)
        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(gain)
        if self.ids['gain_text'].text != new_text:
            self.ids['gain_text'].text = new_text

    def gain_text(self):
        logger.info('[LVP Main  ] ChannelControl.gain_text()')
        self._validate_and_apply_text_input(
            'gain_text', 'gain_slider', 'gain_db', lane=_app_ctx.ctx.camera_executor
        )

    def composite_threshold_slider(self):
        logger.info('[LVP Main  ] ChannelControl.composite_threshold_slider()')
        composite_threshold = self.ids['composite_threshold_slider'].value
        gui_logger.slider(f'COMPOSITE_THRESHOLD_{self.channel}', composite_threshold)
        _app_ctx.ctx.session.set_channel_settings(
            self.channel, composite_brightness_threshold_pct=composite_threshold
        )

    def composite_threshold_text(self):
        logger.info('[LVP Main  ] ChannelControl.composite_threshold_text()')
        self._validate_and_apply_text_input(
            'composite_threshold_text',
            'composite_threshold_slider',
            'composite_brightness_threshold',
        )

    def exp_slider(self):
        # See ill_slider -- programmatic updates must not re-enter (#617).
        if self._initializing:
            return
        logger.info('[LVP Main  ] ChannelControl.exp_slider()')
        exposure = round(self.ids['exp_slider'].value, 2)  # Round to 2 decimals (step=0.01)
        gui_logger.slider(f'EXPOSURE_{self.channel}', exposure)
        self._write_on_lane('exposure_ms', exposure, 'EXPOSURE', _app_ctx.ctx.camera_executor)
        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(exposure)
        if self.ids['exp_text'].text != new_text:
            self.ids['exp_text'].text = new_text

    def exp_text(self) -> None:
        logger.info('[LVP Main  ] ChannelControl.exp_text()')
        # The typed exposure is the channel's intent, stored as typed. The
        # sensor's ceiling is applied where the value reaches the camera
        # (CameraAPI.applied_exposure_ms_for, as the Session applies it), with
        # the intent kept, exactly as a value loaded from the file is; the
        # box no longer carries a second copy of that ceiling.
        self._validate_and_apply_text_input(
            'exp_text', 'exp_slider', 'exposure_ms', lane=_app_ctx.ctx.camera_executor
        )

    def stim_freq_slider(self):
        logger.info('[LVP Main  ] ChannelControl.stim_freq_slider()')
        frequency = self.ids['stim_freq_slider'].value
        gui_logger.slider(f'STIM_FREQ_{self.channel}', frequency)
        _app_ctx.ctx.session.set_channel_settings(self.channel, stim_frequency_hz=frequency)

    def stim_pulse_count_slider(self):
        logger.info('[LVP Main  ] ChannelControl.stim_pulse_count_slider()')
        pulse_count = int(self.ids['stim_pulse_count_slider'].value)
        gui_logger.slider(f'STIM_PULSE_COUNT_{self.channel}', pulse_count)
        _app_ctx.ctx.session.set_channel_settings(self.channel, stim_pulse_count=pulse_count)

    def stim_pulse_width_slider(self):
        logger.info('[LVP Main  ] ChannelControl.stim_pulse_width_slider()')
        pulse_width = int(self.ids['stim_pulse_width_slider'].value)
        gui_logger.slider(f'STIM_PULSE_WIDTH_{self.channel}', pulse_width)
        _app_ctx.ctx.session.set_channel_settings(self.channel, stim_pulse_width_ms=pulse_width)

    def stim_freq_text(self):
        logger.info('[LVP Main  ] ChannelControl.stim_freq_text()')
        self._validate_and_apply_text_input(
            'stim_freq_text',
            'stim_freq_slider',
            'frequency',
            settings_path='stim_config.frequency',
        )

    def stim_pulse_count_text(self):
        logger.info('[LVP Main  ] ChannelControl.stim_pulse_count_text()')
        self._validate_and_apply_text_input(
            'stim_pulse_count_text',
            'stim_pulse_count_slider',
            'pulse_count',
            cast=int,
            settings_path='stim_config.pulse_count',
        )

    def stim_pulse_width_text(self):
        logger.info('[LVP Main  ] ChannelControl.stim_pulse_width_text()')
        self._validate_and_apply_text_input(
            'stim_pulse_width_text',
            'stim_pulse_width_slider',
            'pulse_width',
            cast=int,
            settings_path='stim_config.pulse_width',
        )

    def stim_ill_slider(self):
        logger.info('[LVP Main  ] ChannelControl.stim_ill_slider()')
        illumination = round(self.ids['stim_ill_slider'].value)
        gui_logger.slider(f'STIM_ILL_{self.channel}', illumination)
        _app_ctx.ctx.session.set_channel_settings(self.channel, stim_illumination_ma=illumination)
        new_text = str(illumination)
        if self.ids['stim_ill_text'].text != new_text:
            self.ids['stim_ill_text'].text = new_text

    def stim_ill_text(self):
        logger.info('[LVP Main  ] ChannelControl.stim_ill_text()')
        self._validate_and_apply_text_input(
            'stim_ill_text',
            'stim_ill_slider',
            'illumination_ma',
            cast=int,
            settings_path='stim_config.illumination_ma',
        )

    def false_color(self):
        logger.info('[LVP Main  ] ChannelControl.false_color()')
        enabled = bool(self.ids['false_color'].active)
        gui_logger.toggle(f'FALSE_COLOR_{self.channel}', enabled)
        _app_ctx.ctx.session.set_channel_settings(self.channel, false_color=enabled)
        self.show_as_shown()

    def log_histogram_scale(self) -> None:
        """Record the histogram log-scale checkbox.

        Display-only: the histogram reads this box directly when it redraws,
        so there is no setting to write and nothing to record but the gesture.
        """
        gui_logger.toggle(f'LOG_HISTOGRAM_{self.channel}', bool(self.ids['logHistogram_id'].active))

    def update_acquire(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] ChannelControl.update_acquire()')

        if self.ids['acquire_image'].active:
            mode = AcquireMode.IMAGE
        elif self.ids['acquire_video'].active:
            mode = AcquireMode.VIDEO
        else:
            mode = 'none'
        gui_logger.select(f'ACQUIRE_{self.channel}', mode)

        # The Session stops the channel stimulating when it acquires; the
        # stimulation switch is drawn off to match.
        _app_ctx.ctx.session.set_channel_settings(
            self.channel, acquire=None if mode == 'none' else mode
        )
        if mode != 'none':
            self.ids['stim_disable_btn'].active = True
            self.show_stim_controls = False

        if 'stim_config' in settings[self.channel]:
            self.update_stim_controls_visibility()

    def update_stim_enable(self):
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] ChannelControl.update_stim_enable()')
        enabled = self.ids['stim_enable_btn'].active
        gui_logger.toggle(f'STIM_{self.channel}', enabled)
        # One write: a channel that stimulates captures nothing.
        if enabled:
            ctx.session.set_channel_settings(self.channel, acquire=None, stim_enabled=True)
            self.ids['acquire_none'].active = True
            self.ids['acquire_none'].state = 'down'
        else:
            ctx.session.set_channel_settings(self.channel, stim_enabled=False)

        self.update_stim_controls_visibility()

    def update_autofocus(self):
        logger.info('[LVP Main  ] ChannelControl.update_autofocus()')
        enabled = bool(self.ids['autofocus'].active)
        gui_logger.toggle(f'AUTOFOCUS_ENABLED_{self.channel}', enabled)
        _app_ctx.ctx.session.set_channel_settings(self.channel, autofocus=enabled)

    def save_focus(self):
        gui_logger.button(f'SAVE_FOCUS_{self.channel}')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] ChannelControl.save_focus()')
        # The selected step and the protocol are taken at the click: the
        # step the user was looking at, in the protocol the panel holds now
        # (the panel replaces its protocol on New and Load). The panel's -1
        # is no step selected.
        protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
        protocol = protocol_settings._protocol
        selected_step = int(protocol_settings.curr_step)
        step_index = selected_step if selected_step >= 0 else None
        run_reported(
            lambda: ctx.session.save_focus(protocol, self.channel, step_index=step_index),
            lambda: self._refresh_step_views(ctx, protocol),
            f'SAVE_FOCUS_{self.channel}',
        )

    def _refresh_step_views(self, ctx, protocol):
        """Redraw the stage view and the step editor from *protocol*.

        Shared by every focus action that changes step Z values so the
        labware view and the step editor's focus readout update together.
        It is the redraw of run_reported, which runs it on this thread and
        reports whatever it raises.
        """
        ctx.stage.set_protocol_steps(protocol)
        ctx.motion_settings.ids['protocol_settings_id'].update_step_ui()

    def apply_focus_to_channel_steps(self):
        gui_logger.button(f'APPLY_FOCUS_TO_STEPS_{self.channel}')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] ChannelControl.apply_focus_to_channel_steps()')
        # As Save Focus: the protocol the panel holds at the click.
        protocol = ctx.motion_settings.ids['protocol_settings_id']._protocol
        run_reported(
            lambda: ctx.session.apply_focus_to_channel_steps(protocol, self.channel),
            lambda: self._refresh_step_views(ctx, protocol),
            f'APPLY_FOCUS_TO_STEPS_{self.channel}',
        )

    def goto_focus(self):
        from ui.ui_helpers import move_z

        gui_logger.button(f'GOTO_FOCUS_{self.channel}')
        logger.info('[LVP Main  ] ChannelControl.goto_focus()')
        run_reported(
            lambda: move_z(_app_ctx.ctx.session.saved_focus_z_um(self.channel)),
            None,
            f'GOTO_FOCUS_{self.channel}',
        )

    def led_toggle(self):
        """The Enable LED button's press: record it, then light or darken the shown channel.

        The button sits in the shown channel's drawer, so it is that
        channel's. Writing the button's state is not a press: it never
        dispatches on_release. Its one writer is the listener bridge, drawing
        the LED, so a refused press goes back.
        """
        lit = self.ids['enable_led_btn'].state == 'down'
        gui_logger.toggle(f'LED_{self.channel}', lit)
        session = _app_ctx.ctx.session
        submit_reported(
            lambda: session.light_shown_channel(lit),
            _app_ctx.ctx.ui_listener_bridge.reconcile_led_buttons,
            f'LED_{self.channel}',
        )

    def set_step_state(self, step: dict):
        """Display a protocol step in this channel's widgets. Writes NO settings.

        A person going to a step sees it here; the navigation writes the
        settings itself before applying them. A run's steps do not come here:
        a run moves only the step pointer. What keeps this a pure display write
        is the kv wiring: of the widgets set here only the illumination /
        gain / exposure sliders bind ``on_value``, and each of those
        handlers returns while ``_initializing`` is set; the other sliders
        bind ``on_release`` (a touch-up or the wheel, never a programmatic
        value) and a CheckBox's ``on_release`` never fires from ``.active``.

        Only updates widgets for keys that are present in *step*, so a
        partial update leaves unrelated widget values alone.

        Args:
            step: Protocol step dict.  Recognized keys: 'Illumination',
                'Gain', 'Exposure', 'Sum', 'Auto_Focus', 'Auto_Gain',
                'False_Color', 'Acquire', 'Video Config'.
        """
        gui_logger.display(f'CHANNEL_PANEL_{self.channel}', 'step')
        self._initializing = True
        try:
            if 'Auto_Focus' in step:
                self.ids['autofocus'].active = step['Auto_Focus']
            if 'False_Color' in step:
                self.ids['false_color'].active = step['False_Color']

            if 'Illumination' in step:
                self._show_value_on_widgets('ill_slider', 'ill_text', step['Illumination'])

            if 'Gain' in step:
                self._show_value_on_widgets('gain_slider', 'gain_text', step['Gain'])

            if 'Auto_Gain' in step:
                # The box drives the gain/exposure widgets' enabled state in
                # the kv; on a camera whose Auto Gain control is hidden a
                # ticked box would grey them with nothing to un-grey them.
                # So it shows what the camera will run, not what was stored.
                self.ids['auto_gain'].active = _app_ctx.ctx.scope.camera.applied_auto_gain_for(
                    step['Auto_Gain']
                ).applied

            if 'Exposure' in step:
                self._show_value_on_widgets('exp_slider', 'exp_text', step['Exposure'])

            if 'Sum' in step:
                self.ids['sum_text'].text = str(step['Sum'])
                self.ids['sum_slider'].value = int(step['Sum'])

            # Video config
            vc = step.get('Video Config')
            if isinstance(vc, dict) and 'duration' in vc:
                self.ids['video_duration_text'].text = str(vc['duration'])
                self.ids['video_duration_slider'].value = float(vc['duration'])

            # Acquire type
            if 'Acquire' in step:
                for sel in ('acquire_video', 'acquire_image', 'acquire_none'):
                    self.ids[sel].active = False
                acquire = step['Acquire']
                if acquire == AcquireMode.VIDEO:
                    self.ids['acquire_video'].active = True
                elif acquire == AcquireMode.IMAGE:
                    self.ids['acquire_image'].active = True
                else:
                    self.ids['acquire_none'].active = True
        finally:
            self._initializing = False

    def _show_value_on_widgets(self, slider_id: str, text_id: str, value, cast=float):
        """Show one stored value on the slider and the text box that share it.

        The text box takes the value itself; the slider takes the nearest
        position it can represent. A slider's range is a convenience range that
        can be narrower than what the box accepts, so a stored value above it
        pins the slider at its maximum while the box keeps the real number.
        Only one of the two representations may lose precision, and it is never
        the one the user reads the setting from.

        Neither is written while the operator is on it -- a box being typed
        in holds the write until it is left (``CommitInput``), a slider under
        the finger is left as it is (``show_on_slider``).

        Both writes run with the channel's handlers suppressed: these are
        display, and a slider's on_value handler exists to record what the USER
        did -- unsuppressed, it commits the written value back over the stored
        one and logs it as a drag. The previous flag state is RESTORED rather
        than cleared, so a caller that is already suppressing (a channel still
        initializing, a capability sync narrowing every slider) still has its
        own guard when this returns.
        """
        slider = self.ids[slider_id]
        on_slider = cast(np.clip(value, slider.min, slider.max))
        was_initializing = self._initializing
        self._initializing = True
        try:
            show_on_slider(slider, on_slider)
            self.ids[text_id].text = str(value)
        finally:
            self._initializing = was_initializing

    def render_channel_values_from_settings(self, channel_config=None):
        """Render this channel's stored illumination, gain and exposure.

        The one store-to-widget path for the three settings that appear on both
        a slider and a text box. Callers reconcile the store against the
        camera's PHYSICAL caps before calling: a value the hardware cannot
        honor is wrong in the store, not merely too large for the slider, and
        pinning the slider would hide it instead of correcting it.

        Pass *channel_config* when the caller already holds a snapshot of this
        channel's settings. Reading the store again here would take the lock a
        second time and could see a DIFFERENT state, leaving some widgets
        rendered from one snapshot and some from another; a caller with no
        snapshot passes nothing and this reads the store once, under the lock.
        """
        if channel_config is None:
            ctx = _app_ctx.ctx
            with ctx.settings_lock:
                stored = ctx.settings[self.channel]
                channel_config = {
                    key: stored[key] for _, _, key in _CHANNEL_VALUE_WIDGETS if key in stored
                }
        for slider_id, text_id, settings_key in _CHANNEL_VALUE_WIDGETS:
            if settings_key in channel_config:
                self._show_value_on_widgets(slider_id, text_id, channel_config[settings_key])

    def sync_widgets_from_settings(self):
        """Point every widget of this channel at its stored settings.

        The one settings-to-widgets direction: startup, a change to this
        channel's stored settings (from any writer: the window's listener
        bridge), the end of a run and the end of a standalone autofocus
        (which restored the camera from the settings) all call this. Reads
        the channel's settings once under the settings lock; writes only
        widgets. A box the operator is typing in holds the write until it is
        left (``CommitInput``) and a slider under the finger is left as it
        is (``show_on_slider``): the commit that follows is redrawn in its
        turn.

        The gain and exposure sliders' ``max`` is a camera fact set by the
        caller that learns it; the value written here follows whatever
        max the slider carries.
        """
        ctx = _app_ctx.ctx
        with ctx.settings_lock:
            channel_config = copy.deepcopy(ctx.settings[self.channel])

        gui_logger.display(f'CHANNEL_PANEL_{self.channel}', 'settings')
        self._initializing = True
        try:
            if 'composite_brightness_threshold' in channel_config:
                show_on_slider(
                    self.ids['composite_threshold_slider'],
                    channel_config['composite_brightness_threshold'],
                )

            self.render_channel_values_from_settings(channel_config)

            self.ids['false_color'].active = channel_config['false_color']
            show_on_slider(self.ids['sum_slider'], channel_config.get('sum', 1))

            if channel_config['acquire'] == AcquireMode.IMAGE:
                self.ids['acquire_image'].active = True
            elif channel_config['acquire'] == AcquireMode.VIDEO:
                self.ids['acquire_video'].active = True
            else:
                self.ids['acquire_none'].active = True

            video_config = channel_config['video_config']
            self.ids['video_duration_text'].text = str(video_config['duration'])
            show_on_slider(self.ids['video_duration_slider'], video_config['duration'])

            self.ids['autofocus'].active = channel_config['autofocus']
            # The box shows the enable actually in force, not the bare
            # stored preference, the same way apply_settings shows it.
            self.ids['auto_gain'].active = self.effective_auto_gain()

            # Shipped settings carry no stim_config on the transmitted
            # channels, and a None is representable.
            stim_config = channel_config.get('stim_config')
            if stim_config:
                # Default to hidden until enabled
                self.show_stim_controls = False

                self.ids['stim_enable_btn'].active = stim_config['enabled']
                self.ids['stim_disable_btn'].active = not stim_config['enabled']
                ill = stim_config.get('illumination_ma', 100)
                self.ids['stim_ill_text'].text = str(ill)
                show_on_slider(self.ids['stim_ill_slider'], float(ill))
                self.ids['stim_freq_text'].text = str(stim_config['frequency'])
                show_on_slider(self.ids['stim_freq_slider'], float(stim_config['frequency']))
                self.ids['stim_pulse_width_text'].text = str(stim_config['pulse_width'])
                show_on_slider(
                    self.ids['stim_pulse_width_slider'], float(stim_config['pulse_width'])
                )
                self.ids['stim_pulse_count_text'].text = str(stim_config['pulse_count'])
                show_on_slider(self.ids['stim_pulse_count_slider'], int(stim_config['pulse_count']))

                # Force hide until enabled
                for box in (
                    'stim_ill_box',
                    'stim_pulse_count_box',
                    'stim_freq_box',
                    'stim_pulse_width_box',
                ):
                    self.ids[box].visible = False
                    self.ids[box].opacity = 0

                self.update_stim_controls_visibility()
        finally:
            self._initializing = False

    def effective_auto_gain(self) -> bool:
        """The auto-gain enable actually in force for this channel's camera.

        The API's answer for the saved preference (settings[channel]['auto_gain']):
        a camera without hardware auto-gain runs manual, and its Auto Gain/Exp
        control is hidden. Read, never written back, so a capable camera's
        saved preference survives a swap to a camera without it and back.
        """
        ctx = _app_ctx.ctx
        return ctx.scope.camera.applied_auto_gain_for(
            ctx.settings[self.channel]['auto_gain']
        ).applied

    def apply_settings(self):
        """Show this channel at its stored settings, then display it.

        The Session's show_channel puts the channel's LED (lit if the shown
        channel was) and its camera settings on the hardware, and refuses
        what it cannot do, such as while a run holds the scope; its answer
        reaches the drawers through the shown-channel listener.
        """
        if getattr(self, '_initializing', False):
            return

        logger.debug(f'[LVP Main  ] {self.channel}_ChannelControl.apply_settings()')

        session = _app_ctx.ctx.session
        channel = self.channel
        submit_reported(
            lambda: session.show_channel(channel),
            None,
            f'SHOW_CHANNEL_{channel}',
        )
        self.show_as_shown()

    def show_as_shown(self):
        """Display this channel as the shown channel: its histogram, its
        auto-gain box and its false-colour shader. Display only."""
        ctx = _app_ctx.ctx
        from ui.image_settings import set_histogram_channel

        set_histogram_channel(active_channel=self.channel)

        # Effective enable = the API's answer for the saved preference
        # (effective_auto_gain): on a camera without hardware AG/AE the
        # control is hidden, so a stored True must read as off here -- else
        # the gain/exposure sliders below stay disabled with no UI to clear
        # them. Non-destructive: the stored preference is left intact.
        # The .kv has no Kivy binding from settings to auto_gain.active, so
        # the box is synced here. Programmatic .active = bool fires no
        # on_release (CheckBox only binds on_release in the .kv), so this
        # does not re-enter. The gain/exposure widgets' enabled state follows
        # this box through the kv rule (`disabled: app.lock.run_lockout or
        # auto_gain.active`): an imperative .disabled write here was erased
        # whenever the run lockout cleared, because that rule re-fires on the
        # edge.
        self.ids['auto_gain'].active = self.effective_auto_gain()

        thread = getattr(ctx, 'scope_display_thread', None)
        if thread is not None and not thread.is_paused and ctx.scope_display.use_bullseye is False:
            self.update_shader(dt=0)

    def update_shader(self, dt):
        _app_ctx.ctx.viewer.update_shader(
            channel_record.shown_color(
                self.channel, self.ids['false_color'].active, _app_ctx.ctx.session.channel_colors()
            )
        )
