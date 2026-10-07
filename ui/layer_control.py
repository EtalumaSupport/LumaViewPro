# Copyright Etaluma, Inc.
import copy
import functools
import logging

import numpy as np

from kivy.clock import Clock
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.image_mode as image_mode
from modules import gui_logger
from modules.config_ui_getters import (
    firmware_stim_supported,
    get_exposure_text_max,
    get_layer_illumination_text_max,
)
from ui.ui_helpers import run_reported, submit_reported, typed_number

logger = logging.getLogger('LVP.ui.layer_control')

# Brightfield's illumination bounds come from the resolver in
# modules.config_helpers, beside the LED driver's cap; its exposure bounds
# come from the camera, through the resolvers in modules.config_ui_getters.
SLIDER_DEBOUNCE_S = 0.1
INIT_MAX_RETRIES = 50

# The three settings a layer shows twice -- once on a slider, once in a text
# box -- paired with the widgets that show them. Illumination is absent from
# the layers with no LED, so a renderer skips a key the layer does not carry.
_LAYER_VALUE_WIDGETS = (
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
# in drivers/fx2driver.py (byte-level wire trace) and
# modules/lumascope_api/illumination.py (cache-equality check).
# Toggle by either:
#   * set fx2_debug_wire_enabled: true in the settings
#   * flip _FX2_DEBUG_WIRE = True  below
# ------------------------------------------------------------------
_FX2_DEBUG_WIRE = False


def _fx2_wire_debug_enabled(settings: dict) -> bool:
    return _FX2_DEBUG_WIRE or settings['fx2_debug_wire_enabled']


class LayerControl(BoxLayout):
    layer = StringProperty(None)
    bg_color = ObjectProperty(None)
    illumination_support = BooleanProperty(True)
    stimulation_support = BooleanProperty(False)
    show_stim_controls = BooleanProperty(False)
    autogain_support = BooleanProperty(True)
    # Orthogonal runtime gate (like show_camera_controls): the Auto Gain/Exp
    # checkbox drives the camera's hardware auto-gain, so it is hidden on a
    # camera whose profile reports no hardware AG (IDS U3-34Lx, FX2 LS620).
    # Set from scope.capabilities.camera_supports_auto_gain on connect /
    # scope-change; AND-ed with the per-layer static autogain_support in the kv.
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
    show_camera_controls = BooleanProperty(True)
    # Drives the 8-bit summing depth-loss hint row; the row height follows the
    # label's wrapped texture so the multi-line warning is not clipped.
    sum_depth_hint_active = BooleanProperty(False)
    show_cbt = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logger.debug('[LVP Main  ] LayerControl.__init__()')
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)

        # Flag to prevent apply_settings during initialization
        self._initializing = True

        self.apply_gain_slider = Clock.create_trigger(
            lambda dt: self.apply_settings(), SLIDER_DEBOUNCE_S
        )
        self.apply_exp_slider = Clock.create_trigger(
            lambda dt: self.apply_settings(), SLIDER_DEBOUNCE_S
        )
        self.apply_ill_slider = Clock.create_trigger(
            lambda dt: self.apply_settings(), SLIDER_DEBOUNCE_S
        )
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
        """Show the depth-loss hint below the Sum control only when this layer
        sums in an 8-bit mode (the summed range is truncated on save).
        """
        if 'sum_depth_hint_row' not in self.ids:
            return
        scope_display = getattr(_app_ctx.ctx, 'scope_display', None)
        if scope_display is None:
            return
        sum_count = _app_ctx.ctx.settings.get(self.layer, {}).get('sum')
        self.sum_depth_hint_active = image_mode.depth_truncation_warning_active(
            sum_count, scope_display.image_mode
        )

    def _validate_and_apply_text_input(
        self,
        text_id: str,
        slider_id: str,
        settings_key: str,
        cast=float,
        settings_path: str | None = None,
        value_max: float | None = None,
    ) -> bool:
        """Shared validation for text input -> slider -> settings update.

        Parses text, clips to slider range, updates slider + text + settings,
        and applies. Returns True on success, False on invalid input.

        Args:
            text_id: Kivy widget id for the text input (e.g., 'gain_text')
            slider_id: Kivy widget id for the slider (e.g., 'gain_slider')
            settings_key: Key in settings[self.layer] (e.g., 'gain_db')
            cast: Type to cast the text value (float or int)
            settings_path: Dot-separated sub-path for nested settings
                          (e.g., 'video_config.duration' or 'stim_config.frequency')
            value_max: Optional upper bound for the typed value when it should
                       exceed the slider's own max -- the slider is a coarse
                       quick-pick (e.g. video duration up to 60s) while the
                       text box accepts a larger precise value (e.g. a
                       multi-minute protocol video). The slider then pins at
                       its own max; the setting + text keep the typed value.
        """
        settings = _app_ctx.ctx.settings
        slider = self.ids[slider_id]

        # The log name is derived from the widget id rather than passed in:
        # every text box here is '<name>_text' and its slider twin already logs
        # '<NAME>_<layer>', so the id IS the name and a separate parameter can
        # only drift from it or go unpassed. A non-conforming id is a caller
        # bug and says so rather than logging under a wrong name. Checked here,
        # before the settings write, so a bad caller cannot half-apply.
        if not text_id.endswith('_text'):
            raise ValueError(
                f"text_id {text_id!r} must end in '_text' -- the log name is derived from it"
            )
        record_name = f'{text_id.removesuffix("_text").upper()}_{self.layer}'

        # Captured before any path below rewrites the widget, so the record can
        # carry what the user actually typed rather than what we made of it.
        typed_text = self.ids[text_id].text

        # Read once and share: the refusal path below restores it, and the
        # no-edit check needs it. Two traversals of the same path drift.
        if settings_path:
            val = settings[self.layer]
            for p in settings_path.split('.'):
                val = val[p]
        else:
            val = settings[self.layer][settings_key]

        def put_back():
            self._initializing = True
            try:
                self.ids[text_id].text = str(val)
            finally:
                self._initializing = False

        raw = typed_number(typed_text, cast, put_back)
        if raw is None:
            # An unparseable entry is still a user action, and the reset above
            # would otherwise leave no trace of it. Both halves are recorded:
            # what was typed, and what the box was put back to. Separate names
            # because both lines have to survive -- the pair is what says the
            # entry was refused rather than accepted.
            gui_logger.text_input(record_name, typed_text)
            gui_logger.text_input(f'{record_name}_APPLIED', val)
            return False

        # The kv fires this handler on focus LOSS, not on edit
        # (`on_focus: if not self.focus: root.gain_text()`), so clicking into
        # a box and out again arrives here with the untouched stored value.
        # Clipping it would rewrite the store with the widget's bound: a layer
        # whose stored value legitimately sits above this camera's cap -- the
        # user's intent, kept on purpose -- would be destroyed by a stray
        # click, and the periodic flush would persist the loss. No edit, no
        # commit; the box already shows the stored value.
        if raw == val:
            return False

        upper = slider.max if value_max is None else value_max
        clipped = cast(np.clip(raw, slider.min, upper))

        _app_ctx.ctx.update_settings(f'{self.layer}.{settings_path or settings_key}', clipped)

        # The settings write above is the commit; this is the display half.
        self._show_value_on_widgets(slider_id, text_id, clipped, cast=cast)

        # text_input (not slider): this is a typed commit, and the twin slider
        # emits SLIDER for the same setting, so sharing the verb would make a
        # drag and a keystroke indistinguishable in the bundle.
        gui_logger.text_input(record_name, typed_text)

        # Only when clipping actually moved the value. The comparison is on the
        # PARSED number, not the strings: '5' typed into a float box becomes
        # 5.0, which is the same value and must not look like a correction.
        if raw != clipped:
            gui_logger.text_input(f'{record_name}_APPLIED', clipped)

        return True

    def _init_ui(self, dt=0):
        ctx = _app_ctx.ctx
        if ctx is None:
            self._init_ui_retries += 1
            if self._init_ui_retries > INIT_MAX_RETRIES:
                logger.error(
                    '[LVP Main  ] LayerControl._init_ui: ctx still None after 50 retries, giving up'
                )
                return
            Clock.schedule_once(self._init_ui, 0.1)
            return
        settings = ctx.settings

        if (
            self.layer in common_utils.get_fluorescence_layers()
            and settings['stimulation_enabled']
            and firmware_stim_supported()
        ):
            self.stimulation_support = True
            self.show_stim_controls = True
        else:
            self.stimulation_support = False
            self.show_stim_controls = False

        self.update_stim_controls_visibility()

        # Don't apply settings during initial UI setup - will be done after load_settings
        # Skip initialization of autogain and apply_settings here

    def cleanup_scrollviews(self):
        """
        Clean up ScrollView viewport resources in this LayerControl.
        Called when accordion is collapsed to prevent memory accumulation.
        """
        from ui.ui_helpers import cleanup_scrollview_viewport

        for child in self.walk():
            if isinstance(child, ScrollView):
                cleanup_scrollview_viewport(child)

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

    def ill_slider(self):
        settings = _app_ctx.ctx.settings
        if _app_ctx.ctx.session.run_lockout:
            return
        # Early return on programmatic updates (#617): when another code
        # path sets ill_slider.value directly (load_settings, ill_text,
        # set_step_state), on_value fires and re-enters
        # here. Without this guard, the handler overwrites the caller's
        # settings write and schedules a redundant apply_settings. Callers
        # are responsible for writing settings explicitly when they use
        # _initializing=True.
        if self._initializing:
            return
        logger.info('[LVP Main  ] LayerControl.ill_slider()')
        illumination = round(self.ids['ill_slider'].value)  # Round to integer (step=1)
        # Slider-vs-text divergence trace for the > ~150 mA silent-
        # fail bench investigation. See _FX2_DEBUG_WIRE block at top
        # of this file. INFO level -- this is a key divergence point
        # (int from slider vs float from text).
        if _fx2_wire_debug_enabled(settings):
            logger.info(
                '[FX2 LED diag] ill_slider ENTRY layer=%s raw_value=%r '
                'raw_type=%s -> illumination=%r type=%s source=slider',
                self.layer,
                self.ids['ill_slider'].value,
                type(self.ids['ill_slider'].value).__name__,
                illumination,
                type(illumination).__name__,
            )
        gui_logger.slider(f'ILLUMINATION_{self.layer}', illumination)
        _app_ctx.ctx.update_settings(f'{self.layer}.illumination_ma', illumination)

        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(illumination)
        if self.ids['ill_text'].text != new_text:
            self.ids['ill_text'].text = new_text
        self.apply_ill_slider()

    def ill_text(self) -> None:
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.ill_text()')
        typed_text = self.ids['ill_text'].text
        if not self._validate_and_apply_text_input(
            'ill_text',
            'ill_slider',
            'illumination_ma',
            # Before the scope is built there is no bound to read, and the
            # slider's own max is the only one there is.
            value_max=get_layer_illumination_text_max(self.layer),
        ):
            return
        # Text-entry divergence trace for the > ~150 mA silent-fail
        # bench investigation. See _FX2_DEBUG_WIRE block at top of
        # this file. INFO level -- this is the other key divergence
        # point (float from text vs int from slider).
        if _fx2_wire_debug_enabled(settings):
            illumination = settings[self.layer]['illumination_ma']
            logger.info(
                '[FX2 LED diag] ill_text ENTRY layer=%s raw_text=%r '
                '-> illumination=%r type=%s source=text',
                self.layer,
                typed_text,
                illumination,
                type(illumination).__name__,
            )
        self.apply_settings()

    def sum_slider(self):
        logger.info('[LVP Main  ] LayerControl.sum_slider()')
        total = int(self.ids['sum_slider'].value)
        gui_logger.slider(f'SUM_{self.layer}', total)
        _app_ctx.ctx.update_settings(f'{self.layer}.sum', total)
        self._refresh_sum_depth_hint()
        self.apply_settings()

    def sum_text(self):
        logger.info('[LVP Main  ] LayerControl.sum_text()')
        if self._validate_and_apply_text_input('sum_text', 'sum_slider', 'sum', cast=int):
            self._refresh_sum_depth_hint()
            self.apply_settings()

    def video_duration_slider(self):
        logger.info('[LVP Main  ] LayerControl.video_duration_slider()')
        duration = self.ids['video_duration_slider'].value
        gui_logger.slider(f'VIDEO_DURATION_{self.layer}', duration)
        _app_ctx.ctx.update_settings(f'{self.layer}.video_config.duration', duration)
        self.apply_settings()

    def video_duration_text(self):
        logger.info('[LVP Main  ] LayerControl.video_duration_text()')
        if self._validate_and_apply_text_input(
            'video_duration_text',
            'video_duration_slider',
            'duration',
            cast=int,
            settings_path='video_config.duration',
            # Slider quick-picks up to 60s; the text box accepts longer
            # protocol videos (no protocol cap) up to a 1-hour sanity bound.
            value_max=3600,
        ):
            self.apply_settings()

    def update_auto_gain(self):
        logger.info('[LVP Main  ] LayerControl.update_auto_gain()')
        enabled = self.ids['auto_gain'].state == 'down'
        gui_logger.toggle(f'AUTO_GAIN_{self.layer}', enabled)

        # Leaving auto-gain locks the camera's arm and stores what it reached;
        # the Session does both, so a script leaving auto-gain stores the same
        # thing. The lock waits on the camera, so it runs on the camera lane.
        # The redraw shows whatever the store holds afterwards -- the reached
        # values, or the old ones if the lock was refused -- and the apply
        # re-syncs the box and arms the camera when auto-gain went on.
        ctx = _app_ctx.ctx
        layer = self.layer

        def redraw():
            self.render_layer_values_from_settings()
            self.apply_settings()

        submit_reported(
            lambda: ctx.session.set_layer_auto_gain(layer, enabled),
            redraw,
            f'AUTO_GAIN_{layer}',
            lane=ctx.camera_executor,
        )

    def gain_slider(self):
        if _app_ctx.ctx.session.run_lockout:
            return
        # See ill_slider -- programmatic updates must not re-enter (#617).
        if self._initializing:
            return
        logger.info('[LVP Main  ] LayerControl.gain_slider()')
        gain = round(self.ids['gain_slider'].value, 1)  # Round to 1 decimal (step=0.1)
        gui_logger.slider(f'GAIN_{self.layer}', gain)
        _app_ctx.ctx.update_settings(f'{self.layer}.gain_db', gain)
        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(gain)
        if self.ids['gain_text'].text != new_text:
            self.ids['gain_text'].text = new_text
        if not self.ids['gain_slider'].disabled:
            self.apply_gain_slider()
        ####

    def gain_text(self):
        logger.info('[LVP Main  ] LayerControl.gain_text()')
        if self._validate_and_apply_text_input('gain_text', 'gain_slider', 'gain_db'):
            self.apply_gain_slider()

    def composite_threshold_slider(self):
        logger.info('[LVP Main  ] LayerControl.composite_threshold_slider()')
        composite_threshold = self.ids['composite_threshold_slider'].value
        gui_logger.slider(f'COMPOSITE_THRESHOLD_{self.layer}', composite_threshold)
        _app_ctx.ctx.update_settings(
            f'{self.layer}.composite_brightness_threshold', composite_threshold
        )

    def composite_threshold_text(self):
        logger.info('[LVP Main  ] LayerControl.composite_threshold_text()')
        self._validate_and_apply_text_input(
            'composite_threshold_text',
            'composite_threshold_slider',
            'composite_brightness_threshold',
        )

    def exp_slider(self):
        if _app_ctx.ctx.session.run_lockout:
            return
        # See ill_slider -- programmatic updates must not re-enter (#617).
        if self._initializing:
            return
        logger.info('[LVP Main  ] LayerControl.exp_slider()')
        exposure = round(self.ids['exp_slider'].value, 2)  # Round to 2 decimals (step=0.01)
        gui_logger.slider(f'EXPOSURE_{self.layer}', exposure)
        # exposure = 10 ** self.ids['exp_slider'].value # slider is log_10(ms)
        _app_ctx.ctx.update_settings(f'{self.layer}.exposure_ms', exposure)  # exposure in ms
        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(exposure)
        if self.ids['exp_text'].text != new_text:
            self.ids['exp_text'].text = new_text
        if not self.ids['exp_slider'].disabled:
            self.apply_exp_slider()

    def exp_text(self) -> None:
        logger.info('[LVP Main  ] LayerControl.exp_text()')
        if self._validate_and_apply_text_input(
            'exp_text',
            'exp_slider',
            'exposure_ms',
            # The box is bounded by what the sensor can actually honor, not by
            # the slider's manual range. With no camera to report a cap there
            # is no honest ceiling, so the slider's bound is the only one.
            value_max=get_exposure_text_max(),
        ):
            self.apply_exp_slider()

    def stim_freq_slider(self):
        logger.info('[LVP Main  ] LayerControl.stim_freq_slider()')
        frequency = self.ids['stim_freq_slider'].value
        gui_logger.slider(f'STIM_FREQ_{self.layer}', frequency)
        _app_ctx.ctx.update_settings(f'{self.layer}.stim_config.frequency', frequency)
        self.apply_settings()

    def stim_pulse_count_slider(self):
        logger.info('[LVP Main  ] LayerControl.stim_pulse_count_slider()')
        pulse_count = int(self.ids['stim_pulse_count_slider'].value)
        gui_logger.slider(f'STIM_PULSE_COUNT_{self.layer}', pulse_count)
        _app_ctx.ctx.update_settings(f'{self.layer}.stim_config.pulse_count', pulse_count)
        self.apply_settings()

    def stim_pulse_width_slider(self):
        logger.info('[LVP Main  ] LayerControl.stim_pulse_width_slider()')
        pulse_width = int(self.ids['stim_pulse_width_slider'].value)
        gui_logger.slider(f'STIM_PULSE_WIDTH_{self.layer}', pulse_width)
        _app_ctx.ctx.update_settings(f'{self.layer}.stim_config.pulse_width', pulse_width)
        self.apply_settings()

    def stim_freq_text(self):
        logger.info('[LVP Main  ] LayerControl.stim_freq_text()')
        if self._validate_and_apply_text_input(
            'stim_freq_text',
            'stim_freq_slider',
            'frequency',
            settings_path='stim_config.frequency',
        ):
            self.apply_settings()

    def stim_pulse_count_text(self):
        logger.info('[LVP Main  ] LayerControl.stim_pulse_count_text()')
        if self._validate_and_apply_text_input(
            'stim_pulse_count_text',
            'stim_pulse_count_slider',
            'pulse_count',
            cast=int,
            settings_path='stim_config.pulse_count',
        ):
            self.apply_settings()

    def stim_pulse_width_text(self):
        logger.info('[LVP Main  ] LayerControl.stim_pulse_width_text()')
        if self._validate_and_apply_text_input(
            'stim_pulse_width_text',
            'stim_pulse_width_slider',
            'pulse_width',
            cast=int,
            settings_path='stim_config.pulse_width',
        ):
            self.apply_settings()

    def stim_ill_slider(self):
        logger.info('[LVP Main  ] LayerControl.stim_ill_slider()')
        illumination = round(self.ids['stim_ill_slider'].value)
        gui_logger.slider(f'STIM_ILL_{self.layer}', illumination)
        _app_ctx.ctx.update_settings(f'{self.layer}.stim_config.illumination_ma', illumination)
        new_text = str(illumination)
        if self.ids['stim_ill_text'].text != new_text:
            self.ids['stim_ill_text'].text = new_text
        self.apply_settings()

    def stim_ill_text(self):
        logger.info('[LVP Main  ] LayerControl.stim_ill_text()')
        if self._validate_and_apply_text_input(
            'stim_ill_text',
            'stim_ill_slider',
            'illumination_ma',
            cast=int,
            settings_path='stim_config.illumination_ma',
        ):
            self.apply_settings()

    def false_color(self):
        logger.info('[LVP Main  ] LayerControl.false_color()')
        enabled = bool(self.ids['false_color'].active)
        gui_logger.toggle(f'FALSE_COLOR_{self.layer}', enabled)
        _app_ctx.ctx.update_settings(f'{self.layer}.false_color', enabled)
        self.apply_settings()

    def log_histogram_scale(self) -> None:
        """Record the histogram log-scale checkbox.

        Display-only: the histogram reads this box directly when it redraws,
        so there is no setting to write and nothing to record but the gesture.
        """
        gui_logger.toggle(f'LOG_HISTOGRAM_{self.layer}', bool(self.ids['logHistogram_id'].active))

    def update_acquire(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.update_acquire()')

        if self.ids['acquire_image'].active:
            mode = 'image'
        elif self.ids['acquire_video'].active:
            mode = 'video'
        else:
            mode = 'none'
        gui_logger.select(f'ACQUIRE_{self.layer}', mode)

        # The Session stops the layer stimulating when it acquires; the
        # stimulation switch is drawn off to match.
        _app_ctx.ctx.session.set_layer_acquire(self.layer, None if mode == 'none' else mode)
        if mode != 'none':
            self.ids['stim_disable_btn'].active = True
            self.show_stim_controls = False

        if 'stim_config' in settings[self.layer]:
            self.update_stim_controls_visibility()

    def update_stim_enable(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.update_stim_enable()')
        enabled = self.ids['stim_enable_btn'].active
        gui_logger.toggle(f'STIM_{self.layer}', enabled)
        if self.ids['stim_enable_btn'].active:
            if (
                'stim_config' in settings[self.layer]
                and settings[self.layer]['stim_config'] is not None
            ):
                settings[self.layer]['stim_config']['enabled'] = True
            settings[self.layer]['acquire'] = None
            self.ids['acquire_none'].active = True
            self.ids['acquire_none'].state = 'down'
        elif (
            'stim_config' in settings[self.layer]
            and settings[self.layer]['stim_config'] is not None
        ):
            settings[self.layer]['stim_config']['enabled'] = False

        self.update_stim_controls_visibility()

    def update_autofocus(self):
        logger.info('[LVP Main  ] LayerControl.update_autofocus()')
        enabled = bool(self.ids['autofocus'].active)
        gui_logger.toggle(f'AUTOFOCUS_ENABLED_{self.layer}', enabled)
        _app_ctx.ctx.update_settings(f'{self.layer}.autofocus', enabled)

    def save_focus(self):
        gui_logger.button(f'SAVE_FOCUS_{self.layer}')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] LayerControl.save_focus()')
        # The selected step and the protocol are taken at the click: the
        # step the user was looking at, in the protocol the panel holds now
        # (the panel replaces its protocol on New and Load). The panel's -1
        # is no step selected.
        protocol_settings = ctx.motion_settings.ids['protocol_settings_id']
        protocol = protocol_settings._protocol
        selected_step = int(protocol_settings.curr_step)
        step_idx = selected_step if selected_step >= 0 else None
        run_reported(
            lambda: ctx.session.save_focus(protocol, self.layer, step_idx=step_idx),
            lambda: self._refresh_step_views(ctx, protocol),
            f'SAVE_FOCUS_{self.layer}',
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
        gui_logger.button(f'APPLY_FOCUS_TO_STEPS_{self.layer}')
        ctx = _app_ctx.ctx
        logger.info('[LVP Main  ] LayerControl.apply_focus_to_channel_steps()')
        # As Save Focus: the protocol the panel holds at the click.
        protocol = ctx.motion_settings.ids['protocol_settings_id']._protocol
        run_reported(
            lambda: ctx.session.apply_focus_to_layer_steps(protocol, self.layer),
            lambda: self._refresh_step_views(ctx, protocol),
            f'APPLY_FOCUS_TO_STEPS_{self.layer}',
        )

    def goto_focus(self):
        from ui.ui_helpers import move_absolute

        gui_logger.button(f'GOTO_FOCUS_{self.layer}')
        logger.info('[LVP Main  ] LayerControl.goto_focus()')
        run_reported(
            lambda: move_absolute('Z', _app_ctx.ctx.session.saved_focus(self.layer)),
            None,
            f'GOTO_FOCUS_{self.layer}',
        )

    def led_toggle(self):
        """The Enable LED button's press: record it, then drive the LED.

        The record is here and not in update_led_state, because every
        apply_settings also runs that, and an apply is not a press. Writing
        the button's state is not a press either: it never dispatches
        on_release, so the app may set it without a guard.
        """
        gui_logger.toggle(f'LED_{self.layer}', self.ids['enable_led_btn'].state == 'down')
        self.update_led_state()

    def update_led_state(self, apply_settings=True):
        ctx = _app_ctx.ctx
        # No autofocus guard here: while AF holds the LED ownership lease,
        # led_on/led_off refuse any unleased UI write, so a live UI apply during
        # a scan (e.g. the exposure field losing focus when the AF button is
        # clicked) cannot turn off the channel AF is using. The lease is the
        # structural guard; an early-return here would only duplicate it.
        if self._initializing:
            return
        settings = ctx.settings
        enabled = self.ids['enable_led_btn'].state == 'down'
        illumination = settings[self.layer]['illumination_ma']

        if apply_settings:
            self.apply_settings(update_led=False)

        # The colour string goes to the seam unmapped: the illumination API
        # owns colour-to-channel resolution, so turning OFF a colour this scope
        # cannot drive is a no-op there, and turning one ON fails with the
        # colour named instead of a sentinel channel.
        illumination_api = ctx.scope.illumination
        layer = self.layer
        if enabled:
            logger.info(f'[LVP Main  ] update_led_state: led_on({layer}, {illumination})')
            call = functools.partial(illumination_api.led_on, layer, illumination)
        else:
            call = functools.partial(illumination_api.led_off, layer)
        # The toggle shows what the API reports lit once the command has
        # landed, so a refused one goes back.
        submit_reported(
            call,
            ctx.ui_listener_bridge.reconcile_led_buttons,
            f'LED_{layer}',
            lane=ctx.io_executor,
        )

    # update_led_toggle_ui() removed -- LED observer handles UI sync.
    # See Phase 1 commit 96defe3.

    def set_step_state(self, step: dict):
        """Display a protocol step in this layer's widgets. Writes NO settings.

        A protocol run displays every step here without changing the user's
        live-view settings; manual step navigation writes the settings
        itself before applying them. What keeps this a pure display write
        is the kv wiring: of the widgets set here only the illumination /
        gain / exposure sliders bind ``on_value``, and each of those
        handlers returns while ``_initializing`` is set; the other sliders
        bind ``on_release`` (a touch-up or the wheel, never a programmatic
        value) and a CheckBox's ``on_release`` never fires from ``.active``.

        Only updates widgets for keys that are present in *step*.
        This allows partial updates (e.g. stim-config-only for non-current
        layers) without clobbering unrelated widget values.

        Args:
            step: Protocol step dict.  Recognized keys: 'Illumination',
                'Gain', 'Exposure', 'Sum', 'Auto_Focus', 'Auto_Gain',
                'False_Color', 'Acquire', 'Video Config', 'Stim_Config'.
        """
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
                self.ids['auto_gain'].active = _app_ctx.ctx.scope.imaging.applied_auto_gain_for(
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

            # Stim config (only for this layer's stim settings)
            sc = step.get('Stim_Config')
            if isinstance(sc, dict) and self.layer in sc:
                stim = sc[self.layer]
                if stim.get('enabled', False):
                    self.ids['stim_enable_btn'].active = True
                    self.ids['stim_disable_btn'].active = False
                else:
                    self.ids['stim_disable_btn'].active = True
                    self.ids['stim_enable_btn'].active = False
                self.update_stim_controls_visibility()
                self.ids['stim_ill_text'].text = str(stim.get('illumination_ma', 100))
                self.ids['stim_ill_slider'].value = float(stim.get('illumination_ma', 100))
                self.ids['stim_freq_text'].text = str(stim.get('frequency', 1))
                self.ids['stim_freq_slider'].value = float(stim.get('frequency', 1))
                self.ids['stim_pulse_width_text'].text = str(stim.get('pulse_width', 10))
                self.ids['stim_pulse_width_slider'].value = float(stim.get('pulse_width', 10))
                self.ids['stim_pulse_count_text'].text = str(stim.get('pulse_count', 1))
                self.ids['stim_pulse_count_slider'].value = int(stim.get('pulse_count', 1))

            # Acquire type
            if 'Acquire' in step:
                for sel in ('acquire_video', 'acquire_image', 'acquire_none'):
                    self.ids[sel].active = False
                acquire = step['Acquire']
                if acquire == 'video':
                    self.ids['acquire_video'].active = True
                elif acquire == 'image':
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

        Both writes run with the layer's handlers suppressed: these are
        display, and a slider's on_value handler exists to record what the USER
        did -- unsuppressed, it commits the written value back over the stored
        one and logs it as a drag. The previous flag state is RESTORED rather
        than cleared, so a caller that is already suppressing (a layer still
        initializing, a capability sync narrowing every slider) still has its
        own guard when this returns.
        """
        slider = self.ids[slider_id]
        on_slider = cast(np.clip(value, slider.min, slider.max))
        was_initializing = self._initializing
        self._initializing = True
        try:
            slider.value = on_slider
            self.ids[text_id].text = str(value)
        finally:
            self._initializing = was_initializing

    def render_layer_values_from_settings(self, layer_settings=None):
        """Render this layer's stored illumination, gain and exposure.

        The one store-to-widget path for the three settings that appear on both
        a slider and a text box. Callers reconcile the store against the
        camera's PHYSICAL caps before calling: a value the hardware cannot
        honor is wrong in the store, not merely too large for the slider, and
        pinning the slider would hide it instead of correcting it.

        Pass *layer_settings* when the caller already holds a snapshot of this
        layer's settings. Reading the store again here would take the lock a
        second time and could see a DIFFERENT state, leaving some widgets
        rendered from one snapshot and some from another; a caller with no
        snapshot passes nothing and this reads the store once, under the lock.
        """
        if layer_settings is None:
            ctx = _app_ctx.ctx
            with ctx.settings_lock:
                stored = ctx.settings[self.layer]
                layer_settings = {
                    key: stored[key] for _, _, key in _LAYER_VALUE_WIDGETS if key in stored
                }
        for slider_id, text_id, settings_key in _LAYER_VALUE_WIDGETS:
            if settings_key in layer_settings:
                self._show_value_on_widgets(slider_id, text_id, layer_settings[settings_key])

    def sync_widgets_from_settings(self):
        """Point every widget of this layer at its stored settings.

        The one settings-to-widgets direction: startup, the end of a
        protocol run (which displayed each step here without writing the
        settings) and the end of a standalone autofocus (which restored
        the camera from the settings) all call this. Reads the layer's
        settings once under the settings lock; writes only widgets. An
        uncommitted text edit (typed, no Enter) is deliberately dropped:
        the widget tells the settings' truth again.

        The gain and exposure sliders' ``max`` is a camera fact set by the
        caller that learns it; the value written here follows whatever
        max the slider carries.
        """
        ctx = _app_ctx.ctx
        with ctx.settings_lock:
            layer_settings = copy.deepcopy(ctx.settings[self.layer])

        self._initializing = True
        try:
            if self.layer in common_utils.get_fluorescence_layers():
                self.ids['composite_threshold_slider'].value = layer_settings[
                    'composite_brightness_threshold'
                ]

            self.render_layer_values_from_settings(layer_settings)

            self.ids['false_color'].active = layer_settings['false_color']
            self.ids['sum_slider'].value = layer_settings.get('sum', 1)

            if layer_settings['acquire'] == 'image':
                self.ids['acquire_image'].active = True
            elif layer_settings['acquire'] == 'video':
                self.ids['acquire_video'].active = True
            else:
                self.ids['acquire_none'].active = True

            video_config = layer_settings['video_config']
            self.ids['video_duration_text'].text = str(video_config['duration'])
            self.ids['video_duration_slider'].value = video_config['duration']

            self.ids['autofocus'].active = layer_settings['autofocus']
            # The box shows the enable actually in force, not the bare
            # stored preference, the same way apply_settings shows it.
            self.ids['auto_gain'].active = self.effective_auto_gain()

            # Shipped settings carry no stim_config on the transmitted
            # layers, and a None is representable.
            stim_config = layer_settings.get('stim_config')
            if stim_config:
                # Default to hidden until enabled
                self.show_stim_controls = False

                self.ids['stim_enable_btn'].active = stim_config['enabled']
                self.ids['stim_disable_btn'].active = not stim_config['enabled']
                self.ids['stim_ill_text'].text = str(stim_config.get('illumination_ma', 100))
                self.ids['stim_ill_slider'].value = float(stim_config.get('illumination_ma', 100))
                self.ids['stim_freq_text'].text = str(stim_config['frequency'])
                self.ids['stim_freq_slider'].value = float(stim_config['frequency'])
                self.ids['stim_pulse_width_text'].text = str(stim_config['pulse_width'])
                self.ids['stim_pulse_width_slider'].value = float(stim_config['pulse_width'])
                self.ids['stim_pulse_count_text'].text = str(stim_config['pulse_count'])
                self.ids['stim_pulse_count_slider'].value = int(stim_config['pulse_count'])

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
        """The auto-gain enable actually in force for this layer's camera.

        The API's answer for the saved preference (settings[layer]['auto_gain']):
        a camera without hardware auto-gain runs manual, and its Auto Gain/Exp
        control is hidden. Read, never written back, so a capable camera's
        saved preference survives a swap to a camera without it and back.
        """
        ctx = _app_ctx.ctx
        return ctx.scope.imaging.applied_auto_gain_for(
            ctx.settings[self.layer]['auto_gain']
        ).applied

    def apply_settings(self, update_led=True):

        # Skip apply_settings if layer is still initializing
        if getattr(self, '_initializing', False):
            return

        logger.debug(f'[LVP Main  ] {self.layer}_LayerControl.apply_settings()')

        ctx = _app_ctx.ctx

        camera_executor = ctx.camera_executor
        from ui.image_settings import set_histogram_layer

        def update_shader(dt=None):
            thread = getattr(ctx, 'scope_display_thread', None)
            if (
                thread is not None
                and not thread.is_paused
                and ctx.scope_display.use_bullseye is False
            ):
                self.update_shader(dt=0)

        def disable_leds_for_other_layers(dt=None):
            if self.ids['enable_led_btn'].state == 'down':
                # Turn off any OTHER layer's LED so only this layer's channel
                # stays lit (one LED on at a time at the hardware level).
                # Switch the others off individually rather than blanking all
                # LEDs and re-lighting this one: the nuclear leds_off clears
                # the LED-state cache, which forces this channel to re-fire
                # and blink off then on on every slider move. led_off self-
                # skips a channel that is already off, so this loop touches
                # the bus only for a layer that is actually on -- no cycle on
                # a plain slider move, and this layer's own LED is never
                # disturbed (its current is owned by update_led_state).
                if not ctx.session.run_lockout:
                    illumination_api = ctx.scope.illumination
                    for layer in common_utils.get_layers():
                        if layer == self.layer:
                            continue
                        # None with no LED board installed: nothing is lit.
                        state = illumination_api.get_led_state(channel=layer)
                        if state is not None and state['enabled']:
                            submit_reported(
                                lambda lit=layer: illumination_api.led_off(lit),
                                None,
                                f'LED_{layer}_OFF',
                                lane=ctx.io_executor,
                            )
                # Update button states (visual only -- hardware already handled)
                for layer in common_utils.get_layers():
                    if layer != self.layer:
                        layer_obj = ctx.image_settings.layer_lookup(layer=layer)
                        btn = layer_obj.ids['enable_led_btn']
                        if btn.state != 'normal':
                            btn.state = 'normal'

        if ctx.session.run_lockout:
            # Protocol actively running -- capture() handles camera settings
            # per-step. Don't apply here to avoid duplicate commands (#587/#588).
            logger.debug(
                f'[APPLY_SETTINGS DIAG] {self.layer} -- early return '
                f'(protocol running). Camera settings NOT applied.'
            )
            Clock.schedule_once(disable_leds_for_other_layers, 0)
            Clock.schedule_once(update_shader, 0)
            return
        # All other cases: apply camera settings normally.

        # global gain_vals

        # update illumination to currently selected settings
        # -----------------------------------------------------
        set_histogram_layer(active_layer=self.layer)

        # Queue IO task and update UI after completing IO
        if update_led and not ctx.session.run_lockout:
            self.update_led_state(apply_settings=False)

        disable_leds_for_other_layers()

        if not ctx.session.run_lockout:
            # Effective enable = the API's answer for the saved preference
            # (effective_auto_gain): on a camera without hardware AG/AE the
            # control is hidden, so a stored True must read as off here -- else
            # the gain/exposure sliders below stay disabled with no UI to clear
            # them. Non-destructive: the stored preference is left intact.
            auto_gain_enabled = self.effective_auto_gain()
            # Sync the toggle CheckBox to the settings value before applying
            # to the camera. The .kv has no Kivy binding from settings to
            # auto_gain.active, so when the JSON loads at startup with
            # auto_gain=True the CheckBox stays at its default False;
            # apply_settings would then send AG=True to the camera while the
            # toggle continues to read OFF in the UI. Programmatic
            # .active = bool fires no on_release (CheckBox only binds
            # on_release in the .kv), so this does not re-enter. The
            # gain/exposure widgets' enabled state follows this box through
            # the kv rule (`disabled: app.run_lockout or auto_gain.active`):
            # an imperative .disabled write here was erased whenever the run
            # lockout cleared, because that rule re-fires on the edge.
            self.ids['auto_gain'].active = auto_gain_enabled
            session = ctx.session
            layer = self.layer
            submit_reported(
                lambda: session.apply_layer_camera(layer),
                None,
                f'CAMERA_SETTINGS_{layer}',
                lane=camera_executor,
            )

        # update false color to currently selected settings and shader
        # -----------------------------------------------------
        update_shader()

    def update_shader(self, dt):
        ctx = _app_ctx.ctx
        # logger.info('[LVP Main  ] LayerControl.update_shader()')
        if self.ids['false_color'].active:
            ctx.viewer.update_shader(self.layer)
        else:
            ctx.viewer.update_shader('none')
