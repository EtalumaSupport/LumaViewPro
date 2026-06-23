# Copyright Etaluma, Inc.
import logging
import os

import numpy as np

from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.image_mode as image_mode
from modules import gui_logger
from modules.sequential_io_executor import IOTask

logger = logging.getLogger('LVP.ui.layer_control')

# Brightfield allows higher illumination/exposure than fluorescence channels
# because BF LED power is lower and longer exposures don't risk photobleaching.
BF_MAX_ILLUMINATION = 500
BF_MAX_EXPOSURE_MS = 1000
FLUORESCENCE_MIN_EXPOSURE_MS = 1.0
# AG can drive transmitted-channel exposure down to the camera's
# physical minimum (Pylon ExposureTime.Min ~= 30 us = 0.030 ms on
# common sensors). Sub-threshold values written back to settings via
# update_auto_gain_cb then fire the set_exposure_time(<0.1ms)
# "value should be in milliseconds" warning on every subsequent
# apply_settings (visible in beta9 logs as recurring WARNING spam).
# The threshold matches set_exposure_time's internal warning gate so
# AG-feedback values can never trigger it; live AG can still drive
# the camera lower (the floor applies only to the settings write-back
# in update_auto_gain_cb).
TRANSMITTED_MIN_EXPOSURE_MS = 0.1
SLIDER_DEBOUNCE_S = 0.1
INIT_MAX_RETRIES = 50

# ------------------------------------------------------------------
# Diagnostic toggle for the "illumination slider > ~150 mA silently
# fails to light LED on LS620 FX2" bench investigation
# (2026-04-16). Logs type + value at the slider vs text entry points
# so the bench trace can show whether the two code paths diverge
# here (int vs float) or further downstream. Companion gates live
# in drivers/fx2driver.py (byte-level wire trace) and
# modules/lumascope_api/illumination.py (cache-equality check).
# Toggle by either:
#   * set fx2_debug_wire_enabled: true in data/settings.json
#   * flip _FX2_DEBUG_WIRE = True  below
# ------------------------------------------------------------------
_FX2_DEBUG_WIRE = False


def _read_fx2_wire_setting() -> bool:
    """Read fx2_debug_wire_enabled from settings.json at module import.

    Replaces the prior LVP_FX2_DEBUG_WIRE environment-variable gate.
    """
    from modules.settings_init import load_fx2_debug_wire_setting

    try:
        import lvp_logger

        base_dir = lvp_logger.lvp_appdata
    except (ImportError, AttributeError):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return load_fx2_debug_wire_setting(base_dir)


_FX2_WIRE_SETTING = _read_fx2_wire_setting()


def _fx2_wire_debug_enabled() -> bool:
    return _FX2_DEBUG_WIRE or _FX2_WIRE_SETTING


class LayerControl(BoxLayout):
    layer = StringProperty(None)
    bg_color = ObjectProperty(None)
    illumination_support = BooleanProperty(True)
    stimulation_support = BooleanProperty(False)
    show_stim_controls = BooleanProperty(False)
    autogain_support = BooleanProperty(True)
    exposure_summing_support = BooleanProperty(False)
    show_camera_controls = BooleanProperty(True)
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
        """
        scope_display = getattr(_app_ctx.ctx, 'scope_display', None)
        if scope_display is None:
            return
        scope_display.bind(image_mode=lambda *a: self._refresh_sum_depth_hint())
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
        active = image_mode.depth_truncation_warning_active(sum_count, scope_display.image_mode)
        row = self.ids['sum_depth_hint_row']
        row.height = dp(30) if active else 0
        row.opacity = 1 if active else 0

    def _validate_and_apply_text_input(
        self,
        text_id: str,
        slider_id: str,
        settings_key: str,
        cast=float,
        settings_path: str | None = None,
        gui_log_name: str | None = None,
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
            gui_log_name: Name for gui_logger.slider() call (e.g., 'GAIN')
            value_max: Optional upper bound for the typed value when it should
                       exceed the slider's own max -- the slider is a coarse
                       quick-pick (e.g. video duration up to 60s) while the
                       text box accepts a larger precise value (e.g. a
                       multi-minute protocol video). The slider then pins at
                       its own max; the setting + text keep the typed value.
        """
        settings = _app_ctx.ctx.settings
        slider = self.ids[slider_id]
        try:
            raw = cast(self.ids[text_id].text)
        except (ValueError, TypeError):
            logger.debug(f'[LVP Main  ] Invalid {settings_key} input: {self.ids[text_id].text!r}')
            # Reset to current valid value (M21)
            if settings_path:
                parts = settings_path.split('.')
                val = settings[self.layer]
                for p in parts:
                    val = val[p]
            else:
                val = settings[self.layer][settings_key]
            self._initializing = True
            try:
                self.ids[text_id].text = str(val)
            finally:
                self._initializing = False
            return False

        upper = slider.max if value_max is None else value_max
        clipped = cast(np.clip(raw, slider.min, upper))

        # Update settings
        if settings_path:
            parts = settings_path.split('.')
            target = settings[self.layer]
            for p in parts[:-1]:
                target = target[p]
            target[parts[-1]] = clipped
        else:
            settings[self.layer][settings_key] = clipped

        # Update widgets -- wrapped in _initializing so the slider's
        # on_value handler does not re-enter and double-fire apply_settings
        # (#617). Settings are already written above.
        self._initializing = True
        try:
            # The slider can only represent up to its own max; a typed value
            # above value_max's allowance pins the slider at max while the
            # setting + text keep the larger value.
            slider_value = min(clipped, slider.max)
            slider.value = float(slider_value) if cast is float else int(slider_value)
            self.ids[text_id].text = str(clipped)
        finally:
            self._initializing = False

        if gui_log_name:
            gui_logger.slider(f'{gui_log_name}_{self.layer}', clipped)

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

        from modules.config_ui_getters import firmware_stim_supported

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

        self.init_acquire()
        self.init_autofocus()

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
        settings = _app_ctx.ctx.settings
        self.show_camera_controls = False
        settings[self.layer]['acquire'] = None
        self.ids['acquire_none'].active = True

    def ill_slider(self):
        settings = _app_ctx.ctx.settings
        protocol_running_global = _app_ctx.ctx.protocol_running
        if protocol_running_global.is_set():
            return
        # Early return on programmatic updates (#617): when another code
        # path sets ill_slider.value directly (load_settings, ill_text,
        # set_step_state, camera listener), on_value fires and re-enters
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
        if _fx2_wire_debug_enabled():
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
        settings[self.layer]['ill_ma'] = illumination

        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(illumination)
        if self.ids['ill_text'].text != new_text:
            self.ids['ill_text'].text = new_text
        self.apply_ill_slider()

    def ill_text(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.ill_text()')
        ill_min = self.ids['ill_slider'].min
        if self.layer == 'BF':
            ill_max = BF_MAX_ILLUMINATION
        else:
            ill_max = self.ids['ill_slider'].max
        try:
            ill_val = float(self.ids['ill_text'].text)
        except Exception:
            logger.debug(f'[LVP Main  ] Invalid illumination input: {self.ids["ill_text"].text!r}')
            # Show current valid value so user knows input was rejected (M21)
            self._initializing = True
            try:
                self.ids['ill_text'].text = str(settings[self.layer]['ill_ma'])
            finally:
                self._initializing = False
            return

        illumination = float(np.clip(ill_val, ill_min, ill_max))
        # Text-entry divergence trace for the > ~150 mA silent-fail
        # bench investigation. See _FX2_DEBUG_WIRE block at top of
        # this file. INFO level -- this is the other key divergence
        # point (float from text vs int from slider).
        if _fx2_wire_debug_enabled():
            logger.info(
                '[FX2 LED diag] ill_text ENTRY layer=%s raw_text=%r '
                'parsed_val=%r -> illumination=%r type=%s source=text',
                self.layer,
                self.ids['ill_text'].text,
                ill_val,
                illumination,
                type(illumination).__name__,
            )
        settings[self.layer]['ill_ma'] = illumination

        # Wrap programmatic widget writes so on_value does not re-enter
        # ill_slider and re-fire apply_settings (#617).
        self._initializing = True
        try:
            self.ids['ill_slider'].value = float(
                np.clip(illumination, ill_min, self.ids['ill_slider'].max)
            )
            self.ids['ill_text'].text = str(illumination)
        finally:
            self._initializing = False

        self.apply_settings()

    def sum_slider(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.sum_slider()')
        total = int(self.ids['sum_slider'].value)
        gui_logger.slider(f'SUM_{self.layer}', total)
        settings[self.layer]['sum'] = total
        self._refresh_sum_depth_hint()
        self.apply_settings()

    def sum_text(self):
        logger.info('[LVP Main  ] LayerControl.sum_text()')
        if self._validate_and_apply_text_input('sum_text', 'sum_slider', 'sum', cast=int):
            self._refresh_sum_depth_hint()
            self.apply_settings()

    def video_duration_slider(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.video_duration_slider()')
        duration = self.ids['video_duration_slider'].value
        gui_logger.slider(f'VIDEO_DURATION_{self.layer}', duration)
        settings[self.layer]['video_config']['duration'] = duration
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

    def update_auto_gain(self, init: bool = False):
        camera_executor = _app_ctx.ctx.camera_executor
        logger.info('[LVP Main  ] LayerControl.update_auto_gain()')
        if self.ids['auto_gain'].state == 'down':
            state = True
        else:
            state = False
        if not init:
            gui_logger.toggle(f'AUTO_GAIN_{self.layer}', state)

        for item in ('gain_slider', 'gain_text', 'exp_slider', 'exp_text'):
            self.ids[item].disabled = state

        # When transitioning out of auto-gain, keep last auto-gain settings to apply
        camera_executor.put(
            IOTask(
                action=LayerControl.get_gain_exposure,
                args=(self, init, state),
                callback=LayerControl.update_auto_gain_cb,
                cb_args=(self),
                pass_result=True,
            )
        )

        # actual_gain = lumaview.scope.camera.get_gain()
        # actual_exp = lumaview.scope.camera.get_exposure_t()

    def get_gain_exposure(self, init, state):
        ctx = _app_ctx.ctx
        # Read directly from camera hardware, not cache.
        # During auto-gain, the SDK adjusts gain/exposure but doesn't
        # update the cache -- cache still has the pre-auto-gain values.
        actual_gain = ctx.scope.imaging.get_gain()
        actual_exp = ctx.scope.imaging.get_exposure_time()

        return (init, state, actual_gain, actual_exp)

    def update_auto_gain_cb(self, result=None, exception=None):
        settings = _app_ctx.ctx.settings
        try:
            if exception is not None:
                logger.error(f'LVP Main] Update_auto_gain error: {exception}')
                return

            init = result[0]
            state = result[1]
            gain = result[2]
            exp = result[3]

            if self.ids['auto_gain'].state == 'down':
                state = True
            else:
                state = False

            # If being called on program initialization, we don't want to
            # inadvertantly load the settings from the scope hardware into the software maintained settings
            # print("AUTOGAIN")
            # print(f"init: {init}    state: {state}")
            # print(f"Gain: {gain}    Exp: {exp}")

            if (not init) and (not state):
                # Clamp exposure to a per-class minimum before writing back
                # to settings. AG can drive the camera to its physical
                # minimum (Pylon ~30us on bright samples); writing those
                # raw values to settings produces (a) nearly-black images
                # if the user creates protocol steps from these settings,
                # and (b) recurring set_exposure_time(<0.1ms) WARNING spam
                # on every subsequent apply_settings. Fluorescence + lumi
                # floor at 1ms (sub-ms never realistic in those modes);
                # transmitted (BF/PC/DF) floor at 0.1ms (the warning
                # threshold). Live AG output to the camera is untouched.
                exp_min = self.ids['exp_slider'].min
                exp_max = self.ids['exp_slider'].max
                if self.layer in common_utils.get_image_layers():
                    exp_min = max(exp_min, FLUORESCENCE_MIN_EXPOSURE_MS)
                else:
                    exp_min = max(exp_min, TRANSMITTED_MIN_EXPOSURE_MS)
                exp = float(np.clip(exp, exp_min, exp_max))

                settings[self.layer]['gain_db'] = gain
                settings[self.layer]['exp_ms'] = exp
                # Update sliders/text to show the auto-adjusted values
                self.ids['gain_slider'].value = gain
                self.ids['gain_text'].text = str(round(gain, 1))
                self.ids['exp_slider'].value = exp
                self.ids['exp_text'].text = str(round(exp, 2))

            settings[self.layer]['auto_gain'] = state
            self.apply_settings()

        except Exception as e:
            logger.error(f'LVP Main] Update_auto_gain error: {e}')
            return

    def gain_slider(self):
        settings = _app_ctx.ctx.settings
        protocol_running_global = _app_ctx.ctx.protocol_running
        if protocol_running_global.is_set():
            return
        # See ill_slider -- programmatic updates must not re-enter (#617).
        if self._initializing:
            return
        logger.info('[LVP Main  ] LayerControl.gain_slider()')
        gain = round(self.ids['gain_slider'].value, 1)  # Round to 1 decimal (step=0.1)
        gui_logger.slider(f'GAIN_{self.layer}', gain)
        settings[self.layer]['gain_db'] = gain
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
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.composite_threshold_slider()')
        composite_threshold = self.ids['composite_threshold_slider'].value
        gui_logger.slider(f'COMPOSITE_THRESHOLD_{self.layer}', composite_threshold)
        settings[self.layer]['composite_brightness_threshold'] = composite_threshold

    def composite_threshold_text(self):
        logger.info('[LVP Main  ] LayerControl.composite_threshold_text()')
        self._validate_and_apply_text_input(
            'composite_threshold_text',
            'composite_threshold_slider',
            'composite_brightness_threshold',
        )

    def exp_slider(self):
        settings = _app_ctx.ctx.settings
        protocol_running_global = _app_ctx.ctx.protocol_running
        if protocol_running_global.is_set():
            return
        # See ill_slider -- programmatic updates must not re-enter (#617).
        if self._initializing:
            return
        logger.info('[LVP Main  ] LayerControl.exp_slider()')
        exposure = round(self.ids['exp_slider'].value, 2)  # Round to 2 decimals (step=0.01)
        gui_logger.slider(f'EXPOSURE_{self.layer}', exposure)
        # exposure = 10 ** self.ids['exp_slider'].value # slider is log_10(ms)
        settings[self.layer]['exp_ms'] = exposure  # exposure in ms
        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(exposure)
        if self.ids['exp_text'].text != new_text:
            self.ids['exp_text'].text = new_text
        if not self.ids['exp_slider'].disabled:
            self.apply_exp_slider()

    def exp_text(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.exp_text()')
        exp_min = self.ids['exp_slider'].min
        # exp_max = self.ids['exp_slider'].max
        if self.layer == 'BF':
            exp_max = BF_MAX_EXPOSURE_MS
        else:
            exp_max = self.ids['exp_slider'].max

        try:
            exp_val = float(self.ids['exp_text'].text)
        except Exception:
            logger.debug(f'[LVP Main  ] Invalid exposure input: {self.ids["exp_text"].text!r}')
            # Show current valid value so user knows input was rejected (M21)
            self._initializing = True
            try:
                self.ids['exp_text'].text = str(settings[self.layer]['exp_ms'])
            finally:
                self._initializing = False
            return

        exposure = float(np.clip(exp_val, exp_min, exp_max))

        settings[self.layer]['exp_ms'] = exposure

        # Wrap programmatic widget writes so on_value does not re-enter
        # exp_slider and re-fire apply_exp_slider (#617).
        self._initializing = True
        try:
            self.ids['exp_slider'].value = float(
                np.clip(exposure, exp_min, self.ids['exp_slider'].max)
            )
            # self.ids['exp_slider'].value = float(np.log10(exposure)) # convert slider to log_10
            self.ids['exp_text'].text = str(exposure)
        finally:
            self._initializing = False

        self.apply_exp_slider()

    def stim_freq_slider(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.stim_freq_slider()')
        frequency = self.ids['stim_freq_slider'].value
        gui_logger.slider(f'STIM_FREQ_{self.layer}', frequency)
        try:
            settings[self.layer]['stim_config']['frequency'] = frequency
        except Exception as e:
            logger.error(f'[LVP Main  ] LayerControl.stim_freq_slider() -> {e}')
        self.apply_settings()

    def stim_pulse_count_slider(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.stim_pulse_count_slider()')
        pulse_count = int(self.ids['stim_pulse_count_slider'].value)
        gui_logger.slider(f'STIM_PULSE_COUNT_{self.layer}', pulse_count)
        try:
            settings[self.layer]['stim_config']['pulse_count'] = pulse_count
        except Exception as e:
            logger.error(f'[LVP Main  ] LayerControl.stim_pulse_count_slider() -> {e}')
        self.apply_settings()

    def stim_pulse_width_slider(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.stim_pulse_width_slider()')
        pulse_width = int(self.ids['stim_pulse_width_slider'].value)
        gui_logger.slider(f'STIM_PULSE_WIDTH_{self.layer}', pulse_width)
        try:
            settings[self.layer]['stim_config']['pulse_width'] = pulse_width
        except Exception as e:
            logger.error(f'[LVP Main  ] LayerControl.stim_pulse_width_slider() -> {e}')
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
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.stim_ill_slider()')
        illumination = round(self.ids['stim_ill_slider'].value)
        gui_logger.slider(f'STIM_ILL_{self.layer}', illumination)
        try:
            settings[self.layer]['stim_config']['illumination'] = illumination
        except Exception as e:
            logger.error(f'[LVP Main  ] LayerControl.stim_ill_slider() -> {e}')
        new_text = str(illumination)
        if self.ids['stim_ill_text'].text != new_text:
            self.ids['stim_ill_text'].text = new_text
        self.apply_settings()

    def stim_ill_text(self):
        logger.info('[LVP Main  ] LayerControl.stim_ill_text()')
        if self._validate_and_apply_text_input(
            'stim_ill_text',
            'stim_ill_slider',
            'illumination',
            cast=int,
            settings_path='stim_config.illumination',
        ):
            self.apply_settings()

    def false_color(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.false_color()')
        enabled = bool(self.ids['false_color'].active)
        gui_logger.toggle(f'FALSE_COLOR_{self.layer}', enabled)
        settings[self.layer]['false_color'] = enabled
        self.apply_settings()

    def init_acquire(self):
        settings = _app_ctx.ctx.settings
        if settings[self.layer]['acquire'] == 'image':
            self.ids['acquire_image'].state = 'down'
        elif settings[self.layer]['acquire'] == 'video':
            self.ids['acquire_video'].state = 'down'
        else:
            self.ids['acquire_none'].state = 'down'

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

        if mode == 'image':
            settings[self.layer]['acquire'] = 'image'
            if 'stim_config' in settings[self.layer]:
                settings[self.layer]['stim_config']['enabled'] = False
            self.ids['stim_disable_btn'].active = True
            self.show_stim_controls = False

        elif mode == 'video':
            settings[self.layer]['acquire'] = 'video'
            if 'stim_config' in settings[self.layer]:
                settings[self.layer]['stim_config']['enabled'] = False
                self.ids['stim_disable_btn'].active = True
            self.ids['stim_disable_btn'].active = True
            self.show_stim_controls = False
        else:
            settings[self.layer]['acquire'] = None

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

    def init_autofocus(self):
        settings = _app_ctx.ctx.settings
        if not settings[self.layer]['autofocus']:
            self.ids['autofocus'].state = 'normal'
        else:
            self.ids['autofocus'].state = 'down'

    def update_autofocus(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.update_autofocus()')
        enabled = bool(self.ids['autofocus'].active)
        gui_logger.toggle(f'AUTOFOCUS_ENABLED_{self.layer}', enabled)
        settings[self.layer]['autofocus'] = enabled

    def save_focus(self):
        gui_logger.button(f'SAVE_FOCUS_{self.layer}')
        io_executor = _app_ctx.ctx.io_executor
        logger.info('[LVP Main  ] LayerControl.save_focus()')
        io_executor.put(IOTask(action=self.execute_save_focus))

    def execute_save_focus(self):
        # Stage 3.5+ pattern: hardware-touching executor actions wrap their
        # body in try/except, log the full error to lumaviewpro.log (per
        # the "all info in the production log" rule), and post a friendly
        # user-facing notification. The exception itself is NOT re-raised
        # because we're inside an executor task -- re-raising would just
        # log the same error twice (once here, once via the executor's
        # default handler). See `feedback_logging_policy.md` and
        # `project_lumaviewclassic_repo.md` in auto-memory.
        ctx = _app_ctx.ctx
        settings = ctx.settings
        try:
            old_focus = settings[self.layer].get('focus')
            pos = ctx.scope.motion.get_current_position('Z')
            settings[self.layer]['focus'] = pos
            # Propagate the new saved-focus value to in-memory protocol
            # steps that sit at the previous baseline. Per-well-tuned
            # steps (Z != old_focus) are preserved untouched. The
            # propagation is what makes the saved focus reach a fresh
            # New click without the user having to also tune every well.
            protocol = getattr(ctx, 'protocol', None)
            if protocol is not None:
                updated = protocol.update_layer_focus(layer=self.layer, old_z=old_focus, new_z=pos)
                if updated > 0:
                    logger.info(
                        f'[LVP Main  ] save_focus: propagated layer={self.layer} '
                        f'Z={old_focus} -> Z={pos} to {updated} step(s)'
                    )

                    # Refresh the stage labware view + the steps table so
                    # the updated Z values are visible immediately.
                    def _refresh(_dt):
                        try:
                            ctx.stage.set_protocol_steps(df=protocol.steps())
                            # Steps that tracked the old baseline now hold the
                            # new Z; refresh the step editor so its per-step
                            # focus readout reflects the propagated value.
                            ctx.motion_settings.ids['protocol_settings_id'].update_step_ui()
                        except Exception:
                            # Scheduled main-thread callback: the steps table
                            # can be mid-rebuild on this tick. Log so a stale-Z
                            # labware view / step editor is diagnosable instead
                            # of failing silently.
                            logger.exception(
                                '[LVP Main  ] save_focus: stage / step-editor '
                                f'refresh failed for layer {self.layer} after '
                                'focus propagation; Z readouts may show stale '
                                'values until the next UI update'
                            )

                    Clock.schedule_once(_refresh, 0)
        except Exception as e:
            logger.exception(f'[LVP Main  ] save_focus failed for layer {self.layer}: {e}')
            try:
                from modules.notification_center import notifications

                notifications.error(
                    'Motion',
                    'Save focus failed',
                    f"Couldn't read Z position: {e}",
                )
            except Exception:
                pass

    def goto_focus(self):
        gui_logger.button(f'GOTO_FOCUS_{self.layer}')
        io_executor = _app_ctx.ctx.io_executor
        logger.info('[LVP Main  ] LayerControl.goto_focus()')
        io_executor.put(
            IOTask(
                action=self.execute_goto_focus,
            )
        )

    def execute_goto_focus(self):
        # See execute_save_focus comment for the pattern rationale.
        from ui.ui_helpers import move_absolute_position

        settings = _app_ctx.ctx.settings
        try:
            pos = settings[self.layer]['focus']
            move_absolute_position('Z', pos)  # set current z height in usteps
        except KeyError:
            logger.warning(f'[LVP Main  ] goto_focus: no saved focus for layer {self.layer}')
            try:
                from modules.notification_center import notifications

                notifications.warning(
                    'Motion',
                    'No saved focus',
                    f"Layer '{self.layer}' has no saved focus position. Use SAVE first.",
                )
            except Exception:
                pass
        except Exception as e:
            logger.exception(f'[LVP Main  ] goto_focus failed for layer {self.layer}: {e}')
            try:
                from modules.notification_center import notifications

                notifications.error(
                    'Motion',
                    'Focus move failed',
                    f"Couldn't move Z to saved focus: {e}",
                )
            except Exception:
                pass

    _suppressing_led_log = False  # Class-level flag to prevent duplicate logging

    def update_led_state(self, apply_settings=True):
        ctx = _app_ctx.ctx
        # While autofocus owns the LED, a live UI apply -- such as the
        # exposure field losing focus when the AF button is clicked -- must
        # not turn off the channel autofocus is using, or AF scans a dark
        # frame. Logged so a bench run can confirm the suppression fired.
        if ctx.scope.imaging.is_focusing:
            logger.debug(
                '[LVP Main  ] update_led_state suppressed -- autofocus owns '
                f'the LED (layer={self.layer})'
            )
            return
        # Skip hardware commands during programmatic state changes
        # (e.g., disable_leds_for_other_layers toggling buttons).
        if LayerControl._suppressing_led_log or self._initializing:
            return
        settings = ctx.settings
        camera_executor = ctx.camera_executor
        enabled = self.ids['enable_led_btn'].state == 'down'
        gui_logger.toggle(f'LED_{self.layer}', enabled)
        illumination = settings[self.layer]['ill_ma']

        if apply_settings:
            self.apply_settings(update_led=False)

        camera_executor.put(
            IOTask(
                action=self.set_led_state, kwargs={'enabled': enabled, 'illumination': illumination}
            )
        )
        # self.set_led_state(enabled=enabled, illumination=illumination)

        # self.apply_settings()

    def set_led_state(self, enabled: bool, illumination: float):
        # Hardware-touching action. See execute_save_focus for the
        # try/except + log + notify pattern rationale.
        ctx = _app_ctx.ctx
        try:
            channel = ctx.scope.illumination.color2ch(self.layer)
            if not enabled:
                ctx.scope.illumination.led_off_async(channel)
            else:
                logger.info(
                    f'[LVP Main  ] lumaview.scope.illumination.led_on('
                    f'lumaview.scope.illumination.color2ch({self.layer}), {illumination})'
                )
                ctx.scope.illumination.led_on_async(channel, illumination)
        except Exception as e:
            logger.exception(
                f'[LVP Main  ] set_led_state failed for layer '
                f'{self.layer} (enabled={enabled}, illumination={illumination}): {e}'
            )
            try:
                from modules.notification_center import notifications

                notifications.error(
                    'LED',
                    f'{self.layer} LED command failed',
                    f"Couldn't {'enable' if enabled else 'disable'} the {self.layer} channel: {e}",
                )
            except Exception:
                pass

    # update_led_toggle_ui() removed -- LED observer handles UI sync.
    # See Phase 1 commit 96defe3.

    def set_step_state(self, step: dict):
        """Update widgets to reflect a protocol step.

        Only updates widgets for keys that are present in *step*.
        This allows partial updates (e.g. stim-config-only for non-current
        layers) without clobbering unrelated widget values.

        Suppresses event handlers via ``_initializing`` to prevent
        redundant hardware commands during the batch update.

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
                ill = step['Illumination']
                self.ids['ill_text'].text = str(ill)
                self.ids['ill_slider'].value = float(ill)

            if 'Gain' in step:
                self.ids['gain_text'].text = str(step['Gain'])
                self.ids['gain_slider'].value = float(step['Gain'])

            if 'Auto_Gain' in step:
                self.ids['auto_gain'].active = step['Auto_Gain']

            if 'Exposure' in step:
                self.ids['exp_text'].text = str(step['Exposure'])
                self.ids['exp_slider'].value = float(step['Exposure'])

            if 'Sum' in step:
                self.ids['sum_text'].text = str(step['Sum'])
                self.ids['sum_slider'].value = int(step['Sum'])

            # Video config
            vc = step.get('Video Config')
            if isinstance(vc, dict):
                import copy

                ctx = _app_ctx.ctx
                with ctx.settings_lock:
                    ctx.settings[self.layer]['video_config'] = copy.deepcopy(vc)
                if 'duration' in vc:
                    self.ids['video_duration_text'].text = str(vc['duration'])
                    self.ids['video_duration_slider'].value = float(vc['duration'])

            # Stim config (only for this layer's stim settings)
            sc = step.get('Stim_Config')
            if isinstance(sc, dict) and self.layer in sc:
                import copy

                stim = sc[self.layer]
                ctx = _app_ctx.ctx
                with ctx.settings_lock:
                    ctx.settings[self.layer]['stim_config'] = copy.deepcopy(stim)
                if stim.get('enabled', False):
                    self.ids['stim_enable_btn'].active = True
                    self.ids['stim_disable_btn'].active = False
                else:
                    self.ids['stim_disable_btn'].active = True
                    self.ids['stim_enable_btn'].active = False
                self.update_stim_controls_visibility()
                self.ids['stim_ill_text'].text = str(stim.get('illumination', 100))
                self.ids['stim_ill_slider'].value = float(stim.get('illumination', 100))
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

    def sync_camera_widgets_from_settings(self):
        """Re-point the exposure / gain / illumination widgets at the
        committed settings values.

        An uncommitted text edit (typed, no Enter) survives in the widget
        while autofocus or a protocol restores the camera from settings --
        leaving widget, settings, and hardware three-way divergent (the
        slider says 40 while the camera runs 10). Restore paths call this
        so the widgets tell the truth again; the uncommitted edit is
        deliberately dropped.
        """
        settings = _app_ctx.ctx.settings
        layer_settings = settings.get(self.layer, {})
        try:
            if 'exp_ms' in layer_settings:
                exp = float(layer_settings['exp_ms'])
                self.ids['exp_text'].text = str(round(exp, 2))
                self.ids['exp_slider'].value = exp
            if 'gain_db' in layer_settings:
                gain = float(layer_settings['gain_db'])
                self.ids['gain_text'].text = str(round(gain, 1))
                self.ids['gain_slider'].value = gain
            if 'ill_ma' in layer_settings:
                ill = float(layer_settings['ill_ma'])
                self.ids['ill_text'].text = str(ill)
                self.ids['ill_slider'].value = ill
        except Exception as e:
            logger.warning(f'[LVP Main  ] {self.layer} widget sync from settings failed: {e}')

    def apply_settings(self, ignore_auto_gain=False, update_led=True, protocol=False):

        # Skip apply_settings if layer is still initializing
        if getattr(self, '_initializing', False):
            return

        logger.debug(f'[LVP Main  ] {self.layer}_LayerControl.apply_settings()')

        ctx = _app_ctx.ctx

        # While autofocus owns the camera, a live UI apply -- e.g. the
        # exposure field losing focus because the AF button itself was
        # clicked -- must not push values to the camera mid-scan. The LED
        # leaf (update_led_state) carries the same guard; gating the shared
        # funnel covers every input that routes through here (exposure /
        # gain / illumination text and sliders, stim fields). Programmatic
        # protocol applies are exempt: the runner coordinates with AF
        # itself.
        if not protocol and ctx.scope.imaging.is_focusing:
            logger.debug(
                f'[LVP Main  ] {self.layer}_LayerControl.apply_settings '
                'suppressed -- autofocus owns the camera'
            )
            return

        settings = ctx.settings
        protocol_running_global = ctx.protocol_running
        camera_executor = ctx.camera_executor
        from ui.image_settings import set_histogram_layer

        lumaview = ctx.lumaview

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
                if not protocol_running_global.is_set():
                    for layer in common_utils.get_layers():
                        if layer == self.layer:
                            continue
                        try:
                            state = ctx.scope.illumination.get_led_state(color=layer)
                            if state.get('enabled', False):
                                ctx.scope.illumination.led_off_async(layer)
                        except Exception as e:
                            # Defensive: if get_led_state fails for any
                            # layer (e.g. null driver, hardware fault),
                            # don't block the rest of apply_settings.
                            # Log so the failure is visible in the
                            # production log per the "all info in the
                            # log" rule (was previously silent pass).
                            logger.warning(
                                f'[LVP Main  ] get_led_state({layer}) '
                                f'failed during disable_leds_for_other_layers: {e}'
                            )
                # Update button states (visual only -- hardware already handled)
                LayerControl._suppressing_led_log = True
                try:
                    for layer in common_utils.get_layers():
                        if layer != self.layer:
                            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
                            btn = layer_obj.ids['enable_led_btn']
                            if btn.state != 'normal':
                                btn.state = 'normal'
                finally:
                    LayerControl._suppressing_led_log = False

        if protocol_running_global.is_set():
            # Protocol actively running -- capture() handles camera settings
            # per-step. Don't apply here to avoid duplicate commands (#587/#588).
            logger.debug(
                f'[APPLY_SETTINGS DIAG] {self.layer} -- early return '
                f'(protocol running). Camera settings NOT applied.'
            )
            Clock.schedule_once(disable_leds_for_other_layers, 0)
            Clock.schedule_once(update_shader, 0)
            return
        if protocol and not settings.get('protocol_led_on', False):
            # Protocol preview mode with LEDs OFF -- no need to apply camera
            # settings since there's nothing to display.
            logger.debug(
                f'[APPLY_SETTINGS DIAG] {self.layer} -- early return '
                f'(protocol preview, LEDs off). Camera settings NOT applied.'
            )
            Clock.schedule_once(disable_leds_for_other_layers, 0)
            Clock.schedule_once(update_shader, 0)
            return
        # All other cases: apply camera settings normally.
        # This includes protocol preview with LEDs ON (#613) -- user needs
        # correct gain/exposure to see the step's channel properly.

        # global gain_vals

        # update illumination to currently selected settings
        # -----------------------------------------------------
        if not protocol:
            set_histogram_layer(active_layer=self.layer)

        # Queue IO task and update UI after completing IO
        if update_led and not protocol_running_global.is_set():
            self.update_led_state(apply_settings=False)

        disable_leds_for_other_layers()

        # update exposure to currently selected settings
        # -----------------------------------------------------

        exposure = settings[self.layer]['exp_ms']
        gain = settings[self.layer]['gain_db']

        if not protocol_running_global.is_set():
            auto_gain_enabled = settings[self.layer]['auto_gain']
            # Sync the toggle CheckBox + dependent slider-disabled state to
            # the settings value before applying to the camera. The .kv has
            # no Kivy binding from settings to auto_gain.active, so when the
            # JSON loads at startup with auto_gain=True the CheckBox stays
            # at its default False; apply_settings would then send AG=True
            # to the camera while the toggle continues to read OFF in the
            # UI. Programmatic .active = bool fires no on_release (CheckBox
            # only binds on_release in the .kv), so this does not re-enter.
            self.ids['auto_gain'].active = auto_gain_enabled
            for slider_item in ('gain_slider', 'gain_text', 'exp_slider', 'exp_text'):
                self.ids[slider_item].disabled = auto_gain_enabled
            autogain_settings = None
            if not ignore_auto_gain:
                from modules.config_ui_getters import (
                    get_ag_ae_max_exposure_ms,
                    get_auto_gain_settings,
                )

                autogain_settings = get_auto_gain_settings()
                # Cap how far AG/AE may drive exposure for this layer's
                # channel class (issue #655): without it AG runs exposure
                # to the sensor max on dim scenes, washing out brightfield
                # and making the live auto loop hunt.
                autogain_settings['max_exposure_ms'] = get_ag_ae_max_exposure_ms(self.layer)
            camera_executor.put(
                IOTask(
                    action=lumaview.scope.imaging.apply_layer_camera_settings,
                    kwargs={
                        'gain_db': gain,
                        'exposure_ms': exposure,
                        'auto_gain': auto_gain_enabled,
                        'auto_gain_settings': autogain_settings,
                    },
                )
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
