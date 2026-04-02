# Copyright Etaluma, Inc.
import logging

import numpy as np

from kivy.clock import Clock
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules import gui_logger
import modules.scope_commands as scope_commands
from modules.sequential_io_executor import IOTask

logger = logging.getLogger('LVP.ui.layer_control')

# Brightfield allows higher illumination/exposure than fluorescence channels
# because BF LED power is lower and longer exposures don't risk photobleaching.
BF_MAX_ILLUMINATION = 500
BF_MAX_EXPOSURE_MS = 1000
FLUORESCENCE_MIN_EXPOSURE_MS = 1.0
SLIDER_DEBOUNCE_S = 0.1
INIT_MAX_RETRIES = 50


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
        super(LayerControl, self).__init__(**kwargs)

        logger.debug('[LVP Main  ] LayerControl.__init__()')
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)

        # Flag to prevent apply_settings during initialization
        self._initializing = True

        self.apply_gain_slider = Clock.create_trigger(lambda dt: self.apply_settings(), SLIDER_DEBOUNCE_S)
        self.apply_exp_slider = Clock.create_trigger(lambda dt: self.apply_settings(), SLIDER_DEBOUNCE_S)
        self.apply_ill_slider = Clock.create_trigger(lambda dt: self.apply_settings(), SLIDER_DEBOUNCE_S)
        self._init_ui_retries = 0
        Clock.schedule_once(self._init_ui, 0)


    def _validate_and_apply_text_input(
        self, text_id: str, slider_id: str, settings_key: str,
        cast=float, settings_path: str | None = None,
        gui_log_name: str | None = None,
    ) -> bool:
        """Shared validation for text input → slider → settings update.

        Parses text, clips to slider range, updates slider + text + settings,
        and applies. Returns True on success, False on invalid input.

        Args:
            text_id: Kivy widget id for the text input (e.g., 'gain_text')
            slider_id: Kivy widget id for the slider (e.g., 'gain_slider')
            settings_key: Key in settings[self.layer] (e.g., 'gain')
            cast: Type to cast the text value (float or int)
            settings_path: Dot-separated sub-path for nested settings
                          (e.g., 'video_config.duration' or 'stim_config.frequency')
            gui_log_name: Name for gui_logger.slider() call (e.g., 'GAIN')
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
            self.ids[text_id].text = str(val)
            return False

        clipped = cast(np.clip(raw, slider.min, slider.max))

        # Update settings
        if settings_path:
            parts = settings_path.split('.')
            target = settings[self.layer]
            for p in parts[:-1]:
                target = target[p]
            target[parts[-1]] = clipped
        else:
            settings[self.layer][settings_key] = clipped

        # Update widgets
        slider.value = float(clipped) if cast == float else int(clipped)
        self.ids[text_id].text = str(clipped)

        if gui_log_name:
            gui_logger.slider(f'{gui_log_name}_{self.layer}', clipped)

        return True

    def _init_ui(self, dt=0):
        ctx = _app_ctx.ctx
        if ctx is None:
            self._init_ui_retries += 1
            if self._init_ui_retries > INIT_MAX_RETRIES:
                logger.error('[LVP Main  ] LayerControl._init_ui: ctx still None after 50 retries, giving up')
                return
            Clock.schedule_once(self._init_ui, 0.1)
            return
        settings = ctx.settings

        if self.layer in ['Red', 'Green', 'Blue'] and settings['stimulation_enabled']:
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
        if not self._initializing:
            logger.info('[LVP Main  ] LayerControl.ill_slider()')
        illumination = round(self.ids['ill_slider'].value)  # Round to integer (step=1)
        if not self._initializing:
            gui_logger.slider(f'ILLUMINATION_{self.layer}', illumination)
        settings[self.layer]['ill'] = illumination

        if 'stim_config' in settings[self.layer]:
            settings[self.layer]['stim_config']['illumination'] = illumination

        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(illumination)
        if self.ids['ill_text'].text != new_text:
            self.ids['ill_text'].text = new_text
        if not self._initializing:
            self.apply_ill_slider()


    def ill_text(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.ill_text()')
        ill_min = self.ids['ill_slider'].min
        if self.layer == "BF":
            ill_max = BF_MAX_ILLUMINATION
        else:
            ill_max = self.ids['ill_slider'].max
        try:
            ill_val = float(self.ids['ill_text'].text)
        except Exception:
            logger.debug(f'[LVP Main  ] Invalid illumination input: {self.ids["ill_text"].text!r}')
            # Show current valid value so user knows input was rejected (M21)
            self.ids['ill_text'].text = str(settings[self.layer]['ill'])
            return

        illumination = float(np.clip(ill_val, ill_min, ill_max))

        settings[self.layer]['ill'] = illumination
        self.ids['ill_slider'].value = float(np.clip(illumination, ill_min, self.ids['ill_slider'].max))
        self.ids['ill_text'].text = str(illumination)

        if 'stim_config' in settings[self.layer]:
            settings[self.layer]['stim_config']['illumination'] = illumination

        self.apply_settings()


    def sum_slider(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.sum_slider()')
        total = int(self.ids['sum_slider'].value)
        gui_logger.slider(f'SUM_{self.layer}', total)
        settings[self.layer]['sum'] = total
        self.apply_settings()


    def sum_text(self):
        logger.info('[LVP Main  ] LayerControl.sum_text()')
        if self._validate_and_apply_text_input('sum_text', 'sum_slider', 'sum', cast=int):
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
            'video_duration_text', 'video_duration_slider', 'duration',
            cast=int, settings_path='video_config.duration',
        ):
            self.apply_settings()

    def update_auto_gain(self, init: bool = False):
        settings = _app_ctx.ctx.settings
        camera_executor = _app_ctx.ctx.camera_executor
        logger.info('[LVP Main  ] LayerControl.update_auto_gain()')
        if self.ids['auto_gain'].state == 'down':
            state = True
        else:
            state = False

        for item in ('gain_slider', 'gain_text', 'exp_slider', 'exp_text'):
            self.ids[item].disabled = state

        # When transitioning out of auto-gain, keep last auto-gain settings to apply
        camera_executor.put(IOTask(
            action = LayerControl.get_gain_exposure,
            args=(self, init, state),
            callback=LayerControl.update_auto_gain_cb,
            cb_args=(self),
            pass_result=True
        ))

        # actual_gain = lumaview.scope.camera.get_gain()
        # actual_exp = lumaview.scope.camera.get_exposure_t()


    def get_gain_exposure(self, init, state):
        ctx = _app_ctx.ctx
        # Read directly from camera hardware, not cache.
        # During auto-gain, the SDK adjusts gain/exposure but doesn't
        # update the cache — cache still has the pre-auto-gain values.
        actual_gain = ctx.scope.get_gain()
        actual_exp = ctx.scope.get_exposure_time()

        return (init, state, actual_gain, actual_exp)

    def update_auto_gain_cb(self, result=None, exception=None):
        settings = _app_ctx.ctx.settings
        try:

            if exception is not None:
                logger.error(f"LVP Main] Update_auto_gain error: {exception}")
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
                # Clamp exposure to slider range. Auto-gain can drive exposure
                # to sub-millisecond values (e.g., 0.1ms on a bright field) which
                # produces nearly black fluorescence images if the user then
                # creates protocol steps from these settings. Floor at 1ms for
                # fluorescence channels where sub-ms is never realistic.
                exp_min = self.ids['exp_slider'].min
                exp_max = self.ids['exp_slider'].max
                if self.layer in ('Red', 'Green', 'Blue', 'Lumi'):
                    exp_min = max(exp_min, FLUORESCENCE_MIN_EXPOSURE_MS)
                exp = float(np.clip(exp, exp_min, exp_max))

                settings[self.layer]['gain'] = gain
                settings[self.layer]['exp'] = exp
                # Update sliders/text to show the auto-adjusted values
                self.ids['gain_slider'].value = gain
                self.ids['gain_text'].text = str(round(gain, 1))
                self.ids['exp_slider'].value = exp
                self.ids['exp_text'].text = str(round(exp, 2))

            settings[self.layer]['auto_gain'] = state
            self.apply_settings()

        except Exception as e:
            logger.error(f"LVP Main] Update_auto_gain error: {e}")
            return

    def gain_slider(self):
        settings = _app_ctx.ctx.settings
        protocol_running_global = _app_ctx.ctx.protocol_running
        if protocol_running_global.is_set():
            return
        if not self._initializing:
            logger.info('[LVP Main  ] LayerControl.gain_slider()')
        gain = round(self.ids['gain_slider'].value, 1)  # Round to 1 decimal (step=0.1)
        if not self._initializing:
            gui_logger.slider(f'GAIN_{self.layer}', gain)
        settings[self.layer]['gain'] = gain
        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(gain)
        if self.ids['gain_text'].text != new_text:
            self.ids['gain_text'].text = new_text
        if not self.ids['gain_slider'].disabled and not self._initializing:
            self.apply_gain_slider()
        ####

    def gain_text(self):
        logger.info('[LVP Main  ] LayerControl.gain_text()')
        if self._validate_and_apply_text_input('gain_text', 'gain_slider', 'gain'):
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
            'composite_threshold_text', 'composite_threshold_slider',
            'composite_brightness_threshold',
        )

    def exp_slider(self):
        settings = _app_ctx.ctx.settings
        protocol_running_global = _app_ctx.ctx.protocol_running
        if protocol_running_global.is_set():
            return
        if not self._initializing:
            logger.info('[LVP Main  ] LayerControl.exp_slider()')
        exposure = round(self.ids['exp_slider'].value, 2)  # Round to 2 decimals (step=0.01)
        if not self._initializing:
            gui_logger.slider(f'EXPOSURE_{self.layer}', exposure)
        # exposure = 10 ** self.ids['exp_slider'].value # slider is log_10(ms)
        settings[self.layer]['exp'] = exposure        # exposure in ms
        # Update text only if changed to reduce ScrollView recalculations
        new_text = str(exposure)
        if self.ids['exp_text'].text != new_text:
            self.ids['exp_text'].text = new_text
        if not self.ids['exp_slider'].disabled and not self._initializing:
            self.apply_exp_slider()

    def exp_text(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.exp_text()')
        exp_min = self.ids['exp_slider'].min
        #exp_max = self.ids['exp_slider'].max
        if self.layer == "BF":
            exp_max = BF_MAX_EXPOSURE_MS
        else:
            exp_max = self.ids['exp_slider'].max

        try:
            exp_val = float(self.ids['exp_text'].text)
        except Exception:
            logger.debug(f'[LVP Main  ] Invalid exposure input: {self.ids["exp_text"].text!r}')
            # Show current valid value so user knows input was rejected (M21)
            self.ids['exp_text'].text = str(settings[self.layer]['exp'])
            return

        exposure = float(np.clip(exp_val, exp_min, exp_max))

        settings[self.layer]['exp'] = exposure
        self.ids['exp_slider'].value = float(np.clip(exposure, exp_min, self.ids['exp_slider'].max))
        # self.ids['exp_slider'].value = float(np.log10(exposure)) # convert slider to log_10
        self.ids['exp_text'].text = str(exposure)

        self.apply_exp_slider()

    def stim_freq_slider(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.stim_freq_slider()')
        frequency = self.ids['stim_freq_slider'].value
        gui_logger.slider(f'STIM_FREQ_{self.layer}', frequency)
        try:
            settings[self.layer]['stim_config']['frequency'] = frequency
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_freq_slider() -> {e}")
        self.apply_settings()

    def stim_pulse_count_slider(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.stim_pulse_count_slider()')
        pulse_count = self.ids['stim_pulse_count_slider'].value
        gui_logger.slider(f'STIM_PULSE_COUNT_{self.layer}', pulse_count)
        try:
            settings[self.layer]['stim_config']['pulse_count'] = pulse_count
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_pulse_count_slider() -> {e}")
        self.apply_settings()

    def stim_pulse_width_slider(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.stim_pulse_width_slider()')
        pulse_width = self.ids['stim_pulse_width_slider'].value
        gui_logger.slider(f'STIM_PULSE_WIDTH_{self.layer}', pulse_width)
        try:
            settings[self.layer]['stim_config']['pulse_width'] = pulse_width
        except Exception as e:
            logger.error(f"[LVP Main  ] LayerControl.stim_pulse_width_slider() -> {e}")
        self.apply_settings()

    def stim_freq_text(self):
        logger.info('[LVP Main  ] LayerControl.stim_freq_text()')
        if self._validate_and_apply_text_input(
            'stim_freq_text', 'stim_freq_slider', 'frequency',
            settings_path='stim_config.frequency',
        ):
            self.apply_settings()

    def stim_pulse_count_text(self):
        logger.info('[LVP Main  ] LayerControl.stim_pulse_count_text()')
        if self._validate_and_apply_text_input(
            'stim_pulse_count_text', 'stim_pulse_count_slider', 'pulse_count',
            cast=int, settings_path='stim_config.pulse_count',
        ):
            self.apply_settings()

    def stim_pulse_width_text(self):
        logger.info('[LVP Main  ] LayerControl.stim_pulse_width_text()')
        if self._validate_and_apply_text_input(
            'stim_pulse_width_text', 'stim_pulse_width_slider', 'pulse_width',
            cast=int, settings_path='stim_config.pulse_width',
        ):
            self.apply_settings()

    def false_color(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.false_color()')
        settings[self.layer]['false_color'] = self.ids['false_color'].active
        self.apply_settings()

    def init_acquire(self):
        settings = _app_ctx.ctx.settings
        if settings[self.layer]['acquire'] == "image":
            self.ids['acquire_image'].state = 'down'
        elif settings[self.layer]['acquire'] == 'video':
            self.ids['acquire_video'].state = 'down'
        else:
            self.ids['acquire_none'].state = 'down'

    def update_acquire(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.update_acquire()')

        if self.ids['acquire_image'].active:
            settings[self.layer]['acquire'] = "image"
            if "stim_config" in settings[self.layer]:
                settings[self.layer]['stim_config']['enabled'] = False
            self.ids['stim_disable_btn'].active = True
            self.show_stim_controls = False

        elif self.ids['acquire_video'].active:
            settings[self.layer]['acquire'] = "video"
            if "stim_config" in settings[self.layer]:
                settings[self.layer]['stim_config']['enabled'] = False
                self.ids['stim_disable_btn'].active = True
            self.ids['stim_disable_btn'].active = True
            self.show_stim_controls = False
        else:
            settings[self.layer]['acquire'] = None

        if "stim_config" in settings[self.layer]:
            self.update_stim_controls_visibility()

    def update_stim_enable(self):
        settings = _app_ctx.ctx.settings
        logger.info('[LVP Main  ] LayerControl.update_stim_enable()')
        enabled = self.ids['stim_enable_btn'].active
        gui_logger.toggle(f'STIM_{self.layer}', enabled)
        if self.ids['stim_enable_btn'].active:
            if "stim_config" in settings[self.layer]:
                if settings[self.layer]['stim_config'] is not None:
                    settings[self.layer]['stim_config']['enabled'] = True
            settings[self.layer]['acquire'] = None
            self.ids['acquire_none'].active = True
            self.ids['acquire_none'].state = 'down'
        else:
            if "stim_config" in settings[self.layer]:
                if settings[self.layer]['stim_config'] is not None:
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
        settings[self.layer]['autofocus'] = self.ids['autofocus'].active

    def save_focus(self):
        gui_logger.button(f'SAVE_FOCUS_{self.layer}')
        io_executor = _app_ctx.ctx.io_executor
        logger.info('[LVP Main  ] LayerControl.save_focus()')
        io_executor.put(IOTask(
            action=self.execute_save_focus
        ))

    def execute_save_focus(self):
        ctx = _app_ctx.ctx
        settings = ctx.settings
        pos = ctx.scope.get_current_position('Z')
        settings[self.layer]['focus'] = pos


    def goto_focus(self):
        gui_logger.button(f'GOTO_FOCUS_{self.layer}')
        io_executor = _app_ctx.ctx.io_executor
        logger.info('[LVP Main  ] LayerControl.goto_focus()')
        io_executor.put(IOTask(
            action=self.execute_goto_focus,
        ))

    def execute_goto_focus(self):
        from ui.ui_helpers import move_absolute_position
        settings = _app_ctx.ctx.settings
        pos = settings[self.layer]['focus']
        move_absolute_position('Z', pos)  # set current z height in usteps

    _suppressing_led_log = False  # Class-level flag to prevent duplicate logging

    def update_led_state(self, apply_settings=True):
        # Skip hardware commands during programmatic state changes
        # (e.g., disable_leds_for_other_layers toggling buttons)
        if LayerControl._suppressing_led_log or self._initializing:
            return
        settings = _app_ctx.ctx.settings
        camera_executor = _app_ctx.ctx.camera_executor
        enabled = True if self.ids['enable_led_btn'].state == 'down' else False
        gui_logger.toggle(f'LED_{self.layer}', enabled)
        illumination = settings[self.layer]['ill']

        if apply_settings:
            self.apply_settings(update_led=False)

        camera_executor.put(IOTask(
            action=self.set_led_state,
            kwargs= {
                "enabled": enabled,
                "illumination": illumination
            }
        ))
        #self.set_led_state(enabled=enabled, illumination=illumination)

        # self.apply_settings()


    def set_led_state(self, enabled: bool, illumination: float):
        ctx = _app_ctx.ctx
        io_executor = ctx.io_executor
        channel = ctx.scope.color2ch(self.layer)
        if not enabled:
            scope_commands.led_off(ctx.scope, io_executor, channel)
        else:
            logger.info(f'[LVP Main  ] lumaview.scope.led_on(lumaview.scope.color2ch({self.layer}), {illumination})')
            scope_commands.led_on(ctx.scope, io_executor, channel, illumination)

    def update_led_toggle_ui(self):
        ctx = _app_ctx.ctx
        if ctx.scope.led_connected:
            led_state = ctx.scope.get_led_state(color=self.layer)
            LayerControl._suppressing_led_log = True
            try:
                if led_state['enabled']:
                    self.ids['enable_led_btn'].state = 'down'
                else:
                    self.ids['enable_led_btn'].state = 'normal'
            finally:
                LayerControl._suppressing_led_log = False


    def apply_settings(self, ignore_auto_gain=False, update_led=True, protocol=False):

        # Skip apply_settings if layer is still initializing
        if getattr(self, '_initializing', False):
            return

        logger.info(f'[LVP Main  ] {self.layer}_LayerControl.apply_settings()')

        ctx = _app_ctx.ctx
        settings = ctx.settings
        protocol_running_global = ctx.protocol_running
        camera_executor = ctx.camera_executor
        from ui.image_settings import set_histogram_layer
        lumaview = ctx.lumaview

        def update_shader(dt=None):
            if not ctx.scope_display.paused.is_set():
                if ctx.scope_display.use_bullseye is False:
                    self.update_shader(dt=0)

        def disable_leds_for_other_layers(dt=None):
            if self.ids['enable_led_btn'].state == 'down':
                LayerControl._suppressing_led_log = True
                try:
                    for layer in common_utils.get_layers():
                        if layer != self.layer:
                            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
                            btn = layer_obj.ids['enable_led_btn']
                            # Only set if changed — Kivy skips event dispatch
                            # if value is the same, avoiding cascade overhead
                            if btn.state != 'normal':
                                btn.state = 'normal'
                finally:
                    LayerControl._suppressing_led_log = False


        if protocol or protocol_running_global.is_set():
            Clock.schedule_once(disable_leds_for_other_layers, 0)
            Clock.schedule_once(update_shader, 0)
            return

        # global gain_vals

        # update illumination to currently selected settings
        # -----------------------------------------------------
        if not protocol:
            set_histogram_layer(active_layer=self.layer)


        # Queue IO task and update UI after completing IO
        if update_led:
            if not protocol_running_global.is_set():
                self.update_led_state(apply_settings=False)


        disable_leds_for_other_layers()


        # update exposure to currently selected settings
        # -----------------------------------------------------

        exposure = settings[self.layer]['exp']
        gain = settings[self.layer]['gain']

        if not protocol_running_global.is_set():
            auto_gain_enabled = settings[self.layer]['auto_gain']
            autogain_settings = None
            if not ignore_auto_gain:
                from modules.config_ui_getters import get_auto_gain_settings
                autogain_settings = get_auto_gain_settings()
            camera_executor.put(IOTask(
                action=lumaview.scope.apply_layer_camera_settings,
                kwargs={
                    'gain': gain,
                    'exposure_ms': exposure,
                    'auto_gain': auto_gain_enabled,
                    'auto_gain_settings': autogain_settings,
                }
            ))

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
