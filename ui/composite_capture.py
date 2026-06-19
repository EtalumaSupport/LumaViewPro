# Copyright Etaluma, Inc.
"""
CompositeCapture -- shared image capture capabilities extracted from lumaviewpro.py.

Provides live_capture() and composite_capture() methods inherited by MainDisplay.
"""

import datetime
import logging
import pathlib
import threading

import numpy as np

from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules import gui_logger
from modules.composite_builder import build_composite
from modules.image_save import save_image, save_live_image
import modules.image_utils as image_utils
from modules.sequential_io_executor import IOTask, PRIORITY_MED
from ui.ui_helpers import (
    live_histo_off,
    live_histo_reverse,
    set_last_save_folder,
)

logger = logging.getLogger('LVP.ui.composite_capture')


class CompositeCapture(FloatLayout):
    _capturing = threading.Event()  # Thread-safe guard against rapid double-clicks

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def live_capture(self):
        gui_logger.button('LIVE_CAPTURE')
        if CompositeCapture._capturing.is_set():
            logger.warning('[LVP Main  ] Capture already in progress, ignoring')
            return
        ctx = _app_ctx.ctx
        # Gate on camera-connected before enqueuing. Without this, a
        # rapidly-clicked button after camera disconnect queues many
        # tasks that each raise CaptureError and flood the error log;
        # the F17 sentinel migration made the failure loud where it
        # used to be a silent no-op, which is correct but exposes the
        # missing UI gate.
        if not getattr(ctx.scope, 'camera_connected', True):
            from modules.notification_center import notifications

            notifications.warning(
                'Camera',
                'Camera not connected',
                'Cannot capture -- camera is not connected. Check USB '
                'and reconnect, then try again.',
            )
            return
        CompositeCapture._capturing.set()
        ctx.camera_executor.put(
            IOTask(
                action=self._live_capture_impl, callback=lambda: CompositeCapture._capturing.clear()
            )
        )

    def _live_capture_impl(self):
        from modules.config_ui_getters import get_image_capture_config_from_ui, get_layer_configs

        logger.info('[LVP Main  ] CompositeCapture.live_capture()')

        ctx = _app_ctx.ctx
        settings = ctx.settings

        file_root = 'live_'
        color = 'BF'
        well_label = ctx.scope.runtime_state.get_well_label()

        image_capture_config = get_image_capture_config_from_ui()
        force_to_8bit_pixel_depth = image_capture_config['capture_depth'] == 8
        save_encoding = image_capture_config['save_encoding']

        for layer in common_utils.get_layers():
            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
            accordion_item_obj = ctx.image_settings.accordion_item_lookup(layer=layer)
            if not accordion_item_obj.collapse:
                append = f'{well_label}_{layer}'
                if layer_obj.ids['false_color'].active:
                    color = layer

                break

        save_folder = pathlib.Path(settings['live_folder']) / 'Manual'
        separate_folder_per_channel = ctx.motion_settings.ids[
            'microscope_settings_id'
        ]._seperate_folder_per_channel
        if separate_folder_per_channel:
            save_folder = save_folder / layer

        save_folder.mkdir(parents=True, exist_ok=True)
        set_last_save_folder(dir=save_folder)

        # Stage B1: update_scopedisplay retired -- the display thread
        # runs a continuous FPS-paced loop, so "kick the display" is
        # a no-op. Pass None to the sum_iteration loop; downstream
        # callers tolerate None (already a documented convention).
        sum_iteration_callback = None

        layer_configs = get_layer_configs(specific_layers=layer)
        sum_delay_s = layer_configs[layer]['exposure_ms'] / 1000
        sum_count = layer_configs[layer]['sum']

        if ctx.engineering_mode is False:
            return save_live_image(
                ctx.scope,
                save_folder,
                file_root,
                append,
                color,
                force_to_8bit=force_to_8bit_pixel_depth,
                output_format=settings['image_output_format']['live'],
                sum_count=sum_count,
                sum_delay_s=sum_delay_s,
                sum_iteration_callback=sum_iteration_callback,
                turn_off_all_leds_after=False,
                jpeg_quality=settings.get('jpg_quality', 90),
            )

        else:
            use_bullseye = ctx.scope_display.use_bullseye
            use_crosshairs = ctx.scope_display.use_crosshairs

            if not use_bullseye and not use_crosshairs:
                return save_live_image(
                    ctx.scope,
                    save_folder,
                    file_root,
                    append,
                    color,
                    force_to_8bit=force_to_8bit_pixel_depth,
                    output_format=settings['image_output_format']['live'],
                    sum_count=sum_count,
                    sum_delay_s=sum_delay_s,
                    sum_iteration_callback=sum_iteration_callback,
                    turn_off_all_leds_after=False,
                    jpeg_quality=settings.get('jpg_quality', 90),
                    save_encoding=save_encoding,
                )

            image_orig = ctx.scope.imaging.capture_and_wait(force_to_8bit=force_to_8bit_pixel_depth)
            if image_orig is None:
                return

            # Save both versions of the image (unaltered and overlayed)
            now = datetime.datetime.now()
            time_string = now.strftime('%Y%m%d_%H%M%S')
            append = f'{append}_{time_string}'

            # If not in 8-bit mode, generate an 8-bit copy of the image for
            # visualization. image_orig is a single native-depth capture here,
            # so its depth is the camera's payload depth (read at the capture
            # point so the downconvert scales against the real range).
            if not force_to_8bit_pixel_depth:
                image = image_utils.convert_to_8bit(image_orig, ctx.scope.imaging.significant_bits)
            else:
                image = image_orig

            # Original image may be in 8 or 12-bit
            save_image(
                ctx.scope,
                array=image_orig,
                save_folder=save_folder,
                file_root=file_root,
                append=append,
                color=color,
                tail_id_mode=None,
                output_format=settings['image_output_format']['live'],
                jpeg_quality=settings.get('jpg_quality', 90),
                save_encoding=save_encoding,
            )

            if use_bullseye:
                bullseye_image = ctx.scope_display.transform_to_bullseye(image)
            else:
                bullseye_image = image

            if use_crosshairs:
                crosshairs_image = ctx.scope_display.add_crosshairs(bullseye_image)
            else:
                crosshairs_image = bullseye_image

            # Overlay image is in 8-bits
            save_image(
                ctx.scope,
                array=crosshairs_image,
                save_folder=save_folder,
                file_root=file_root,
                append=f'{append}_overlay',
                color=color,
                tail_id_mode=None,
                output_format=settings['image_output_format']['live'],
                jpeg_quality=settings.get('jpg_quality', 90),
                save_encoding=save_encoding,
            )

    # capture and save a composite image using the current settings
    def composite_capture(self):
        gui_logger.button('COMPOSITE_CAPTURE')
        ctx = _app_ctx.ctx

        if CompositeCapture._capturing.is_set():
            logger.warning('[LVP Main  ] Composite capture already in progress, ignoring')
            return
        # Same camera-connected gate as live_capture -- see comment there.
        if not getattr(ctx.scope, 'camera_connected', True):
            from modules.notification_center import notifications

            notifications.warning(
                'Camera',
                'Camera not connected',
                'Cannot capture composite -- camera is not connected. '
                'Check USB and reconnect, then try again.',
            )
            return
        CompositeCapture._capturing.set()

        z_stage_present = not ctx.disable_homing

        logger.info('[LVP Main  ] CompositeCapture.composite_capture()')

        # Suspend video false coloring during composite capture.
        # The video recorder applies a single false color (set at recording start)
        # to ALL frames, but composite capture cycles through multiple channels.
        # Without this, every frame records as the initial channel's color.
        saved_video_false_color = getattr(self, 'video_false_color', None)
        self.video_false_color = None

        # Log per-channel settings for composite debugging
        settings = ctx.settings
        for layer in (
            *common_utils.get_transmitted_layers(),
            *common_utils.get_fluorescence_layers(),
        ):
            ls = settings.get(layer, {})
            if ls.get('acquire') == 'image':
                logger.info(
                    f'[COMPOSITE ] {layer}: gain={ls.get("gain_db")}, exp={ls.get("exp_ms")}ms, '
                    f'ill={ls.get("ill_ma")}mA, sum={ls.get("sum", 1)}, '
                    f'threshold={ls.get("composite_brightness_threshold", "?")}%'
                )

        initial_layer = common_utils.get_opened_layer(ctx.image_settings)

        if ctx.scope.illumination.get_led_state(initial_layer)['enabled']:
            led_restore_state = True
        else:
            led_restore_state = False

        live_histo_off()

        if not ctx.scope.imaging.camera_active:
            return

        # Resolve the image mode on the main thread (reads UI widgets) and pass
        # the derived facts into the worker, which runs off-thread and must not
        # touch Kivy widgets.
        from modules.config_ui_getters import get_image_capture_config_from_ui

        image_capture_config = get_image_capture_config_from_ui()
        capture_depth = image_capture_config['capture_depth']
        save_encoding = image_capture_config['save_encoding']

        # Run hardware-blocking work on worker_pool at MED priority so it
        # doesn't freeze the UI or contend with io_executor. HIGH-priority
        # abort/cleanup tasks still jump ahead. Composite capture is
        # bounded (~seconds) so it doesn't starve LOW background work.
        ctx.worker_pool.put(
            IOTask(
                action=self._composite_capture_worker,
                kwargs={
                    'z_stage_present': z_stage_present,
                    'initial_layer': initial_layer,
                    'led_restore_state': led_restore_state,
                    'capture_depth': capture_depth,
                    'save_encoding': save_encoding,
                    'saved_video_false_color': saved_video_false_color,
                },
                priority=PRIORITY_MED,
            )
        )

    def _composite_capture_worker(
        self,
        z_stage_present,
        initial_layer,
        led_restore_state,
        capture_depth,
        save_encoding,
        saved_video_false_color=None,
    ):
        """Runs on background thread -- performs hardware I/O without blocking UI."""
        try:
            self._composite_capture_worker_inner(
                z_stage_present=z_stage_present,
                initial_layer=initial_layer,
                led_restore_state=led_restore_state,
                capture_depth=capture_depth,
                save_encoding=save_encoding,
                saved_video_false_color=saved_video_false_color,
            )
        except Exception as ex:
            logger.error(f'[COMPOSITE] _composite_capture_worker failed: {ex}', exc_info=True)
            from modules.notification_center import notifications

            notifications.error('Composite', 'Composite Capture Failed', str(ex))
        finally:
            # Always clear _capturing so the button resets even on error.
            # Without this, a save_image failure leaves _capturing set and
            # all subsequent composite clicks are blocked. (#610 session)
            CompositeCapture._capturing.clear()
            self.video_false_color = saved_video_false_color

            def _restore_ui_on_error(dt):
                try:
                    ctx = _app_ctx.ctx
                    ctx.lumaview.ids['composite_btn'].state = 'normal'
                    live_histo_reverse()
                except Exception:
                    pass

            Clock.schedule_once(_restore_ui_on_error, 0)

    def _composite_capture_worker_inner(
        self,
        z_stage_present,
        initial_layer,
        led_restore_state,
        capture_depth,
        save_encoding,
        saved_video_false_color=None,
    ):
        """Inner worker -- actual composite capture logic."""
        ctx = _app_ctx.ctx
        settings = ctx.settings

        # Snapshot settings at entry for thread safety -- avoids seeing partial
        # updates from the UI thread during the capture sequence.
        all_layers = (
            *common_utils.get_transmitted_layers(),
            *common_utils.get_fluorescence_layers(),
            *common_utils.get_luminescence_layers(),
        )
        with ctx.settings_lock:
            layer_settings = {layer: dict(settings[layer]) for layer in all_layers}
            live_folder = settings['live_folder']
            image_output_format = dict(settings['image_output_format'])

        acquired_channel_count = 0
        most_recent_aq_channel = None

        if capture_depth == 12:
            dtype = np.uint16
            max_value = 4095
        else:
            dtype = np.uint8
            max_value = 255

        transmitted_image = None
        channel_images = {}
        brightness_thresholds = {}

        # Capture transmitted channel (BF/PC/DF) -- use first found as base
        for trans_layer in common_utils.get_transmitted_layers():
            if layer_settings[trans_layer]['acquire'] == 'image':
                acquired_channel_count += 1
                most_recent_aq_channel = trans_layer

                if z_stage_present:
                    focus_pos = layer_settings[trans_layer]['focus']
                    ctx.scope.motion.move_absolute_sync(
                        'Z',
                        focus_pos,
                        wait_until_complete=True,
                    )

                gain = layer_settings[trans_layer]['gain_db']
                ctx.scope.imaging.set_gain_sync(gain)
                exposure = layer_settings[trans_layer]['exp_ms']
                ctx.scope.imaging.set_exposure_sync(exposure)
                illumination = layer_settings[trans_layer]['ill_ma']

                ctx.scope.illumination.led_on_sync(
                    ctx.scope.illumination.color2ch(trans_layer),
                    illumination,
                )

                transmitted_image = np.array(
                    ctx.scope.imaging.capture_and_wait_sync(
                        force_to_8bit=capture_depth == 8,
                    ),
                    dtype=dtype,
                )
                ctx.scope.illumination.leds_off_sync()

                # Can only use one transmitted channel per composite
                break

        ctx.scope.illumination.leds_off_sync()

        # Capture fluorescence and luminescence channels
        for layer in (
            *common_utils.get_fluorescence_layers(),
            *common_utils.get_luminescence_layers(),
        ):
            if layer_settings[layer]['acquire'] == 'image':
                acquired_channel_count += 1
                most_recent_aq_channel = layer

                if z_stage_present:
                    focus_pos = layer_settings[layer]['focus']
                    ctx.scope.motion.move_absolute_sync(
                        'Z',
                        focus_pos,
                        wait_until_complete=True,
                    )

                gain = layer_settings[layer]['gain_db']
                ctx.scope.imaging.set_gain_sync(gain)
                exposure = layer_settings[layer]['exp_ms']
                ctx.scope.imaging.set_exposure_sync(exposure)
                sum_count = layer_settings[layer]['sum']
                # Stage B1: see comment above; update_scopedisplay retired.
                sum_iteration_callback = None

                # Compute brightness threshold (percentage -> absolute value)
                brightness_thresholds[layer] = (
                    layer_settings[layer]['composite_brightness_threshold'] / 100 * max_value
                )

                illumination = layer_settings[layer]['ill_ma']

                # Luminescence channels don't use an LED
                if layer not in common_utils.get_transmitted_layers():
                    ctx.scope.illumination.led_on_sync(
                        ctx.scope.illumination.color2ch(layer),
                        illumination,
                    )

                img_gray = ctx.scope.imaging.capture_and_wait_sync(
                    force_to_8bit=capture_depth == 8,
                    sum_count=sum_count,
                    sum_delay_s=exposure / 1000,
                    sum_iteration_callback=sum_iteration_callback,
                )
                ctx.scope.illumination.leds_off_sync()

                channel_images[layer] = np.array(img_gray)

            ctx.scope.illumination.leds_off_sync()

            # Unschedule histogram on main thread -- widget access must not happen from worker
            def _unschedule_histo(dt, layer_name=layer):
                lo = ctx.image_settings.layer_lookup(layer=layer_name)
                Clock.unschedule(lo.ids['histo_id'].histogram)

            Clock.schedule_once(_unschedule_histo, 0)
            logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')

        # Validate: at least one channel must have been captured
        if transmitted_image is None and len(channel_images) == 0:
            from modules.notification_center import notifications

            notifications.warning(
                'Composite',
                'No Channels Selected',
                'No channels are selected for capture. Enable at least one channel before using Composite Capture.',
            )
            logger.warning('[COMPOSITE] No channels selected -- nothing to capture')
            return

        # Build composite image from collected channels
        img = build_composite(
            channel_images=channel_images,
            transmitted_image=transmitted_image,
            brightness_thresholds=brightness_thresholds,
            dtype=dtype,
            max_value=max_value,
        )

        # File saving can run on this thread (no UI dependency)
        append = f'{ctx.scope.runtime_state.get_well_label()}'

        save_folder = pathlib.Path(live_folder) / 'Manual'
        save_folder.mkdir(parents=True, exist_ok=True)
        set_last_save_folder(dir=save_folder)

        if acquired_channel_count != 1 and acquired_channel_count != 0:
            save_image(
                ctx.scope,
                array=img,
                save_folder=save_folder,
                file_root='composite_',
                append=append,
                color=None,
                tail_id_mode='increment',
                output_format=image_output_format['live'],
                save_encoding=save_encoding,
            )
        elif acquired_channel_count != 0:
            save_image(
                ctx.scope,
                array=img,
                save_folder=save_folder,
                file_root=f'{most_recent_aq_channel}_Image_',
                append=append,
                color=None,
                tail_id_mode='increment',
                output_format=image_output_format['live'],
                save_encoding=save_encoding,
            )
        else:
            logger.info('[Composite Capture  ] No image saved as no channels were selected')

        # UI updates must happen on the main thread
        def _restore_ui(dt):
            ctx.lumaview.ids['composite_btn'].state = 'normal'
            live_histo_reverse()
            opened_layer_obj = common_utils.get_opened_layer_obj(ctx.image_settings)
            opened_layer_obj._initializing = True
            try:
                if led_restore_state:
                    opened_layer_obj.ids['enable_led_btn'].state = 'down'
                else:
                    opened_layer_obj.ids['enable_led_btn'].state = 'normal'
            finally:
                opened_layer_obj._initializing = False
            opened_layer_obj.apply_settings(update_led=True)

        # Restore video false color that was suspended during composite capture
        self.video_false_color = saved_video_false_color

        CompositeCapture._capturing.clear()
        Clock.schedule_once(_restore_ui, 0)
