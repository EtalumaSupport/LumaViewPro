# Copyright Etaluma, Inc.
"""
CompositeCapture — shared image capture capabilities extracted from lumaviewpro.py.

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
from modules.composite_builder import build_composite
import modules.image_utils as image_utils
import modules.scope_commands as scope_commands
from modules.ui_helpers import (
    live_histo_off, live_histo_reverse, set_last_save_folder,
)

logger = logging.getLogger('LVP.ui.composite_capture')


class CompositeCapture(FloatLayout):

    _capturing = threading.Event()  # Thread-safe guard against rapid double-clicks

    def __init__(self, **kwargs):
        super(CompositeCapture,self).__init__(**kwargs)

    # Gets the current well label (ex. A1, C2, ...)
    def get_well_label(self):
        from modules.config_getters import get_selected_labware

        ctx = _app_ctx.ctx
        _, labware = get_selected_labware()

        # Get target position
        try:
            x_target = ctx.scope.get_target_position('X')
            y_target = ctx.scope.get_target_position('Y')
        except Exception:
            logger.exception('[LVP Main  ] Error talking to Motor board.')
            raise

        x_target, y_target = ctx.coordinate_transformer.stage_to_plate(
            labware=labware,
            stage_offset=ctx.settings['stage_offset'],
            sx=x_target,
            sy=y_target
        )

        return labware.get_well_label(x=x_target, y=y_target)


    def live_capture(self):
        if CompositeCapture._capturing.is_set():
            logger.warning('[LVP Main  ] Capture already in progress, ignoring')
            return
        CompositeCapture._capturing.set()
        try:
            self._live_capture_impl()
        finally:
            CompositeCapture._capturing.clear()

    def _live_capture_impl(self):
        from modules.config_getters import get_layer_configs

        logger.info('[LVP Main  ] CompositeCapture.live_capture()')

        ctx = _app_ctx.ctx
        settings = ctx.settings

        file_root = 'live_'
        color = 'BF'
        well_label = self.get_well_label()

        use_full_pixel_depth = ctx.scope_display.use_full_pixel_depth
        force_to_8bit_pixel_depth = not use_full_pixel_depth

        for layer in common_utils.get_layers():
            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
            accordion_item_obj =  ctx.image_settings.accordion_item_lookup(layer=layer)
            if not accordion_item_obj.collapse:
                append = f'{well_label}_{layer}'
                if layer_obj.ids['false_color'].active:
                    color = layer

                break

        save_folder = pathlib.Path(settings['live_folder']) / "Manual"
        separate_folder_per_channel = ctx.motion_settings.ids['microscope_settings_id']._seperate_folder_per_channel
        if separate_folder_per_channel:
            save_folder = save_folder / layer

        save_folder.mkdir(parents=True, exist_ok=True)
        set_last_save_folder(dir=save_folder)

        sum_iteration_callback = ctx.scope_display.update_scopedisplay

        layer_configs = get_layer_configs(specific_layers=layer)
        sum_delay_s=layer_configs[layer]['exposure']/1000
        sum_count=layer_configs[layer]['sum']

        if ctx.engineering_mode is False:
            return ctx.scope.save_live_image(
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
            )

        else:
            use_bullseye = ctx.scope_display.use_bullseye
            use_crosshairs = ctx.scope_display.use_crosshairs

            if not use_bullseye and not use_crosshairs:
                return ctx.scope.save_live_image(
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
                )

            image_orig = ctx.scope.get_image(force_to_8bit=force_to_8bit_pixel_depth)
            if image_orig is False:
                return

            # Save both versions of the image (unaltered and overlayed)
            now = datetime.datetime.now()
            time_string = now.strftime("%Y%m%d_%H%M%S")
            append = f"{append}_{time_string}"

            # If not in 8-bit mode, generate an 8-bit copy of the image for visualization
            if use_full_pixel_depth:
                image = image_utils.convert_12bit_to_8bit(image_orig)
            else:
                image = image_orig

            # Original image may be in 8 or 12-bit
            ctx.scope.save_image(
                array=image_orig,
                save_folder=save_folder,
                file_root=file_root,
                append=append,
                color=color,
                tail_id_mode=None,
                output_format=settings['image_output_format']
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
            ctx.scope.save_image(
                array=crosshairs_image,
                save_folder=save_folder,
                file_root=file_root,
                append=f"{append}_overlay",
                color=color,
                tail_id_mode=None,
                output_format=settings['image_output_format']
            )


    # capture and save a composite image using the current settings
    def composite_capture(self):
        ctx = _app_ctx.ctx

        if CompositeCapture._capturing.is_set():
            logger.warning('[LVP Main  ] Composite capture already in progress, ignoring')
            return
        CompositeCapture._capturing.set()

        z_stage_present = not ctx.disable_homing

        logger.info('[LVP Main  ] CompositeCapture.composite_capture()')

        initial_layer = common_utils.get_opened_layer(ctx.image_settings)

        if ctx.scope.get_led_state(initial_layer)['enabled']:
            led_restore_state = True
        else:
            led_restore_state = False

        live_histo_off()

        if ctx.scope.camera.active is None:
            return

        scope_display = self.ids['viewer_id'].ids['scope_display_id']
        use_full_pixel_depth = scope_display.use_full_pixel_depth

        # Run hardware-blocking work on a background thread to avoid freezing UI
        t = threading.Thread(
            target=self._composite_capture_worker,
            kwargs={
                'z_stage_present': z_stage_present,
                'initial_layer': initial_layer,
                'led_restore_state': led_restore_state,
                'use_full_pixel_depth': use_full_pixel_depth,
            },
            daemon=True,
            name='CompositeCapture',
        )
        t.start()

    def _composite_capture_worker(
        self,
        z_stage_present,
        initial_layer,
        led_restore_state,
        use_full_pixel_depth,
    ):
        """Runs on background thread — performs hardware I/O without blocking UI."""
        ctx = _app_ctx.ctx
        settings = ctx.settings
        io_executor = ctx.io_executor
        camera_executor = ctx.camera_executor

        # Snapshot settings at entry for thread safety — avoids seeing partial
        # updates from the UI thread during the capture sequence.
        all_layers = (
            *common_utils.get_transmitted_layers(),
            *common_utils.get_fluorescence_layers(),
            *common_utils.get_luminescence_layers(),
        )
        with ctx.settings_lock:
            layer_settings = {layer: dict(settings[layer]) for layer in all_layers}
            frame_settings = dict(settings['frame'])
            live_folder = settings['live_folder']
            image_output_format = dict(settings['image_output_format'])

        acquired_channel_count = 0
        most_recent_aq_channel = None

        if use_full_pixel_depth:
            dtype = np.uint16
            max_value = 4095
        else:
            dtype = np.uint8
            max_value = 255

        transmitted_image = None
        channel_images = {}
        brightness_thresholds = {}

        # Capture transmitted channel (BF/PC/DF) — use first found as base
        for trans_layer in common_utils.get_transmitted_layers():
            if layer_settings[trans_layer]["acquire"] == "image":
                acquired_channel_count += 1
                most_recent_aq_channel = trans_layer

                if z_stage_present:
                    focus_pos = layer_settings[trans_layer]['focus']
                    scope_commands.move_absolute_sync(
                        ctx.scope, io_executor, 'Z', focus_pos,
                        wait_until_complete=True,
                    )

                gain = layer_settings[trans_layer]['gain']
                scope_commands.set_gain_sync(ctx.scope, camera_executor, gain)
                exposure = layer_settings[trans_layer]['exp']
                scope_commands.set_exposure_sync(ctx.scope, camera_executor, exposure)
                illumination = layer_settings[trans_layer]['ill']

                scope_commands.led_on_sync(
                    ctx.scope, io_executor,
                    ctx.scope.color2ch(trans_layer), illumination,
                )

                transmitted_image = np.array(
                    scope_commands.capture_and_wait_sync(
                        ctx.scope, camera_executor,
                        force_to_8bit=not use_full_pixel_depth,
                    ),
                    dtype=dtype,
                )
                scope_commands.leds_off_sync(ctx.scope, io_executor)

                # Can only use one transmitted channel per composite
                break

        scope_commands.leds_off_sync(ctx.scope, io_executor)

        # Capture fluorescence and luminescence channels
        for layer in (*common_utils.get_fluorescence_layers(), *common_utils.get_luminescence_layers()):
            if layer_settings[layer]['acquire'] == "image":
                acquired_channel_count += 1
                most_recent_aq_channel = layer

                if z_stage_present:
                    focus_pos = layer_settings[layer]['focus']
                    scope_commands.move_absolute_sync(
                        ctx.scope, io_executor, 'Z', focus_pos,
                        wait_until_complete=True,
                    )

                gain = layer_settings[layer]['gain']
                scope_commands.set_gain_sync(ctx.scope, camera_executor, gain)
                exposure = layer_settings[layer]['exp']
                scope_commands.set_exposure_sync(ctx.scope, camera_executor, exposure)
                sum_count = layer_settings[layer]['sum']
                sum_iteration_callback = ctx.scope_display.update_scopedisplay

                # Compute brightness threshold (percentage → absolute value)
                brightness_thresholds[layer] = layer_settings[layer]["composite_brightness_threshold"] / 100 * max_value

                illumination = layer_settings[layer]['ill']

                # Luminescence channels don't use an LED
                if layer not in common_utils.get_transmitted_layers():
                    scope_commands.led_on_sync(
                        ctx.scope, io_executor,
                        ctx.scope.color2ch(layer), illumination,
                    )

                img_gray = scope_commands.capture_and_wait_sync(
                    ctx.scope, camera_executor,
                    force_to_8bit=not use_full_pixel_depth,
                    sum_count=sum_count,
                    sum_delay_s=exposure/1000,
                    sum_iteration_callback=sum_iteration_callback,
                )
                scope_commands.leds_off_sync(ctx.scope, io_executor)

                channel_images[layer] = np.array(img_gray)

            scope_commands.leds_off_sync(ctx.scope, io_executor)

            # Unschedule histogram on main thread — widget access must not happen from worker
            def _unschedule_histo(dt, layer_name=layer):
                lo = ctx.image_settings.layer_lookup(layer=layer_name)
                Clock.unschedule(lo.ids['histo_id'].histogram)
            Clock.schedule_once(_unschedule_histo, 0)
            logger.info('[LVP Main  ] Clock.unschedule(lumaview...histogram)')

        # Build composite image from collected channels
        img = build_composite(
            channel_images=channel_images,
            transmitted_image=transmitted_image,
            brightness_thresholds=brightness_thresholds,
            dtype=dtype,
            max_value=max_value,
        )

        # File saving can run on this thread (no UI dependency)
        append = f'{self.get_well_label()}'

        save_folder = pathlib.Path(live_folder) / "Manual"
        save_folder.mkdir(parents=True, exist_ok=True)
        set_last_save_folder(dir=save_folder)

        if acquired_channel_count != 1 and acquired_channel_count != 0:
            ctx.scope.save_image(
                array=img,
                save_folder=save_folder,
                file_root='composite_',
                append=append,
                color=None,
                tail_id_mode='increment',
                output_format=image_output_format['live']
            )
        elif acquired_channel_count != 0:
            ctx.scope.save_image(
                array=img,
                save_folder=save_folder,
                file_root=f"{most_recent_aq_channel}_Image_",
                append=append,
                color=None,
                tail_id_mode='increment',
                output_format=image_output_format['live']
            )
        else:
            logger.info("[Composite Capture  ] No image saved as no channels were selected")

        # UI updates must happen on the main thread
        def _restore_ui(dt):
            ctx.lumaview.ids['composite_btn'].state = 'normal'
            live_histo_reverse()
            opened_layer_obj = common_utils.get_opened_layer_obj(ctx.image_settings)
            if led_restore_state:
                opened_layer_obj.ids['enable_led_btn'].state = 'down'
            else:
                opened_layer_obj.ids['enable_led_btn'].state = 'normal'
            opened_layer_obj.apply_settings(update_led=True)

        CompositeCapture._capturing.clear()
        Clock.schedule_once(_restore_ui, 0)
