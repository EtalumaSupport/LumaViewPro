# Copyright Etaluma, Inc.
"""
CompositeCapture -- shared image capture capabilities extracted from lumaviewpro.py.

Provides live_capture() and composite_capture() methods inherited by MainDisplay.
"""

import datetime
import logging
import pathlib
import threading


from kivy.uix.floatlayout import FloatLayout

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
from modules import gui_logger
from modules.image_save import save_image, save_live_image
import modules.image_utils as image_utils
from modules.sequential_io_executor import IOTask, PRIORITY_HIGH
from ui.ui_helpers import (
    live_histo_off,
    live_histo_reverse,
    reset_title,
    run_with_refusal_boundary,
    set_last_save_folder,
    set_title_event_text,
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
        false_color_on = False
        well_label = ctx.scope.runtime_state.get_well_label()

        image_capture_config = get_image_capture_config_from_ui()
        force_to_8bit_pixel_depth = image_capture_config.capture_depth == 8
        save_encoding = image_capture_config.save_encoding

        for layer in common_utils.get_layers():
            layer_obj = ctx.image_settings.layer_lookup(layer=layer)
            accordion_item_obj = ctx.image_settings.accordion_item_lookup(layer=layer)
            if not accordion_item_obj.collapse:
                # Empty well label (zero-well Blank labware): no leading
                # underscore from a missing segment.
                append = f'{well_label}_{layer}' if well_label else layer
                # In engineering mode the name carries the turret position,
                # composed by the writer's own renderer so a manual capture
                # and a protocol step spell it the same way and a filename
                # reader recognises it. A position the scope has not
                # reported yet adds nothing.
                append = common_utils.build_step_name(
                    common_utils.StepNameComponents(
                        custom_prefix=append,
                        turret_position=(
                            ctx.scope.motion._last_turret_position if ctx.engineering_mode else None
                        ),
                    )
                )
                # The checkbox answers how the frame is DISPLAYED, and nothing
                # else. What was imaged is the opened layer, passed separately
                # below -- reading the channel off this checkbox is what made
                # every false-color-off capture claim to be brightfield while
                # its own filename said otherwise.
                false_color_on = layer_obj.ids['false_color'].active

                break

        save_folder = pathlib.Path(settings['live_folder']) / 'Manual'
        separate_folder_per_channel = settings['separate_folder_per_channel']
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

        # The dark-floor expectation is derived inside the capture from
        # commanded LED state: a live capture judges the frame against what
        # is actually lit right now, not against the settings store -- a
        # layer configured at 200 mA with its LED off is a deliberate dark
        # capture, and a black frame under a lit channel fails loudly.
        # Whether an overlay is active is the only question a capture has to
        # ask. The operator can switch crosshairs or the bullseye on at any
        # time, so gating the overlaid copy on a build mode meant the screen
        # showed an overlay the capture then declined to save, with nothing
        # reporting the omission.
        use_bullseye = ctx.scope_display.use_bullseye
        use_crosshairs = ctx.scope_display.use_crosshairs

        if not use_bullseye and not use_crosshairs:
            return save_live_image(
                ctx.scope,
                save_folder=save_folder,
                file_root=file_root,
                append=append,
                channel=layer,
                false_color_on=false_color_on,
                force_to_8bit=force_to_8bit_pixel_depth,
                output_format=settings['image_output_format']['live'],
                all_ones_check=True,
                sum_count=sum_count,
                sum_delay_s=sum_delay_s,
                sum_iteration_callback=sum_iteration_callback,
                turn_off_all_leds_after=False,
                jpeg_quality=settings.get('jpg_quality', 90),
                save_encoding=save_encoding,
            )

        # Summing is carried here exactly as save_live_image carries it above:
        # an overlay is a display choice, and switching one on must not
        # silently reduce a summed capture to a single frame.
        image_orig = ctx.scope.imaging._capture_and_wait_impl(
            force_to_8bit=force_to_8bit_pixel_depth,
            all_ones_check=True,
            timeout_s=1.0,
            sum_count=sum_count,
            sum_delay_s=sum_delay_s,
            sum_iteration_callback=sum_iteration_callback,
        )
        if image_orig is None:
            return

        # Save both versions of the image (unaltered and overlayed)
        now = datetime.datetime.now()
        time_string = now.strftime('%Y%m%d_%H%M%S')
        append = f'{append}_{time_string}'

        # If not in 8-bit mode, generate an 8-bit copy of the image for
        # visualization. image_orig is a single native-depth capture here,
        # so its depth is the per-frame delivery stamp (not a live format
        # query, which can fail or already describe a newer format) so
        # the downconvert scales against the real range.
        if not force_to_8bit_pixel_depth:
            image = image_utils.convert_to_8bit(image_orig, ctx.scope.imaging.last_significant_bits)
        else:
            image = image_orig

        # Original image may be in 8 or 12-bit. Its depth is the per-frame
        # delivery stamp of the capture above (the same value the 8-bit copy
        # is scaled by), handed down so the save marks the file at the frame's
        # true depth rather than the camera's live format. Summing widens that
        # depth, so the frame count resolves it here for the same reason it
        # does inside save_live_image.
        save_image(
            ctx.scope,
            array=image_orig,
            save_folder=save_folder,
            file_root=file_root,
            append=append,
            channel=layer,
            false_color_on=false_color_on,
            tail_id_mode=None,
            output_format=settings['image_output_format']['live'],
            jpeg_quality=settings.get('jpg_quality', 90),
            save_encoding=save_encoding,
            significant_bits=ctx.scope.imaging.capture_frame_depth(image_orig, sum_count),
        )

        if use_bullseye:
            bullseye_image = ctx.scope_display.transform_to_bullseye(image)
        else:
            bullseye_image = image

        if use_crosshairs:
            crosshairs_image = ctx.scope_display.add_crosshairs(bullseye_image)
        else:
            crosshairs_image = bullseye_image

        # Overlay image is in 8-bits (rendered display image), so its depth is
        # read off the rendered array and not widened by the frame count --
        # the downconvert above already normalised the summed range away.
        save_image(
            ctx.scope,
            array=crosshairs_image,
            save_folder=save_folder,
            file_root=file_root,
            append=f'{append}_overlay',
            channel=layer,
            false_color_on=false_color_on,
            tail_id_mode=None,
            output_format=settings['image_output_format']['live'],
            jpeg_quality=settings.get('jpg_quality', 90),
            save_encoding=save_encoding,
            significant_bits=ctx.scope.imaging.capture_frame_depth(crosshairs_image),
        )

    # capture and save a composite image using the current settings
    def composite_capture(self):
        """Start a composite run, or stop the one already running.

        A composite is a sequenced run like a scan or a z-stack, so this
        is a run starter and nothing more. It states no run parameters and
        assembles no config: everything the run needs is settings the
        engine already reads, and duplicating that assembly here is what
        put a second composite implementation in the GUI to begin with.

        Only the concerns the engine cannot own stay here. It cannot know
        the toggle was clicked a second time, it does not share the guard
        that makes the two capture buttons mutually exclusive, and it has
        no refusal for a camera that is connected but not yet streaming.
        Everything else -- a rival run, files still draining, too few
        channels -- is the engine's refusal to raise, not this starter's
        to pre-check.
        """
        gui_logger.button('COMPOSITE_CAPTURE')
        ctx = _app_ctx.ctx
        composite_btn = self.ids['composite_btn']
        runner = ctx.session.create_protocol_runner()

        # The button is its own stop control, so a second click means stop.
        # It reads as one of two things: a toggle already back to 'normal',
        # or a click arriving while this starter's own run is live. The
        # trigger source is what separates the second case from a click
        # during someone ELSE's run, which must fall through to the engine
        # and be refused rather than aborting a run this button never
        # started.
        #
        # Reset goes onto the worker pool at high priority because the pool
        # runs exactly one worker: a stop that queued behind ordinary work
        # would not arrive until that work finished, which is the thing the
        # user is trying to interrupt.
        if composite_btn.state == 'normal' or (
            runner.is_running() and runner.run_trigger_source() == 'composite'
        ):
            ctx.worker_pool.put(IOTask(action=runner.reset, priority=PRIORITY_HIGH))
            return

        # Every gate below puts the toggle back before returning. Left
        # 'down', it makes the NEXT click read as the second click of a
        # pair, and that click is swallowed as an abort of a run that was
        # never started.
        if CompositeCapture._capturing.is_set():
            logger.warning('[LVP Main  ] Composite capture already in progress, ignoring')
            composite_btn.state = 'normal'
            return

        from modules.notification_center import notifications

        if not getattr(ctx.scope, 'camera_connected', True):
            notifications.warning(
                'Camera',
                'Camera not connected',
                'Cannot capture composite -- camera is not connected. '
                'Check USB and reconnect, then try again.',
            )
            composite_btn.state = 'normal'
            return

        if not ctx.scope.imaging.active_cached:
            notifications.warning(
                'Camera',
                'Camera not active',
                'Cannot capture composite -- the camera is not streaming. '
                'Wait for the camera to start, then try again.',
            )
            composite_btn.state = 'normal'
            return

        # Set only once every gate above has passed, and cleared in exactly
        # one place per outcome: the finally below for anything that does
        # not reach a live run, and the run's own completion for anything
        # that does. A guard left set is permanent -- both capture entry
        # points return at their is_set() check before enqueuing the work
        # whose completion would clear it -- so a path with no clearer
        # disables both capture buttons for the life of the process. That
        # is why the clear sits in a finally rather than at each exit: the
        # refusal boundary only catches the typed refusal, and a
        # programming error at the call site raises straight past it.
        CompositeCapture._capturing.set()
        started = False
        try:
            live_histo_off()
            set_title_event_text('Compositing...')

            settings = ctx.settings
            parent_dir = pathlib.Path(settings['live_folder']).resolve() / 'Manual' / 'Composites'

            def _start():
                nonlocal started
                runner.start_composite(
                    sequence_name='composite',
                    parent_dir=parent_dir,
                    callbacks={'run_complete': self._composite_finished},
                    run_trigger_source='composite',
                    engineering_mode=ctx.engineering_mode,
                )
                started = True
                # Only reachable once the run is committed, so the saved
                # folder can only ever name THIS run's directory.
                set_last_save_folder(dir=runner.run_dir())

            # A refusal has already been logged and shown to the user by the
            # engine's funnel; there is nothing left to report, only the
            # cosmetics to undo, and the finally does that.
            run_with_refusal_boundary(_start, on_refused=lambda: None)
        except Exception as e:
            logger.error(f'[LVP Main  ] composite_capture failed: {e}', exc_info=True)
            from ui.notification_popup import show_notification_popup

            show_notification_popup(title='Error', message=str(e))
        finally:
            if not started:
                self._composite_finished()

    def _composite_finished(self, **kwargs):
        """Hand the UI back after a composite ends or never starts.

        One handler for both, because every step is level-based rather
        than a guess about what the run did: the reconcile reads the LED
        driver instead of assuming, and the histogram and title helpers
        are idempotent. A second handler for the not-started path would
        be the same five lines with one omitted.

        This fires at RUN end, not merge end. The merged file lands about
        a second later, exactly as it does for every other run kind; a
        button that waited for it would be the only one in the app that
        did.
        """
        self.ids['composite_btn'].state = 'normal'
        reset_title()
        live_histo_reverse()
        # The run's LED restore has settled by now, so reconcile every
        # enable toggle to what the driver actually reports: a restore that
        # emits no LED events leaves the buttons stale otherwise.
        _app_ctx.ctx.ui_listener_bridge.reconcile_led_buttons()
        CompositeCapture._capturing.clear()
