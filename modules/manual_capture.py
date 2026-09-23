# Copyright Etaluma, Inc.
"""Capture a still and save it: the one path for the GUI, scripts and REST.

The caller says only what it alone knows -- which drawer is open, whether
that layer is shown in false colour, which overlays it wants -- and every
other decision (channel, folder, name, summing, format, depth, encoding,
retry budget, overlay copy) is made here from the session's settings and the
API. Two callers making the same call get the same file.
"""

import concurrent.futures
import logging
import pathlib
import threading
from collections.abc import Callable
from typing import Any

import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
import modules.image_utils as image_utils
from modules import capture_overlays
from modules.exceptions import CaptureError, HardwareCommandRefusedError
from modules.image_save import save_image
from modules.lumascope_api.imaging import capture_failure_cause

logger = logging.getLogger('LVP.modules.manual_capture')

# How long the capture keeps re-grabbing for a frame that passes the content
# gate. One budget whether or not an overlay is asked for: the overlay is a
# copy of the same frame, not a reason to give up on it sooner.
_CAPTURE_RETRY_BUDGET_S = 5.0

_MEMBER = 'manual_capture.capture'


class ManualCaptureController:
    """One still at a time, captured on the camera lane and saved.

    Args:
        scope: The Lumascope instance.
        settings_snapshot: Returns a private copy of the session's settings;
            one is taken per capture, when the capture is asked for.
        engineering_mode: The session's as-built mode, used when a caller
            does not pass the live flag.
    """

    def __init__(
        self,
        *,
        scope: Any,
        settings_snapshot: Callable[[], dict],
        engineering_mode: bool,
    ):
        self._scope = scope
        self._settings_snapshot = settings_snapshot
        self._engineering_mode = engineering_mode
        # Held from the call until the camera-lane body has finished, so a
        # caller that stops waiting cannot let a second still overtake a slow
        # grab. Released on the lane, which a plain Lock allows.
        self._in_flight = threading.Lock()

    def set_scope(self, scope: Any) -> None:
        """Rewire onto a NEW scope after a reconnect. A still already in
        flight finishes on the scope it started on."""
        self._scope = scope

    @property
    def in_flight(self) -> bool:
        """True from a capture call until its camera-lane body has ended."""
        return self._in_flight.locked()

    def capture(
        self,
        *,
        layer: str | None,
        false_color_on: bool,
        bullseye: bool = False,
        crosshairs: bool = False,
        engineering_mode: bool | None = None,
    ) -> 'concurrent.futures.Future[list[pathlib.Path]]':
        """Capture one still and save it; returns at once.

        Args:
            layer: The layer whose drawer is open, or None. Names the channel
                only when no LED is lit (``resolve_channel_identity``).
            false_color_on: Whether that layer is shown in false colour --
                how the file is rendered, never what it says was imaged.
            bullseye: Also save a copy with the bullseye colour map.
            crosshairs: Also save a copy with the centre crosshairs. With
                either overlay, one extra file is written, from the same
                frame, beside the unmarked one.
            engineering_mode: Whether the name carries the turret slot. None
                takes the session's as-built mode; a host whose plugin flips
                the mode at run time passes its live flag.

        Returns:
            A Future of the paths written, the unmarked file first. It
            raises what the capture raised: ``ObjectiveUnknownError`` when
            the objective in the light path is unknown (nothing captured),
            ``HardwareCommandRefusedError`` when a run holds the camera,
            ``CaptureError`` (reason ``'no_frame_returned'``) when no frame
            passed, with the capture engine's cause as its message. The
            Future sets no timeout: its caller bounds its own wait.

        Raises:
            ValueError: ``layer`` is not a channel. Refused before anything
                is named, created or captured.
            HardwareCommandRefusedError: reason ``'capture_in_flight'``,
                while an earlier still has not finished.
        """
        if layer is not None and layer not in common_utils.get_layers():
            raise ValueError(
                f'layer {layer!r} is not a channel; expected one of {common_utils.get_layers()}'
            )
        if not self._in_flight.acquire(blocking=False):
            raise HardwareCommandRefusedError('capture_in_flight', _MEMBER)
        try:
            request = _StillRequest(
                scope=self._scope,
                settings=self._settings_snapshot(),
                layer=layer,
                false_color_on=false_color_on,
                bullseye=bullseye,
                crosshairs=crosshairs,
                engineering_mode=(
                    self._engineering_mode if engineering_mode is None else engineering_mode
                ),
            )
            future: concurrent.futures.Future = concurrent.futures.Future()
            future.set_running_or_notify_cancel()
            threading.Thread(
                target=self._run,
                args=(request, future),
                name='manual-capture',
                daemon=True,
            ).start()
        except BaseException:
            self._in_flight.release()
            raise
        return future

    def _run(self, request: '_StillRequest', future: concurrent.futures.Future) -> None:
        # The camera executor's own future is a per-thread waiter reused on
        # that thread's next submit, so it cannot be handed to a caller; this
        # thread waits on it and settles a standard Future instead.
        try:
            paths = request.scope.imaging._dispatch_camera(
                self._still_body,
                _MEMBER,
                args=(request,),
                timeout_s=None,
            )
        except BaseException as exc:
            if not request.body_started.is_set():
                # Refused or cancelled before the lane ran the body, so the
                # body's own release never happens.
                self._in_flight.release()
            future.set_exception(exc)
            return
        future.set_result(paths)

    def _still_body(self, request: '_StillRequest') -> list[pathlib.Path]:
        """The whole still on the camera worker: grab, record and write.

        One lane task, so no camera write can land between the grab and the
        record the save builds about it. The save reads the camera and LED
        when it builds that record; writing off the lane needs a record that
        travels with the frame, which the save does not yet take.
        """
        request.body_started.set()
        try:
            return self._capture_and_save(request)
        finally:
            self._in_flight.release()

    def _capture_and_save(self, request: '_StillRequest') -> list[pathlib.Path]:
        scope = request.scope
        settings = request.settings
        capture_config = config_helpers.get_image_capture_config_from_settings(settings)

        # What was imaged: the lit channel, else the open drawer, else
        # brightfield -- the rule a manual recording uses too.
        channel = common_utils.resolve_channel_identity(scope.illumination, request.layer)

        well_label = scope.runtime_state.get_well_label()
        # A zero-well plate has no label; no leading underscore for it.
        append = f'{well_label}_{channel}' if well_label else channel
        # The writer's own renderer, so a manual still and a protocol step
        # spell the turret slot the same way; an unknown slot adds nothing.
        append = common_utils.build_step_name(
            common_utils.StepNameComponents(
                custom_prefix=append,
                turret_position=(
                    scope.motion.get_turret_slot() if request.engineering_mode else None
                ),
            )
        )

        save_folder = pathlib.Path(settings['live_folder']) / 'Manual'
        if settings['separate_folder_per_channel']:
            save_folder = save_folder / channel
        save_folder.mkdir(parents=True, exist_ok=True)

        # The summing row is the imaged channel's; the delay is a settle
        # between summed frames.
        layer_config = config_helpers.get_layer_configs(settings, [channel])[channel]
        sum_count = layer_config['sum']
        sum_delay_s = layer_config['exposure_ms'] / 1000

        # Read beside the grab: the file records the objective the frame was
        # taken with, and an unknown one refuses before anything is captured.
        objective_id, _ = scope.runtime_state.resolve_current_objective()
        array = scope.imaging._capture_and_wait_impl(
            force_to_8bit=capture_config.capture_depth == 8,
            all_ones_check=True,
            timeout_s=_CAPTURE_RETRY_BUDGET_S,
            sum_count=sum_count,
            sum_delay_s=sum_delay_s,
        )
        if array is None:
            raise CaptureError(
                capture_failure_cause(scope.imaging.last_capture_info),
                'no_frame_returned',
            )
        # The frame's own depth (8 for uint8, 16 for a summed container, else
        # the per-frame delivery stamp), taken now, before any later grab can
        # change what the camera reports.
        significant_bits = scope.imaging.capture_frame_depth(array, sum_count)

        raw_path = save_image(
            scope,
            array,
            save_folder=save_folder,
            file_root='live_',
            append=append,
            tail_id_mode='increment',
            significant_bits=significant_bits,
            channel=channel,
            false_color_on=request.false_color_on,
            output_format=capture_config.output_format_live,
            jpeg_quality=capture_config.jpg_quality,
            save_encoding=capture_config.save_encoding,
            objective_id=objective_id,
        )
        paths = [raw_path]

        if request.bullseye or request.crosshairs:
            # The overlay is drawn on the 8-bit rendering, scaled against the
            # frame's own depth: a summed frame exceeds the per-frame range,
            # and scaling it against that range refuses the frame.
            overlay = image_utils.convert_to_8bit(array, significant_bits)
            if request.bullseye:
                overlay = capture_overlays.transform_to_bullseye(overlay)
            if request.crosshairs:
                overlay = capture_overlays.add_crosshairs(overlay)
            # Named after the unmarked file it copies, so the pair sorts
            # together and cannot be mismatched.
            raw_stem = raw_path.name[: len(raw_path.name) - len(''.join(raw_path.suffixes))]
            paths.append(
                save_image(
                    scope,
                    overlay,
                    save_folder=save_folder,
                    file_root='',
                    append=f'{raw_stem}_overlay',
                    tail_id_mode='if_collision',
                    significant_bits=scope.imaging.capture_frame_depth(overlay),
                    channel=channel,
                    false_color_on=request.false_color_on,
                    output_format=capture_config.output_format_live,
                    jpeg_quality=capture_config.jpg_quality,
                    save_encoding=capture_config.save_encoding,
                    objective_id=objective_id,
                )
            )

        # Both depths, because they differ: a scaled encoding left-justifies a
        # 12-bit capture to fill the 16-bit container, so the file is 16-bit
        # while the sensor gave 12, and logging one of them reads as a file
        # tagged at the wrong depth.
        saved_significant_bits = image_utils.written_significant_bits(
            capture_config.save_encoding,
            significant_bits,
            array.dtype,
            image_utils.is_color_image(array),
        )
        logger.info(
            f'[ManualCapture] encoding={capture_config.save_encoding} '
            f'capture_bits={significant_bits} saved_significant_bits={saved_significant_bits} '
            f'dtype={array.dtype} shape={array.shape} -> {", ".join(p.name for p in paths)}'
        )
        return paths


class _StillRequest:
    """Everything one still is decided from, fixed when it is asked for."""

    def __init__(
        self,
        *,
        scope: Any,
        settings: dict,
        layer: str | None,
        false_color_on: bool,
        bullseye: bool,
        crosshairs: bool,
        engineering_mode: bool,
    ):
        self.scope = scope
        self.settings = settings
        self.layer = layer
        self.false_color_on = false_color_on
        self.bullseye = bullseye
        self.crosshairs = crosshairs
        self.engineering_mode = engineering_mode
        # Set as the lane body's first act: after it, the body owns releasing
        # the in-flight guard.
        self.body_started = threading.Event()
