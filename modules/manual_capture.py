# Copyright Etaluma, Inc.
"""Capture a still and save it: the one path for the GUI, scripts and REST.

The caller says only what it alone knows -- whether the shown layer is
displayed in false colour, which overlays it wants -- and every other
decision (channel, folder, name, summing, format, depth, encoding,
retry budget, overlay copy) is made here from the session's settings and the
API. Two callers making the same call get the same file.
"""

import concurrent.futures
import dataclasses
import logging
import pathlib
import threading
import time
from collections.abc import Callable
from typing import Any

import modules.common_utils as common_utils
import modules.config_helpers as config_helpers
import modules.image_utils as image_utils
from modules import capture_overlays, path_utils
from modules.exceptions import (
    CaptureFailedError,
    CapturePositionNotRecordedNotice,
    HardwareCommandRefusedError,
    LiveFolderPathRefusedError,
)
from modules.image_mode import OutputFormat
from modules.image_save import save_image
from modules.image_utils import ImageStatistics
from modules.layer_record import ChannelColors
from modules.lumascope_api.camera import LayerCameraRequest
from modules.lumascope_api.frame_record import CapturedStill, FrameRecord
from modules.lumascope_api.illumination import lit_other_than
from modules.recording_frames import frame_fact
from modules.api_surface import api, api_fields
from modules.operation import Operation
from modules.protocol_image_writer import WRITE_STALL_FATAL_S

logger = logging.getLogger('LVP.modules.manual_capture')

# How long the capture keeps re-grabbing for a frame that passes the content
# gate. One budget whether or not an overlay is asked for: the overlay is a
# copy of the same frame, not a reason to give up on it sooner.
_CAPTURE_RETRY_BUDGET_S = 5.0

_MEMBER = 'manual_capture.capture'


@api_fields(
    'channel',
    'file',
    'frame',
    'objective_id',
    'output_format',
    'plate_x_mm',
    'plate_y_mm',
    'significant_bits',
    'stage_z_um',
    'statistics',
    'well_label',
)
@dataclasses.dataclass(frozen=True)
class SavedStill:
    """One file a still wrote, with the instrument's record of it.

    Built at the capture from what the still read beside its grab, for
    every format alike: a JPG embeds none of it, so the record is the one
    place a JPG's channel, position and frame are stated.

    Attributes:
        file: The file written.
        channel: The layer it is named and set for.
        objective_id: The objective in the light path when it was taken.
        plate_x_mm: Plate X where it was taken; None when the axis had no
            reference.
        plate_y_mm: Plate Y, likewise.
        stage_z_um: Stage Z, likewise.
        well_label: The well at that position; None off the plate's wells,
            on a plate with none, or at an unknown position.
        significant_bits: The depth of the array saved: the frame's as
            captured, or 8 for an overlay copy, the frame's 8-bit rendering.
            A scaled encoding stores the frame left-justified in its file.
        output_format: The format it was saved in; never the hyperstack, which
            a single capture has no meaning for.
        frame: The instrument's account of the frame (``FrameRecord``); an
            overlay copy carries its frame's.
        statistics: The values of the array saved, in its counts at
            ``significant_bits``.
    """

    file: pathlib.Path
    channel: str
    objective_id: str
    plate_x_mm: float | None
    plate_y_mm: float | None
    stage_z_um: float | None
    well_label: str | None
    significant_bits: int
    output_format: OutputFormat
    frame: FrameRecord
    statistics: ImageStatistics


class ManualCaptureController:
    """One still at a time, captured on the camera lane and saved.

    Args:
        scope: The Lumascope instance.
        settings_snapshot: Returns a private copy of the session's settings;
            one is taken per capture, when the capture is asked for.
        shown_layer: Reads the layer of the session's shown channel, on the
            camera lane as the frame is taken: it names the channel when no
            LED is lit.
        layer_camera_request: Reads a layer's stored camera settings from
            the session's store, on the camera lane, for the camera to hold.
        engineering_mode: Reads the session's engineering mode, once per
            capture, when the capture is asked for: a plugin may turn it on
            after the controller is built.
        channel_colors: Reads every layer's colour, from the Session's one
            read of it, once per capture, when the capture is asked for.
    """

    def __init__(
        self,
        *,
        scope: Any,
        settings_snapshot: Callable[[], dict],
        shown_layer: Callable[[], str],
        layer_camera_request: Callable[[str], LayerCameraRequest],
        engineering_mode: Callable[[], bool],
        channel_colors: Callable[[], ChannelColors],
    ):
        self._scope = scope
        self._settings_snapshot = settings_snapshot
        self._shown_layer = shown_layer
        self._layer_camera_request = layer_camera_request
        self._engineering_mode = engineering_mode
        self._channel_colors = channel_colors
        # Held from the call until the camera-lane body has finished, so a
        # caller that stops waiting cannot let a second still overtake a slow
        # grab. Released on the lane, which a plain Lock allows.
        self._in_flight = threading.Lock()

    @api
    @property
    def in_flight(self) -> bool:
        """True from a capture call until its camera-lane body has ended.

        Raises:
            Nothing.
        """
        return self._in_flight.locked()

    @api
    def capture(
        self,
        *,
        false_color_on: bool,
        bullseye: bool = False,
        crosshairs: bool = False,
    ) -> Operation:
        """Capture one still and save it; returns at once.

        Args:
            false_color_on: Whether the shown layer is displayed in false
                colour -- how the file is rendered, never what it says was
                imaged.
            bullseye: Also save a copy with the bullseye colour map.
            crosshairs: Also save a copy with the centre crosshairs. With
                either overlay, one extra file is written, from the same
                frame, beside the unmarked one.

        The still is named for the shown layer (``session.show_channel``),
        read as its frame is taken, and the camera holds that layer's
        stored settings for the frame and after it; after a raw camera
        write it keeps the written values, and the frame's record says no
        layer was held. In the session's engineering mode the name carries
        the turret slot.

        Returns:
            An ``Operation`` whose result is a ``SavedStill`` per file
            written, the unmarked file first: its file and the
            instrument's record of it. Done once the still has let go of the camera lane's
            guard, so a second still asked then is not refused. The
            still is on the camera lane when this returns, so a run started
            after it waits for it when it is running or was put on an idle
            lane. A still has no stop: ``stop()`` does nothing. The
            operation raises what the
            capture raised: ``ObjectiveUnknownError`` when the objective in
            the light path is unknown (nothing captured),
            ``LiveFolderPathRefusedError`` (reason
            ``'capture_location_unusable'``) when the live folder cannot
            take the still, ``CaptureFailedError`` when no frame passed,
            with the capture's own reason (``capture_and_wait``);
            ``HardwareCommandRefusedError`` (reason ``'not_connected'``)
            when the camera is absent or was removed during the still,
            ``'camera_not_streaming'`` when its feed is stopped,
            ``'other_led_lit'`` when an LED other than the shown layer's is
            lit before the frame or lit it (nothing is saved), and
            ``'exclusive_activity_running'`` when a run took the scope
            while the still was queued behind other camera work, waiting or
            running when it was put on the lane, or when a recording runs and
            the camera would have to change layers for the still (nothing is
            written then, and the recording is unchanged); ``CancelledError`` when the lane dropped it unrun (the
            scope disconnected); ``LaneCallStillRunningError`` when it has
            not finished within the bound a capture and its write are given:
            the still may yet be taken and saved, its file logged when it is,
            and a second still is refused ``'capture_in_flight'`` until then.

        Raises:
            HardwareCommandRefusedError: reason ``'capture_in_flight'``,
                while an earlier still has not finished.
        """
        if not self._in_flight.acquire(blocking=False):
            raise HardwareCommandRefusedError('capture_in_flight', _MEMBER)
        try:
            request = _StillRequest(
                scope=self._scope,
                settings=self._settings_snapshot(),
                false_color_on=false_color_on,
                bullseye=bullseye,
                crosshairs=crosshairs,
                engineering_mode=self._engineering_mode(),
                channel_colors=self._channel_colors(),
            )
            waiter: concurrent.futures.Future = concurrent.futures.Future()
            # Marked running before the lane holds it, so only the lane
            # settles it and nothing releases the guard a second time
            # beside the lane. The guard is let go when the lane settles it,
            # before the caller is told, and never when the bound runs out:
            # a still past its bound may still run.
            waiter.set_running_or_notify_cancel()
            waiter.add_done_callback(lambda _: self._release_if_unrun(request))
            # Submitted on the caller's thread, under the caller's taking: a
            # diagnostic holding the scope takes a still under its own, and
            # a run that takes the scope after this returns finds the still
            # already on the lane.
            told = request.scope.camera._submit_camera(
                self._still_body,
                _MEMBER,
                args=(request,),
                waiter=waiter,
                bound_s=self._bound_s(request),
            )
        except BaseException:
            self._in_flight.release()
            raise
        return Operation.following(told)

    def _bound_s(self, request: '_StillRequest') -> float:
        """How long the caller waits for the still: the bound a capture's call
        has (``ImagingAPI._capture_wait_s``) for the sum of the layer shown
        now, the camera's hold of that layer, and the write's own stall
        bound. A still retaken on another shown layer shares the first
        capture's budget."""
        layer = self._shown_layer()
        layer_config = config_helpers.get_layer_configs(request.settings, [layer])[layer]
        grab_s = request.scope.imaging._capture_wait_s(
            (), _CAPTURE_RETRY_BUDGET_S, layer_config['sum'], layer_config['exposure_ms'] / 1000
        )
        return grab_s + request.scope.camera._CAMERA_WRITE_TIMEOUT_S + WRITE_STALL_FATAL_S

    def _release_if_unrun(self, request: '_StillRequest') -> None:
        # Refused while queued or dropped before the lane ran the body, so
        # the body's own release never happens.
        if not request.body_started.is_set():
            self._in_flight.release()

    def _still_body(self, request: '_StillRequest') -> list[SavedStill]:
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

    def _grab_named(self, scope, capture_config) -> tuple[str, CapturedStill]:
        """The still and the channel it was taken under.

        The channel is the shown layer, the name a manual recording takes
        too, read on the lane, and the camera holds that layer's stored
        settings for the grab (``ImagingAPI._capture_holding_layer_impl``),
        which refuses it while another LED is lit. The shown layer is read
        again once the frame has passed: a layer shown during the grab
        (``show_channel``) means the frame is taken again on it, and every
        retake shares the first capture's budget. The frame's record says
        which LEDs lit it, and one other than the shown layer's refuses the
        still.

        Raises:
            CaptureFailedError: ``'deadline_expired'``, the shown layer kept
                changing past the first capture's budget; and whatever the
                capture raises.
            HardwareCommandRefusedError: ``'other_led_lit'``, another LED
                was lit before the grab or lit the frame.
        """
        started = time.monotonic()
        first_capture = None
        channel = self._shown_layer()
        while True:
            # The summing row is the imaged channel's, from the store the camera
            # is made to hold; the delay is a settle between summed frames.
            layer_config = config_helpers.get_layer_configs(self._settings_snapshot(), [channel])[
                channel
            ]
            # The capture's body, not its public member: this still is already
            # the lane task its taking admitted, and the public member would be
            # admitted again, inline, against whatever holds the scope now -- a
            # run that took it after the still was put on the lane would refuse
            # the still it waits for.
            still = scope.imaging._capture_holding_layer_impl(
                channel,
                self._layer_camera_request(channel),
                _MEMBER,
                force_to_8bit=capture_config.capture_depth == 8,
                all_ones_check=True,
                timeout_s=_CAPTURE_RETRY_BUDGET_S,
                sum_count=layer_config['sum'],
                sum_delay_s=layer_config['exposure_ms'] / 1000,
            )
            if first_capture is None:
                first_capture = scope.imaging.last_capture_info
            shown = self._shown_layer()
            if shown == channel:
                if lit_other_than(scope.illumination, channel, still.record.illumination_ma):
                    raise HardwareCommandRefusedError('other_led_lit', _MEMBER)
                return channel, still
            if time.monotonic() - started > first_capture['deadline_s']:
                raise CaptureFailedError('deadline_expired')
            logger.info(
                f'[ManualCapture] The shown layer changed from {channel} to {shown} while '
                f'the still was taken; it is taken again on {shown}.'
            )
            channel = shown

    def _capture_and_save(self, request: '_StillRequest') -> list[SavedStill]:
        scope = request.scope
        settings = request.settings
        capture_config = config_helpers.get_image_capture_config_from_settings(settings)

        # The live folder is asked of the location owner before anything is
        # captured, and only the levels inside it are created here.
        try:
            manual_folder = path_utils.require_capture_location(settings['live_folder']) / 'Manual'
        except path_utils.CaptureLocationError as exc:
            raise LiveFolderPathRefusedError(
                'capture_location_unusable', str(settings['live_folder']), str(exc)
            ) from exc
        _make_folder(settings, manual_folder)

        # Read beside the grab: the file records the objective the frame was
        # taken with, and an unknown one refuses before anything is captured.
        objective_id, _ = scope.runtime_state.resolve_current_objective()
        channel, still = self._grab_named(scope, capture_config)
        array = still.image
        save_folder = manual_folder
        if settings['separate_folder_per_channel']:
            save_folder = manual_folder / channel
            _make_folder(settings, save_folder)

        # The instrument's account of this frame, handed back with it.
        frame_record = still.record
        # A blown or dark still is kept and saved, and told here, once: the
        # capture records it and leaves the telling to whoever took it.
        if frame_record.blown:
            logger.warning(
                f'[ManualCapture] The still is {frame_record.saturated_fraction * 100.0:.0f}% '
                f'saturated -- likely over-exposure or a stale camera gain; the frame may '
                f'be unusable.'
            )
        if frame_record.dark:
            logger.warning(
                '[ManualCapture] The still is dark -- no pixel above the dark floor with '
                'illumination on -- and was saved.'
            )
        labware = scope.runtime_state.get_labware()
        # The frame's own depth (8 for uint8, a sum's the bits it can reach,
        # else the per-frame delivery stamp), from its record.
        significant_bits = still.significant_bits
        # Where the stage was, read beside the grab with the reader a
        # recording uses for every frame: an axis without its reference
        # answers no position, and the file then states none for it.
        fact = frame_fact(scope, to_plate=scope.runtime_state.plate_transform())

        # The well at the position the file records, read beside the grab.
        well_label = fact.well_label(labware)
        if well_label is None and scope.capabilities.has_xy_stage:
            # The image is real; only where it was taken is not known. It is
            # saved, and says so once, here -- the save takes this label, so
            # this is the one place it is known to be missing. A scope with no
            # XY stage has no well to know, and homing would not give it one.
            from modules.notification_center import notifications

            notifications.report_outcome(
                CapturePositionNotRecordedNotice(), solicited=False, category='Capture'
            )
        # A zero-well plate or an unknown position has no label; no leading
        # underscore for it.
        append = f'{well_label}_{channel}' if well_label else channel
        # The writer's own renderer, so a manual still and a protocol step
        # spell the turret slot the same way; an unknown slot, or no turret,
        # adds nothing.
        turret = scope.motion.turret_status() if request.engineering_mode else None
        append = common_utils.build_step_name(
            common_utils.StepNameComponents(
                custom_prefix=append,
                turret_position=turret.slot if turret is not None else None,
            )
        )

        raw_path = save_image(
            scope,
            array,
            save_folder=save_folder,
            file_root='live_',
            append=append,
            tail_id_mode='increment',
            significant_bits=significant_bits,
            white_bits=frame_record.frame_significant_bits,
            channel=channel,
            false_color_on=request.false_color_on,
            channel_colors=request.channel_colors,
            output_format=capture_config.output_format_live,
            jpeg_quality=capture_config.jpg_quality,
            save_encoding=capture_config.save_encoding,
            plate_x_mm=fact.plate_x_mm,
            plate_y_mm=fact.plate_y_mm,
            stage_z_um=fact.z_um,
            objective_id=objective_id,
            frame_record=frame_record,
            labware=labware,
            well_label=well_label,
            # A manual still is taken where the stage stands; no autofocus
            # chose its focus.
            fallback_focus=None,
        )

        def saved(path: pathlib.Path, written: Any, bits: int) -> SavedStill:
            return SavedStill(
                file=path,
                channel=channel,
                objective_id=objective_id,
                plate_x_mm=fact.plate_x_mm,
                plate_y_mm=fact.plate_y_mm,
                stage_z_um=fact.z_um,
                well_label=well_label,
                significant_bits=bits,
                output_format=capture_config.output_format_live,
                frame=frame_record,
                statistics=image_utils.image_statistics(written),
            )

        stills = [saved(raw_path, array, significant_bits)]

        if request.bullseye or request.crosshairs:
            # The overlay is drawn on the 8-bit rendering the frame is shown
            # as: a sum against one frame's white, brighter, so its bullseye
            # bands sit where the sum is bright.
            overlay = image_utils.convert_sum_to_8bit(
                array, frame_record.frames_summed, frame_record.frame_significant_bits
            )
            if request.bullseye:
                overlay = capture_overlays.transform_to_bullseye(overlay)
            if request.crosshairs:
                overlay = capture_overlays.add_crosshairs(overlay)
            # Named after the unmarked file it copies, so the pair sorts
            # together and cannot be mismatched.
            raw_stem = raw_path.name[: len(raw_path.name) - len(''.join(raw_path.suffixes))]
            # The 8-bit rendering's depth, whatever the frame's.
            overlay_bits = 8
            overlay_path = save_image(
                scope,
                overlay,
                save_folder=save_folder,
                file_root='',
                append=f'{raw_stem}_overlay',
                tail_id_mode='if_collision',
                significant_bits=overlay_bits,
                # Already the 8-bit rendering, shown as it is.
                white_bits=None,
                channel=channel,
                false_color_on=request.false_color_on,
                channel_colors=request.channel_colors,
                output_format=capture_config.output_format_live,
                jpeg_quality=capture_config.jpg_quality,
                save_encoding=capture_config.save_encoding,
                plate_x_mm=fact.plate_x_mm,
                plate_y_mm=fact.plate_y_mm,
                stage_z_um=fact.z_um,
                objective_id=objective_id,
                frame_record=frame_record,
                labware=labware,
                well_label=well_label,
                fallback_focus=None,
            )
            stills.append(saved(overlay_path, overlay, overlay_bits))

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
            f'dtype={array.dtype} shape={array.shape} -> '
            f'{", ".join(s.file.name for s in stills)}'
        )
        return stills


def _make_folder(settings: dict, folder: pathlib.Path) -> None:
    """Create ``folder`` inside the live folder, or refuse the still.

    Raises:
        LiveFolderPathRefusedError: ``'capture_location_unusable'``.
    """
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LiveFolderPathRefusedError(
            'capture_location_unusable',
            str(settings['live_folder']),
            f'{folder.parent} is not an accessible save location. Check that it '
            'exists and any external drive is connected, then try again.',
        ) from exc


class _StillRequest:
    """Everything one still is decided from, fixed when it is asked for."""

    def __init__(
        self,
        *,
        scope: Any,
        settings: dict,
        false_color_on: bool,
        bullseye: bool,
        crosshairs: bool,
        engineering_mode: bool,
        channel_colors: ChannelColors,
    ):
        self.scope = scope
        self.settings = settings
        self.false_color_on = false_color_on
        self.bullseye = bullseye
        self.crosshairs = crosshairs
        self.engineering_mode = engineering_mode
        self.channel_colors = channel_colors
        # Set as the lane body's first act: after it, the body owns releasing
        # the in-flight guard.
        self.body_started = threading.Event()
