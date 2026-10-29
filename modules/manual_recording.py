# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Manual video recording: the controller between callers and the engine.

Drives ``VideoRecordingEngine`` for manual recordings (the GUI Record
button today, REST/SDK callers tomorrow). The controller owns everything
caller-shaped the engine deliberately does not: reading settings into
the immutable ``RecordingConfig`` snapshot, building the per-leg write
edge (TIFF-per-frame or VFR MP4), registering the camera frame listener,
the rolling disk-floor stop, the elapsed-time duration cap, and the
post-drain finish (MP4 close, optional hyperstack, drop notification).

GUI-agnostic by construction: no Kivy imports; the caller polls status
properties for titles/buttons, and dispatches its own cleanup from the
recording's ``Operation``.

Timing axis: the engine receives host-epoch seconds. When the camera
reports hardware timestamp ticks and a tick frequency, per-frame times
are the camera's own clock rebased onto the host epoch at the first
frame -- camera-grade intervals on the host axis, which cadence
selection and the VFR pts both need. Without usable ticks the host
arrival time is used and the manifest's timestamp grade reports it.
"""

import concurrent.futures
import datetime
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from drivers.camera import Frame
import modules.image_mode as image_mode
import modules.image_save as image_save
import modules.image_utils as image_utils
import modules.layer_record as layer_record
import modules.path_utils as path_utils
from lvp_logger import logger, version as lvp_version
from modules.activity_claim import ControlSession, guarded_write
from modules.common_utils import (
    DISK_FLOOR_CHECK_INTERVAL_S,
    MIN_REQUIRED_DISK_MB,
    check_disk_space_ok,
)
from modules.coord_transformations import PlatePosition
from modules.labware import Labware
from modules.axes import StageAxis
from modules.config_helpers import (
    get_image_capture_config_from_settings,
    get_manual_video_max_duration,
)
from modules.exceptions import (
    POST_PROCESSING_RUNNING_MESSAGE,
    CaptureFailedError,
    FrameListenerNotRegisteredError,
    HyperstackRefusedError,
    InputFramesUnusableError,
    RecordingDetailsNotSavedError,
    RecordingFinalizeError,
    RecordingPositionNotRecordedNotice,
    RecordingRefusedError,
    RecordingStoppedError,
    VideoFramesDroppedError,
    VideoWriterFailedError,
)
from modules.lumascope_api.frame_record import FrameTime
from modules.lumascope_api.illumination import snapshot_lit_pairs
from modules.notification_center import notifications
from modules.operation import Operation
from modules.recording_frames import (
    MANUAL_HYPERSTACK_FILENAME,
    CameraTickRebaser,
    RecordingCamera,
    arrival,
    manual_frame_filename_template,
    orient_and_fit,
    filename_time,
    recording_camera,
    write_frame,
)
from modules.recording_manifest import gather_host_provenance
from modules.video_cadence import StallWatch, effective_recording_fps, stall_threshold_s
from modules.video_recording import (
    END_REASON_START_FAILED,
    RecordingConfig,
    VideoRecordingEngine,
)
from modules.scheduler import Scheduler
from modules.video_writer import VideoWriter
from modules.api_surface import Admission, api
from modules.caller import current_caller

if TYPE_CHECKING:
    from modules.post_processing_api import PostProcessingAPI

# The channel LumaViewPro comes up on. A recording that can name no other
# channel was taken on this one, so identity always has a real value and
# nothing downstream has to represent a channel-less recording.

# Health-check cadence while a recording is live. Matches the cadence the
# GUI historically polled at; the check is three reads, so ten a second
# is cheap and catches a duration cap or a dead feed within a tenth of a
# second.
HEALTH_CHECK_INTERVAL_S = 0.1


def sweep_recording_scratch(live_folder) -> None:
    """Delete a leftover pre-engine recording scratch file at startup.

    Earlier releases recorded through a multi-GB ``recording_temp.dat``
    memmap in the user's live folder; a crash mid-recording stranded it
    there (headerless raw -- nothing recoverable exists). The engine
    writes final artifacts directly, so any such file is pure litter:
    delete it with one INFO line carrying its size. Only the current
    live folder is swept; an orphan in a previously-used folder is
    unreachable and stays.
    """
    path = Path(live_folder) / 'recording_temp.dat'
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except FileNotFoundError:
        return
    except OSError as e:
        logger.warning(f'[ManualRecord] Could not stat leftover scratch {path}: {e}')
        return
    try:
        path.unlink()
        logger.info(f'[ManualRecord] Deleted leftover recording scratch {path} ({size_mb:.0f} MB)')
    except OSError as e:
        logger.warning(f'[ManualRecord] Could not delete leftover scratch {path}: {e}')


# A controller member that gives up a recording's pending frames, admitted
# and counted by the session's claim (``guarded_write``).
_guarded_write = guarded_write(lambda controller: controller._claim)


class ManualRecordingController:
    """One manual-recording flow: snapshot, record, drain, finish.

    Args:
        scope: The Lumascope instance (frame listener, camera identity,
            exposure, stage position).
        settings: The live settings dict; read ONLY at ``start`` -- the
            recording runs from its immutable snapshot.
        shown_layer: Reads the layer of the session's shown channel, once,
            at ``start``, once the recording holds the scope: it names the
            recording.
        channel_colors: Reads every layer's colour, from the Session's one
            read of it, once, at ``start``.
        activity_claim: The session's compare-and-claim handle; passed
            through to the engine so a recording and a protocol run are
            mutually exclusive.
        scheduler: The session's scheduler; the controller arms its own
            periodic health check on it at ``start`` and disarms at
            finish. Required: a controller that can start a recording
            but has no armed health bound is the trap this parameter
            exists to make unconstructible.
        post_processing: The session's builds; a recording's stack is
            built through it.
        clock: Wall-time source (seconds) for what is measured against
            frame timestamps: the engine's cadence and the camera tick
            rebaser. Tests drive it.
        elapsed_clock: Elapsed-time source (seconds) for the recording's
            bounds: its elapsed time, the duration cap, the stall watch and
            the disk probe's interval. Monotonic, because wall time steps
            (an NTP correction, a host resume) and a step would end a good
            recording early or let one run past its cap. Tests drive it.
    """

    def __init__(
        self,
        *,
        scope: Any,
        settings: dict,
        shown_layer: Callable[[], str],
        channel_colors: Callable[[], layer_record.ChannelColors],
        activity_claim: Any,
        scheduler: Scheduler,
        post_processing: 'PostProcessingAPI',
        clock: Callable[[], float] = time.time,
        elapsed_clock: Callable[[], float] = time.monotonic,
    ):
        self._post_processing = post_processing
        self._scope = scope
        self._settings = settings
        self._shown_layer = shown_layer
        self._channel_colors = channel_colors
        self._claim = activity_claim
        self._scheduler = scheduler
        self._clock = clock
        self._elapsed_clock = elapsed_clock
        self._engine: VideoRecordingEngine | None = None
        self._config: RecordingConfig | None = None
        self._plan: _RecordingPlan | None = None
        self._writer: VideoWriter | None = None
        self._start_ts: float | None = None
        self._stall_watch: StallWatch | None = None
        self._effective_fps: float | None = None
        self._rebaser: CameraTickRebaser | None = None
        self._hyperstack_rows: list | None = None
        self._last_disk_check_ts = 0.0
        self._operation: Operation | None = None
        self._finish_thread: threading.Thread | None = None
        # Whether a recording's finish has yet to end. Its own store, not the
        # finish thread's liveness: the thread announces its end as its last
        # step, while still alive, and a listener told of that end must read
        # it as ended.
        self._finishing = False
        self._health_handle: object | None = None
        # Guards the per-recording state slots published by start():
        # the health check reads them from the scheduler's timer thread,
        # and a check racing a start could otherwise pair the new
        # engine with the previous recording's start timestamp.
        self._state_lock = threading.Lock()
        # Why the most recent recording ended when its engine was dropped:
        # only a start unwound after its commit drops it. Every other end
        # is the engine's own reason.
        self._failed_start_reason: str | None = None

    # ------------------------------------------------------------------
    # Status surface (GUI polls these; all None-safe)
    # ------------------------------------------------------------------

    @api
    @property
    def is_recording(self) -> bool:
        """True while cadence selection is open.

        Raises:
            Nothing.
        """
        return self._engine is not None and self._engine.is_recording

    @api
    @property
    def is_draining(self) -> bool:
        """True while queued frames are still being written.

        Raises:
            Nothing.
        """
        return self._engine is not None and self._engine.is_draining

    @api
    @property
    def pending_writes(self) -> int:
        """Frames enqueued but not yet on disk.

        Raises:
            Nothing.
        """
        return self._engine.pending_writes if self._engine is not None else 0

    @api
    @property
    def is_busy(self) -> bool:
        """True until the recording, its drain, AND the finish complete.

        The app-close gate reads this: the recording is not safely over
        until the post-drain finish (MP4 close, hyperstack) has run.

        Raises:
            Nothing.
        """
        return self.is_recording or self.is_draining or self._finishing

    @api
    @property
    def elapsed_s(self) -> float:
        """Seconds since the recording started; 0.0 when idle.

        Raises:
            Nothing.
        """
        if self._start_ts is None or not self.is_recording:
            return 0.0
        return self._elapsed_clock() - self._start_ts

    @api
    @property
    def save_folder(self) -> Path | None:
        """The active (or last) recording's output folder.

        Raises:
            Nothing.
        """
        return self._plan.save_folder if self._plan is not None else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @api(stop='Operation.stop')
    def start(self, *, false_color_on: bool = False) -> Operation:
        """Snapshot configuration and open a recording; returns its operation.

        Channel IDENTITY (what was imaged) and false-color RENDERING (how
        it is displayed) are independent. Identity is recorded on every
        frame; only rendering is conditional. Collapsing the two is what
        made a brightfield hyperstack impossible to build, since the
        toggle being off left the channel unnamed. The channel is the shown
        channel's layer (``session.show_channel``), read once the recording
        holds the scope, when no light can change and the shown layer cannot
        move; a start that would make the name false is refused.

        Args:
            false_color_on: Whether the shown layer is displayed in false colour.
                Controls rendering alone; grayscale when False.

        Returns:
            The recording's ``Operation``: done once its frames, MP4 close and
            any hyperstack build are over and a new recording may start, with
            the recording's ``end_reason``. Its ``stop()`` stops this
            recording as ``stop`` does, and does nothing once it has stopped,
            so it never stops a recording started later.

        Raises:
            RecordingRefusedError: The start cannot proceed, and nothing is
                committed: ``'recording_active'``, a previous recording is
                still recording, draining or finishing;
                ``'exclusive_activity_running'``, another exclusive activity
                is running; ``'post_processing_running'``, a build is on
                the post-processing lane; ``'falsifying_change_in_flight'``, a
                turret, frame or light change is under way;
                ``'other_led_lit'``, an LED other than the shown layer's is
                lit; ``'camera_holds_another_layer'``, the camera holds
                another layer's settings than the shown one's;
                ``'camera_exposure_unknown'``, the camera has no known
                exposure; ``'capture_location_unusable'``, the save
                location cannot be used; ``'names_exhausted'``, the recording's
                name and every numbered variant are taken there;
                ``'insufficient_disk'``, free disk is
                below the floor.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected; ``'scope_disconnected'`` after
                ``disconnect()``. Nothing is committed.
            ControlSessionRefusedError: ``'control_session_held'``, someone
                else holds the control session. Nothing is committed.
            ObjectiveUnknownError: No one can say which objective is in the
                light path, so the recording's frames would carry no scale.
                Nothing is committed.
            CaptureFailedError: ``'recording_not_started'``, the camera did not
                accept the frame listener; the start is unwound.
        """
        # Exclusivity has to span the finish, not just the drain. The
        # engine frees its claim before the finish thread stops reading
        # this controller's per-recording state, so a second start landing
        # in that gap rebinds every slot underneath a recording that is
        # still closing its encoder and building its hyperstack. Must stay
        # the FIRST statement.
        #
        # Shares the engine's reason code, whose message says "stop it"
        # while this one says "wait". Considered a distinct code; rejected
        # because nothing dispatches on .reason. Revisit if an L2 caller
        # ever branches on it.
        if self.is_busy:
            raise RecordingRefusedError(
                reason='recording_active',
                message=(
                    'A recording is still finishing. Wait for it to complete, then record again.'
                ),
            )
        if self._post_processing.has_work():
            raise RecordingRefusedError(
                reason='post_processing_running', message=POST_PROCESSING_RUNNING_MESSAGE
            )

        settings = self._settings
        scope = self._scope

        scope.camera.refuse_camera_not_connected('start')

        exposure = scope.camera.exposure_ms
        # A camera whose exposure is unknown (never reported, or lost to a
        # failed auto read-back) answers None here; the dead-feed bound
        # derives from it, so refuse loudly instead of fabricating one.
        if exposure is None or exposure <= 0:
            raise RecordingRefusedError(
                reason='camera_exposure_unknown',
                message='The camera has not reported its exposure time, so the '
                'recording cannot be started. Reconnect the camera and try again.',
            )
        video_settings = settings['video']
        max_fps = video_settings['max_fps']
        # Manual recording asks for no rate of its own: it records every
        # frame the camera delivers, limited only by the user's setting
        # when one is set.
        effective_fps = effective_recording_fps(None, max_fps)

        duration_s = get_manual_video_max_duration(settings)
        video_as_frames = settings['video_as_frames']
        capture_config = get_image_capture_config_from_settings(settings)
        camera = scope.capabilities

        start_dt = datetime.datetime.now()
        start_time_str = start_dt.strftime('%Y-%m-%d_%H.%M.%S')
        if video_as_frames:
            save_folder = Path(settings['live_folder']) / 'Manual' / f'Video_{start_time_str}'
        else:
            save_folder = Path(settings['live_folder']) / 'Manual'

        # The live folder is there before its free space means anything: a
        # disk probe on a missing folder raises instead of refusing.
        try:
            live_folder = path_utils.require_capture_location(settings['live_folder'])
        except path_utils.CaptureLocationError as exc:
            raise RecordingRefusedError(
                reason='capture_location_unusable',
                message=str(exc),
            ) from exc

        # PR-2 floor semantics: manual recording is stop-when-I-say, so
        # the pre-flight is a floor check (can we start safely?), not a
        # whole-budget reservation; the rolling check in the write edge
        # carries the guarantee through the recording. Probed on the
        # live folder -- the save subfolders may not exist yet.
        ok, free_mb = check_disk_space_ok(live_folder, MIN_REQUIRED_DISK_MB)
        if not ok:
            raise RecordingRefusedError(
                reason='insufficient_disk',
                message=f'Only {free_mb / 1024:.1f} GB free at the save location; '
                f'at least {MIN_REQUIRED_DISK_MB / 1024:.0f} GB is required to '
                'start a recording. Free up space and try again.',
            )

        if video_as_frames:
            # Reserve the per-recording folder here: after the last refusal
            # that can still turn this press away, and before RecordingConfig
            # freezes output_dir, so the config carries the name actually
            # taken. Two presses inside one second derive the same name, and
            # joining the first folder mixes both recordings' frames under one
            # manifest with indices that restart.
            try:
                # Create the Manual/ level -- LVP's own subfolder, absent until
                # the first recording -- but NOT its parents: if the configured
                # live folder itself is gone, that is an unplugged drive or a
                # stale path, and the allocator refuses rather than building a
                # fresh tree on whatever is mounted there.
                save_folder.parent.mkdir(exist_ok=True)
                save_folder = path_utils.allocate_directory(save_folder)
            except OSError as exc:
                raise RecordingRefusedError(
                    reason='capture_location_unusable',
                    message=(
                        f'{save_folder.parent.parent} is not an accessible save '
                        'location. Check that it exists and any external drive '
                        'is connected, then try again.'
                    ),
                ) from exc
            except path_utils.CaptureLocationError as exc:
                # The location's own reason, kept: every numbered name taken
                # is a different fact from a location that is not there.
                raise RecordingRefusedError(reason=exc.reason, message=str(exc)) from exc

        manifest_extra = {
            # As a separate RENDERING fact from the channel the frames are
            # of, whether they were shown in false colour: the video builder
            # colours a rebuild only when it was on, so frames the user
            # deliberately saved gray stay gray, in the colour of the day.
            # The channel is set once the recording holds the scope.
            'false_color': false_color_on,
            'camera': {
                'model': camera.camera_model,
                'serial': camera.camera_serial_number,
                'timestamp_tick_hz': camera.camera_timestamp_tick_hz,
            },
            'provenance': {
                'host': gather_host_provenance(),
                'software': {'lvp_version': lvp_version},
            },
        }
        hyperstack = (
            video_as_frames
            and capture_config.output_format_sequenced == image_mode.OutputFormat.HYPERSTACK
        )
        # What the file will claim of the scope -- its frame size, its
        # objective, its plate and the camera's facts -- is read by the engine
        # once it holds the claim: a turret or frame change can neither start
        # then nor still be running, so the file cannot claim the scope as it
        # was an instant before one.
        claimed = {}

        def _make_config() -> RecordingConfig:
            claimed['layer'] = layer = self._shown_layer()
            _refuse_a_name_that_would_be_false(scope, layer)
            manifest_extra['channel'] = layer
            # The size the file is made at: held, else read, so an unknown one is
            # asked for (CameraReadFailedError when the camera does not answer).
            frame_size = scope.camera.frame_size or scope.camera.read_frame_size()
            claimed['frame_size'] = frame_size
            # The objective cannot change while a recording runs; an unknown
            # one refuses the start, as it refuses a capture: a file with no
            # objective would carry no scale while the scope went on with one
            # the person had not confirmed.
            claimed['objective'], _ = scope.runtime_state.resolve_current_objective()
            claimed['plate'] = scope.runtime_state.get_labware() if video_as_frames else None
            claimed['to_plate'] = scope.runtime_state.plate_transform() if video_as_frames else None
            claimed['camera'] = recording_camera(scope) if video_as_frames else None
            claimed['config'] = RecordingConfig(
                fps=effective_fps,
                duration_s=duration_s,
                width=frame_size.width,
                height=frame_size.height,
                bit_depth=capture_config.capture_depth,
                output_dir=save_folder,
                filename_template=manual_frame_filename_template(),
                timestamp_overlay=video_settings['timestamp_overlay'],
                manifest_extra=manifest_extra,
                # The MP4 leg's artifacts share the flat Manual folder, so its
                # manifest is named after the video the writer actually wrote --
                # the writer renames itself on collision, so two recordings
                # inside one second write two videos, and a manifest named from
                # the start timestamp would describe whichever one it did not
                # measure. The frames leg owns a per-recording folder and keeps
                # the default name.
                manifest_filename=('recording_manifest.json' if video_as_frames else None),
            )
            return claimed['config']

        engine = VideoRecordingEngine(
            write_frame=self._write_frame,
            claim=self._claim,
            clock=self._clock,
        )
        # The frames leg's folder, reserved above; a start that fails from
        # here on leaves it holding nothing.
        frames_folder = save_folder if video_as_frames else None

        # Its stop is this recording's own engine's, never the controller's
        # current one: a late stop reaches an engine already closed, which
        # keeps the reason it closed with. The frame listener stays until
        # the finish removes it; it offers nothing to a closed engine.
        operation = Operation(stop=lambda: engine.stop('user_stop'))

        # Engine start is the commit point: it acquires the claim or
        # raises. Assign the rest of the controller's state only after it
        # succeeds. The engine itself is stored first: it announces the
        # recording live from inside start, and a listener told of that
        # edge reads is_recording through this controller. A start that
        # raises puts back the engine that was there.
        with self._state_lock:
            previous_engine, self._engine = self._engine, engine
        try:
            engine.start(_make_config)
        except BaseException:
            with self._state_lock:
                self._engine = previous_engine
            _discard_if_empty(frames_folder)
            raise
        writer = None
        try:
            config = claimed['config']
            plan = _RecordingPlan(
                video_as_frames=video_as_frames,
                save_folder=save_folder,
                layer=claimed['layer'],
                false_color_on=false_color_on,
                channel_colors=self._channel_colors(),
                save_encoding=capture_config.save_encoding,
                capture_depth=capture_config.capture_depth,
                camera=claimed['camera'],
                hyperstack=hyperstack,
                objective_id=claimed['objective'],
                labware=claimed['plate'],
                to_plate=claimed['to_plate'],
            )
            if video_as_frames:
                _say_what_the_frames_cannot_record(scope, plan.to_plate)
            else:
                save_folder.mkdir(exist_ok=True, parents=True)
                writer = VideoWriter(
                    output_path=save_folder / f'Video_{start_time_str}.mp4',
                    fps=effective_fps,
                    width=claimed['frame_size'].width,
                    height=claimed['frame_size'].height,
                    # Rendering: grey keeps the encoder gray, which is what a
                    # recording with the toggle off must produce.
                    color=layer_record.shown_color(plan.layer, false_color_on, plan.channel_colors),
                    include_timestamp_overlay=config.timestamp_overlay,
                    vfr=True,
                )
            with self._state_lock:
                self._config = config
                self._plan = plan
                self._writer = writer
                self._start_ts = self._elapsed_clock()
                self._effective_fps = effective_fps
                self._stall_watch = StallWatch(self._stall_threshold_now(effective_fps, exposure))
                self._rebaser = CameraTickRebaser(camera.camera_timestamp_tick_hz, self._clock)
                self._hyperstack_rows = [] if hyperstack else None
                self._last_disk_check_ts = 0.0
                self._operation = operation
                self._failed_start_reason = None
            # The frames folder was reserved before the commit point; nothing
            # to create here.

            scope.imaging.add_frame_listener(self._on_camera_frame, name='manual_recording')
            self._finish_thread = threading.Thread(
                target=self._finish_after_drain, name='ManualRecordingFinish', daemon=True
            )
            self._finishing = True
            self._finish_thread.start()
            # Armed last, after every state slot above is published: a
            # check firing between engine.start and the publish would
            # read the previous recording's start timestamp against the
            # new engine and stop the fresh recording as expired.
            self._disarm_health_check()
            self._health_handle = self._scheduler.schedule_interval(
                self._health_check, HEALTH_CHECK_INTERVAL_S
            )
        except FrameListenerNotRegisteredError as refused:
            self._unwind_failed_start(engine, writer, frames_folder)
            # Said as what the person pressed Record for: the recording, not
            # the listener it would have needed.
            raise CaptureFailedError('recording_not_started') from refused
        except BaseException:
            # Past the commit point the engine holds the claim, and this is
            # the only frame holding the writer -- the engine is handed a
            # write_frame callable and can never close it.
            self._unwind_failed_start(engine, writer, frames_folder)
            raise
        rate = (
            'every delivered frame' if effective_fps is None else f'{effective_fps:.2f} fps limit'
        )
        logger.info(
            f'[ManualRecord] Recording started: {rate}, '
            f'max {duration_s:.0f} s, {"frames" if video_as_frames else "mp4"} '
            f'-> {save_folder}'
        )
        return operation

    def _unwind_failed_start(
        self, engine: VideoRecordingEngine, writer: Any, frames_folder: Path | None
    ) -> None:
        """Undo a start that raised after the engine committed.

        Order matters: stop delivering frames, then end the recording so
        the writer lane drains and frees the claim, and only then dispose
        the artifact -- disposing first would leave the lane writing into
        a closed encoder and count those frames as saved.
        """
        # Idempotent no-op when the failure preceded the arming.
        self._disarm_health_check()
        self._failed_start_reason = END_REASON_START_FAILED
        # The listener registers late in start(), so an earlier failure
        # unwinds without one ever having been added: the removal is then a
        # no-op.
        self._scope.imaging.remove_frame_listener(self._on_camera_frame)

        # Deliberately does not wait for the drain: the lane only has to pop
        # a sentinel to exit and free the claim, and blocking here would run
        # on the single-worker camera executor, so a wedged writer would take
        # the executor down with it rather than just this recording.
        try:
            engine.stop(END_REASON_START_FAILED)
        except Exception:
            logger.exception('[ManualRecord] engine did not end cleanly during start unwind')

        if writer is not None:
            try:
                writer.close()
                if writer.frame_count == 0:
                    # A start that never completed leaves an empty container
                    # behind; the next recording's collision resolver would
                    # otherwise rename around a file holding no frames.
                    writer.output_path.unlink(missing_ok=True)
            except Exception:
                logger.exception('[ManualRecord] writer disposal failed during start unwind')
        _discard_if_empty(frames_folder)

        self._engine = None
        self._writer = None
        self._finish_thread = None
        self._finishing = False

    @api
    def stop(self) -> None:
        """Close selection; the drain and finish continue on their own.

        Every caller of this -- the Record button, app close, an L2 client
        -- is the user stopping the recording, so its ``end_reason`` is
        ``'user_stop'``. The controller's own enders (the duration cap, the
        disk floor, a lost camera) name their own, and no caller can write
        one of those into the manifest.

        Raises:
            Nothing.
        """
        self._end('user_stop')

    def stop_the_recording_taken_for(self, session: ControlSession) -> None:
        """Stop the recording when it was taken for *session*; nothing otherwise.

        For the session ending a control session. A recording that has
        ended since, and the next one -- the operator's, admitted once the
        session ended -- are not it. Ends as ``stop`` does, ``'user_stop'``.
        """
        self._end('user_stop', session)

    def _end(self, reason: str, taken_for: ControlSession | None = None) -> None:
        """Close selection as *reason*, the manifest's ``end_reason``; with *taken_for*, only that session's recording."""
        engine = self._engine
        if engine is None or not engine.stop(reason, taken_for=taken_for):
            return
        logger.info(f'[ManualRecord] stopped by {current_caller().words} ({reason})')
        # After the stop: a closed engine takes no frame, so a frame that
        # lands between the two is dropped, as one after the removal is.
        self._scope.imaging.remove_frame_listener(self._on_camera_frame)

    @api
    @property
    def end_reason(self) -> str | None:
        """Why the most recent recording ended, or None before any has.

        Manifest vocabulary ('user_stop', 'duration_elapsed',
        'camera_stalled', ...), read from the recording's engine as soon
        as its selection closes, so a caller watching is_recording flip
        False can tell a camera death from a user stop while the frames
        still drain. The first close names it: a later stop does not
        rewrite it.

        Raises:
            Nothing.
        """
        engine = self._engine
        if engine is not None and not engine.is_recording:
            return engine.end_reason
        return self._failed_start_reason

    def _health_check(self) -> None:
        """Watch the recording's health; armed by the controller itself.

        Owns the elapsed-time duration cap (the engine's frame budget is
        frame-driven, so a dead feed would never fill it) and both
        camera-death checks: the disconnect latch, and the stall watch
        for a feed that dies without an event -- delivery stops while
        the camera still reads active. start() schedules this on the
        session's scheduler and the post-drain finish disarms it, so
        every host -- GUI or headless -- carries the same bounds. The
        state snapshot is taken under the lock, which is released
        before any stop or notification runs.
        """
        with self._state_lock:
            engine = self._engine
            config = self._config
            start_ts = self._start_ts
            stall_watch = self._stall_watch
            effective_fps = self._effective_fps
        if engine is None or not engine.is_recording or start_ts is None or config is None:
            return
        if self._elapsed_clock() - start_ts >= config.duration_s:
            self._end('duration_elapsed')
            return
        if not self._scope.camera.active:
            self._stop_for_camera_loss(reason='camera_disconnected')
            return
        if stall_watch is None:
            return
        # Exposure and gain stay open during a recording, so the interval
        # between frames can grow after start: the bound follows it.
        stall_watch.raise_threshold(self._stall_threshold_now(effective_fps))
        if stall_watch.stalled(engine.frames_selected, self._elapsed_clock()):
            self._stop_for_camera_loss(reason='camera_stalled')

    def _stall_threshold_now(self, effective_fps: float | None, floor_ms: float = 0.0) -> float:
        """The feed-death bound for the longest exposure the camera may be using now."""
        exposure_ms = max(floor_ms, self._scope.camera.longest_exposure_ms or 0.0)
        return stall_threshold_s(effective_fps, exposure_ms / 1000.0)

    def _disarm_health_check(self) -> None:
        handle = self._health_handle
        self._health_handle = None
        if handle is not None:
            self._scheduler.unschedule(handle)

    def _stop_for_camera_loss(self, reason: str) -> None:
        # The person reads one sentence for both reasons; the log keeps which.
        logger.info(f'[ManualRecord] Camera feed lost mid-recording ({reason})')
        self._end(reason)
        notifications.report_outcome(
            RecordingStoppedError(reason), solicited=False, category='Recording'
        )

    @api(admitted_by=(Admission.GUARDED_WRITE,))
    @_guarded_write
    def discard_pending(self) -> None:
        """Drop the unwritten backlog loudly (the app-close discard path).

        Raises:
            Nothing.
        """
        if self._engine is not None:
            self._engine.discard_pending()

    # ------------------------------------------------------------------
    # Camera-thread ingest
    # ------------------------------------------------------------------

    def _on_camera_frame(self, frame: Frame) -> None:
        """SDK-thread listener: rebase the timestamp, offer to the engine.

        The frame's facts -- where it was, what lit it -- are read HERE,
        when the frame arrives, and ride the queue with it: the write runs
        later, behind the backlog, and a read then would put a stage move
        or an LED change on the wrong frames. Only a
        frames recording writes a per-frame file, so only it pays.
        """
        engine = self._engine
        if engine is None or not engine.is_recording:
            return
        plan = self._plan
        chunks = dict(frame.chunks) if frame.chunks is not None else None
        timestamp_s, source = self._rebaser.frame_time_s(frame.timestamp, chunks)
        facts = (
            arrival(
                self._scope,
                frame,
                frame_time=FrameTime(
                    at=datetime.datetime.fromtimestamp(timestamp_s), source=source
                ),
                channel_tiebreak=plan.layer,
                to_plate=plan.to_plate,
                camera=plan.camera,
            )
            if plan.video_as_frames
            else None
        )
        engine.ingest_frame(frame.image, timestamp_s, chunks, fact=facts)

    # ------------------------------------------------------------------
    # Writer-lane edge
    # ------------------------------------------------------------------

    def _write_frame(self, image, timestamp_s, frame_number, config, chunks, facts) -> Path:
        """Write one kept frame as its final artifact (runs on the lane)."""
        self._check_disk_floor(config)

        image = orient_and_fit(image, config.width, config.height)

        plan = self._plan
        if plan.video_as_frames:
            return self._write_tiff_frame(image, timestamp_s, frame_number, config, chunks, facts)
        return self._write_mp4_frame(image, timestamp_s)

    def _write_tiff_frame(self, image, timestamp_s, frame_number, config, chunks, facts) -> Path:
        plan = self._plan
        fact, light, _record = facts
        if config.bit_depth == 8 and image.dtype != np.uint8:
            image = image_utils.convert_to_8bit(image, config.bit_depth)

        file_loc = config.output_dir / config.filename_template.format(
            n=frame_number, ts=filename_time(facts)
        )
        write_frame(
            self._scope,
            image,
            file_loc,
            facts,
            frame_number=frame_number,
            objective_id=plan.objective_id,
            labware=plan.labware,
            fallback_focus=None,
            false_color_on=plan.false_color_on,
            channel_colors=plan.channel_colors,
            save_encoding=plan.save_encoding,
        )

        if self._hyperstack_rows is not None:
            # 'Scan Count' is the T-axis ordinal per the execution-record
            # contract; within one recording the temporal ordinal IS the
            # frame number (this dataframe never mixes with scan-indexed
            # rows -- it feeds only the per-recording hyperstack build).
            # The row is THIS frame's fact, the same one its file carries:
            # the stage and the LEDs are open to other callers while a
            # manual recording runs, so a start-time snapshot would be a
            # claim about frames it never saw.
            # The position as the frame's file states it, to its precision.
            position = image_save.position_metadata_fields(
                fact.plate_x_mm, fact.plate_y_mm, fact.z_um
            )
            plate_pos_mm = position.get('plate_pos_mm')
            self._hyperstack_rows.append(
                {
                    'Filepath': file_loc.name,
                    'Scan Count': frame_number,
                    # Channel identity: what was imaged. Independent of the
                    # false-color toggle, which governs display only, so it
                    # is recorded on every frame regardless of rendering.
                    'Color': light.channel,
                    'Z-Slice': 0,
                    'X': plate_pos_mm['x'] if plate_pos_mm else None,
                    'Y': plate_pos_mm['y'] if plate_pos_mm else None,
                    'Z': position.get('z_pos_um'),
                }
            )
        return file_loc

    def _write_mp4_frame(self, image, timestamp_s) -> Path:
        writer = self._writer
        significant_bits = self._plan.capture_depth if image.dtype != np.uint8 else None
        writer.add_frame(image=image, timestamp=timestamp_s, significant_bits=significant_bits)
        return writer.output_path

    def _check_disk_floor(self, config) -> None:
        """Rolling floor probe; a breach stops selection, never the drain.

        Runs on the write lane -- the thread consuming disk -- so the
        probe needs no scheduler and works headless. Stopping selection
        yields an honest short delivery; frames already on disk stay.
        """
        now = self._elapsed_clock()
        if now - self._last_disk_check_ts < DISK_FLOOR_CHECK_INTERVAL_S:
            return
        self._last_disk_check_ts = now
        try:
            ok, free_mb = check_disk_space_ok(config.output_dir, MIN_REQUIRED_DISK_MB)
        except Exception as e:
            logger.warning(f'[ManualRecord] Disk-floor probe failed: {e}')
            return
        if not ok and self._engine is not None and self._engine.is_recording:
            logger.info(
                f'[ManualRecord] Free disk fell to {free_mb:.0f} MB (floor '
                f'{MIN_REQUIRED_DISK_MB} MB)'
            )
            notifications.report_outcome(
                RecordingStoppedError('disk_floor'), solicited=False, category='Recording'
            )
            self._end('disk_floor')

    # ------------------------------------------------------------------
    # Post-drain finish (its own short-lived thread, one per recording)
    # ------------------------------------------------------------------

    def _finish_after_drain(self) -> None:
        """Wait out the drain, then finish the artifacts and report.

        Runs on the per-recording finish thread: the work here (MP4
        close, hyperstack build) is too heavy for a UI poll and must not
        occupy the shared FILE lane for the length of a drain tail.
        """
        engine = self._engine
        operation = self._operation
        engine.wait_for_drain()
        # The recording is over on every path that gets here (the drain
        # completes only after selection closes); the health check has
        # nothing left to watch. Disarmed here rather than in stop()
        # because the dominant end -- the frame budget filling inside
        # the frame callback -- never goes through stop().
        self._disarm_health_check()
        self._scope.imaging.remove_frame_listener(self._on_camera_frame)

        result = None
        writer_dropped = 0
        try:
            # Close the encoder before reading measured truth. An unclosed
            # container is a corrupt file on disk, while the result is only a
            # report -- and result() raises when the engine never finished
            # finalizing, which is exactly when the close matters most.
            if self._writer is not None:
                self._writer.close()
                # The MP4 writer swallows per-frame encode errors into its
                # own counter; fold them into the user-facing total. The
                # manifest carries the engine-counted failures -- the
                # frames+manifest leg is the reference artifact.
                writer_dropped = self._writer.dropped_frames

            result = engine.result()

            if self._hyperstack_rows is not None:
                self._build_hyperstack()
        except HyperstackRefusedError as refused:
            # The builder's reason is the whole answer, in its own words:
            # "check the log" is no answer to a REST caller or to a user who
            # cannot read one. The frames are on disk as recorded.
            notifications.report_outcome(refused, solicited=False, category='Recording')
        except Exception as failed:
            fault = RecordingFinalizeError(protocol_step=False)
            fault.__cause__ = failed
            notifications.report_outcome(fault, solicited=False, category='Recording')
        finally:
            # Reporting needs the measured truth; completing the recording
            # does not. Keeping them apart is what lets the operation end --
            # and the UI leave its recording state -- after a failed finish.
            if result is not None:
                if result.writer_failure is not None:
                    died = VideoWriterFailedError(protocol_step=False)
                    died.__cause__ = result.writer_failure
                    notifications.report_outcome(died, solicited=False, category='Recording')
                if result.manifest_failure is not None:
                    unsaved = RecordingDetailsNotSavedError()
                    unsaved.__cause__ = result.manifest_failure
                    notifications.report_outcome(
                        unsaved, solicited=False, category='Video Recording'
                    )
                # Announce the artifact only once its existence is known.
                # A recording that captured nothing leaves NO file at all --
                # the mp4 muxer writes neither header nor trailer for an
                # empty stream -- so naming output_path here would print a
                # path that was never created. frames_written is the same
                # authority the row-recording branches use, rather than a
                # second opinion from a filesystem probe.
                if result.frames_written == 0:
                    logger.info(
                        '[ManualRecord] No video written: the recording captured 0 '
                        'frames, so no file was produced'
                    )
                elif self._writer is not None:
                    logger.info(f'[ManualRecord] Video written to {self._writer.output_path}')

                dropped = result.write_failures + writer_dropped
                if dropped > 0 and not result.aborted:
                    notifications.report_outcome(
                        VideoFramesDroppedError(
                            dropped, result.frames_selected, protocol_step=False
                        ),
                        solicited=False,
                        category='Recording',
                    )
                logger.info(
                    f'[ManualRecord] Finished: {result.frames_written} written, '
                    f'{result.write_failures} failed, measured '
                    f'{result.measured_fps:.2f} fps over {result.measured_duration_s:.2f} s'
                )
            end_reason = engine.end_reason
            # Once it reads False a new recording may start and rebind every
            # per-recording slot this thread reads above. The claim was
            # released when the drain ended, so this end is no transition of
            # the claim's and is announced on its own. The operation is done
            # last, so whoever reads its result may start the next recording.
            self._finishing = False
            self._claim.announce()
            concurrent.futures.Future.set_result(operation, end_reason)

    def _build_hyperstack(self) -> None:
        plan = self._plan
        df = pd.DataFrame(self._hyperstack_rows)
        output = plan.save_folder / MANUAL_HYPERSTACK_FILENAME
        try:
            result = self._post_processing.recording_hyperstack(
                df,
                plan.save_folder,
                output,
                save_encoding=plan.save_encoding,
                channel_colors=plan.channel_colors,
            )
        except InputFramesUnusableError as unusable:
            # Frames the stack cannot use are a hyperstack the recording did
            # not get, told under that heading in the builder's words.
            raise HyperstackRefusedError(
                str(unusable), _HYPERSTACK_REASON_OF[unusable.reason]
            ) from unusable
        # The builder reports refusal in its return value rather than by
        # raising. Raise it as the typed refusal so the finish handler tells
        # the user the builder's reason; announcing a hyperstack the builder
        # declined to write would name a file that does not exist.
        if not result['status']:
            raise HyperstackRefusedError(result['error'])
        logger.info(f'[ManualRecord] Hyperstack created at {output}')


# Each reason a recording's frames cannot be used, as the hyperstack refusal's.
_HYPERSTACK_REASON_OF = {
    'unreadable': 'unreadable_input_frame',
    'mixed_pixel_types': 'mixed_input_pixel_types',
    'geometry': 'hyperstack_refused',
}
if set(_HYPERSTACK_REASON_OF) != set(InputFramesUnusableError._TITLES):
    raise ImportError('every InputFramesUnusableError reason needs a hyperstack refusal reason')


def _say_what_the_frames_cannot_record(scope, to_plate) -> None:
    """Tell the user, once at start, what a frames recording cannot yet record.

    Each frame records the position the scope knows when the frame
    arrives, so an axis unknown now is recorded from the moment it is
    known -- a home during the recording makes the later frames say where
    they were. The user still hears it once, in the axis's own name,
    because a file with no position for its first frames is a surprise
    without it. The turret is not part of the position. With no labware
    or stage offset selected there is no plate frame to state X and Y in,
    and no home will change that, so that is said too.
    """
    unknown = [ax for ax in StageAxis if ax in scope.motion.axes_without_position()]
    if not unknown and to_plate is not None:
        return
    notifications.report_outcome(
        RecordingPositionNotRecordedNotice(unknown_axes=unknown, has_plate=to_plate is not None),
        solicited=False,
        category='Recording',
    )


def _refuse_a_name_that_would_be_false(scope, layer: str) -> None:
    """Refuse a recording named for ``layer`` that would be lit or set for another.

    Read once the recording holds the scope: from then on no light can be
    turned on and the shown layer cannot change, so what is read here holds
    for every frame. A recording is named for the shown layer, so another
    LED lit would image the frames in a channel they are not named for, and
    a camera holding another layer's settings would shoot them at settings
    that are not the named layer's. A camera that holds no layer whole (a
    raw camera write came after one) names no other layer.

    Raises:
        RecordingRefusedError: ``'other_led_lit'``, an LED other than
            ``layer``'s is lit; ``'camera_holds_another_layer'``, the camera
            holds another layer's settings.
    """
    readings = scope.illumination.get_led_states() or ()
    lit = {address for address, _ma in snapshot_lit_pairs(readings)}
    others = [
        r.name or f'LED address {r.address}'
        for r in readings
        if r.address in lit and r.name != layer
    ]
    if others:
        raise RecordingRefusedError(
            reason='other_led_lit',
            message=(
                f'{", ".join(others)} is lit while {layer} is shown, and a recording is '
                'named for the shown layer. Turn it off, or show its layer, then record.'
            ),
        )
    held = scope.camera.get_held_layer()
    if held is not None and held != layer:
        raise RecordingRefusedError(
            reason='camera_holds_another_layer',
            message=(
                f"The camera holds {held}'s settings while {layer} is shown. Show "
                f'{layer} again to put its settings on the camera, then record.'
            ),
        )


def _discard_if_empty(frames_folder: Path | None) -> None:
    """Remove the frames leg's reserved folder after a start that failed.

    The folder is reserved before the engine's commit point, so a start
    that fails afterwards leaves an empty ``Video_<timestamp>`` folder in
    the user's Manual folder, reading as a recording that holds nothing --
    the mp4 leg already deletes its empty container for the same reason.
    Only an empty folder is removed: anything written into it is kept.
    """
    if frames_folder is None:
        return
    try:
        frames_folder.rmdir()
    except FileNotFoundError:
        return
    except OSError as e:
        logger.warning(f'[ManualRecord] kept {frames_folder} after a failed start: {e}')


@dataclass(frozen=True)
class _RecordingPlan:
    """Frozen caller-side snapshot the write edges read (leg, encoding,
    identity); the engine's RecordingConfig carries the engine-visible
    half of the same snapshot."""

    video_as_frames: bool
    save_folder: Path
    # Resolved at start() and never None, so no write edge has to decide
    # what an unnamed channel means.
    layer: str
    false_color_on: bool
    # Every layer's colour as the recording began: each frame is coloured
    # as the channel that lit it, so one colour would not do.
    channel_colors: layer_record.ChannelColors
    save_encoding: str
    capture_depth: int
    # The camera facts every frame shares, read at start(); None when the
    # recording saves no frame files.
    camera: RecordingCamera | None
    hyperstack: bool
    # Resolved once at start(), not per frame: the objective cannot change
    # mid-recording, and a per-frame resolve would let one stack hold frames
    # that disagree about their own scale.
    objective_id: str
    # The plate the frames' positions are stated on, read at start(); None
    # when the recording saves no frame files.
    labware: Labware | None
    # The plate transform bound at start(), so every frame is stated in the
    # frame of reference the recording began in; None when the recording
    # saves no frame files, and frames record no plate position.
    to_plate: Callable[[float, float], PlatePosition] | None
