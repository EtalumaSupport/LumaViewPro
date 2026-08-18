# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Manual video recording: the controller between callers and the engine.

Drives ``VideoRecordingEngine`` for manual recordings (the GUI Record
button today, REST/SDK callers tomorrow). The controller owns everything
caller-shaped the engine deliberately does not: reading settings into
the immutable ``RecordingConfig`` snapshot, building the per-leg write
edge (TIFF-per-frame or VFR MP4), registering the camera frame listener,
the rolling disk-floor stop, the wall-clock duration cap, and the
post-drain finish (MP4 close, optional hyperstack, drop notification).

GUI-agnostic by construction: no Kivy imports; the caller polls status
properties for titles/buttons and passes ``on_complete`` for its own
cleanup dispatch.

Timing axis: the engine receives host-epoch seconds. When the camera
reports hardware timestamp ticks and a tick frequency, per-frame times
are the camera's own clock rebased onto the host epoch at the first
frame -- camera-grade intervals on the host axis, which cadence
selection and the VFR pts both need. Without usable ticks the host
arrival time is used and the manifest's timestamp grade reports it.
"""

import datetime
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import modules.image_mode as image_mode
import modules.image_save as image_save
import modules.image_utils as image_utils
import modules.path_utils as path_utils
from lvp_logger import logger, version as lvp_version
from modules.common_utils import (
    DISK_FLOOR_CHECK_INTERVAL_S,
    MIN_REQUIRED_DISK_MB,
    check_disk_space_ok,
)
from modules.config_helpers import (
    get_image_capture_config_from_settings,
    get_manual_video_max_duration,
)
from modules.exceptions import RecordingRefusedError
from modules.notification_center import notifications
from modules.recording_frames import (
    MANUAL_HYPERSTACK_FILENAME,
    CameraTickRebaser,
    manual_frame_filename_template,
    orient_and_fit,
    resolve_recording_pixel_size,
    tiff_frame_metadata,
)
from modules.recording_manifest import gather_host_provenance
from modules.stack_builder import StackBuilder
from modules.video_cadence import StallWatch, effective_recording_fps, stall_threshold_s
from modules.video_recording import (
    END_REASON_START_FAILED,
    RecordingConfig,
    VideoRecordingEngine,
)
from modules.video_writer import VideoWriter

# The channel LumaViewPro comes up on. A recording that can name no other
# channel was taken on this one, so identity always has a real value and
# nothing downstream has to represent a channel-less recording.
DEFAULT_LAYER = 'BF'


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


class ManualRecordingController:
    """One manual-recording flow: snapshot, record, drain, finish.

    Args:
        scope: The Lumascope instance (frame listener, camera identity,
            exposure, stage position).
        settings: The live settings dict; read ONLY at ``start`` -- the
            recording runs from its immutable snapshot.
        activity_claim: The session's compare-and-claim handle; passed
            through to the engine so a recording and a protocol run are
            mutually exclusive.
        clock: Injectable time source (seconds); tests drive it.
    """

    def __init__(self, *, scope: Any, settings: dict, activity_claim: Any, clock=time.time):
        self._scope = scope
        self._settings = settings
        self._claim = activity_claim
        self._clock = clock
        self._engine: VideoRecordingEngine | None = None
        self._config: RecordingConfig | None = None
        self._plan: _RecordingPlan | None = None
        self._writer: VideoWriter | None = None
        self._start_ts: float | None = None
        self._stall_watch: StallWatch | None = None
        self._rebaser: CameraTickRebaser | None = None
        self._hyperstack_rows: list | None = None
        self._last_disk_check_ts = 0.0
        self._on_complete: Callable[[], None] | None = None
        self._finish_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Status surface (GUI polls these; all None-safe)
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """True while cadence selection is open."""
        return self._engine is not None and self._engine.is_recording

    @property
    def is_draining(self) -> bool:
        """True while queued frames are still being written."""
        return self._engine is not None and self._engine.is_draining

    @property
    def pending_writes(self) -> int:
        """Frames enqueued but not yet on disk."""
        return self._engine.pending_writes if self._engine is not None else 0

    @property
    def is_busy(self) -> bool:
        """True until the recording, its drain, AND the finish complete.

        The app-close gate reads this: the recording is not safely over
        until the post-drain finish (MP4 close, hyperstack) has run.
        """
        thread = self._finish_thread
        return self.is_recording or self.is_draining or (thread is not None and thread.is_alive())

    @property
    def elapsed_s(self) -> float:
        """Seconds since the recording started; 0.0 when idle."""
        if self._start_ts is None or not self.is_recording:
            return 0.0
        return self._clock() - self._start_ts

    @property
    def save_folder(self) -> Path | None:
        """The active (or last) recording's output folder."""
        return self._plan.save_folder if self._plan is not None else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _resolve_channel_identity(self, layer: str | None) -> str:
        """The channel this recording imaged: lit LED, else open layer, else BF.

        A lit LED is direct evidence of what is illuminating the sample,
        so it outranks the open accordion -- a channel can be lit while a
        different layer's drawer is open, and the light is the truth.

        The open layer is the tiebreak for luminescence, which images with
        no LED lit at all and would otherwise be unnameable.

        get_led_states() returns {} when no LED board is present, so a
        board-less scope falls through to the layer with no special case.
        """
        for color, state in self._scope.illumination.get_led_states().items():
            if state.get('enabled'):
                return color
        return layer or DEFAULT_LAYER

    def start(
        self,
        *,
        layer: str | None = None,
        false_color_on: bool = False,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Snapshot configuration and open a recording.

        Channel IDENTITY (what was imaged) and false-color RENDERING (how
        it is displayed) are independent. Identity is recorded on every
        frame; only rendering is conditional. Collapsing the two is what
        made a brightfield hyperstack impossible to build, since the
        toggle being off left the channel unnamed.

        Args:
            layer: The layer whose accordion is open, or None. Used only
                when no LED is lit -- see _resolve_channel_identity.
            false_color_on: Whether that layer's false-color toggle is on.
                Controls rendering alone; grayscale when False.
            on_complete: Invoked (on the finish thread) after the drain
                and all post-drain work; the caller dispatches its own
                UI cleanup from it.

        Raises:
            RecordingRefusedError: The start cannot proceed -- a previous
                recording is still recording, draining or finishing,
                another exclusive activity is running, the camera is
                inactive or has no known exposure, or free disk is below
                the floor. Nothing is committed when this raises.
        """
        # Exclusivity has to span the finish, not just the drain. The
        # engine frees its claim before the finish thread stops reading
        # this controller's per-recording state, so a second start landing
        # in that gap rebinds every slot underneath a recording that is
        # still closing its encoder and building its hyperstack. Must stay
        # the FIRST statement: the refusals below are not side-effect-free
        # (the rate clamp pops an FPS-budget warning).
        #
        # Shares the engine's reason code, whose message says "stop it"
        # while this one says "wait". Considered a distinct code; rejected
        # because nothing dispatches on .reason. Revisit if an L2 caller
        # ever branches on it.
        if self.is_busy:
            raise RecordingRefusedError(
                reason='recording_active',
                title='Recording Active',
                message=(
                    'A recording is still finishing. Wait for it to complete, then record again.'
                ),
            )

        settings = self._settings
        scope = self._scope

        if not scope.imaging.camera_active:
            raise RecordingRefusedError(
                reason='camera_inactive',
                title='Camera Not Active',
                message='The camera is not streaming, so there is nothing to record. '
                'Check the camera connection and try again.',
            )

        exposure = scope.imaging.camera_exposure_ms
        # The exposure cache seeds 0.0 and keeps the prior value when a
        # read fails, so a camera whose exposure was never successfully
        # read reports 0 here; the recording rate derives from it, so
        # refuse loudly instead of fabricating a rate.
        if exposure is None or exposure <= 0:
            raise RecordingRefusedError(
                reason='camera_exposure_unknown',
                title='Camera Exposure Unavailable',
                message='The camera has not reported its exposure time, so the '
                'recording rate cannot be set. Reconnect the camera and try again.',
            )
        exposure_fps = 1000.0 / exposure

        video_settings = settings.get('video', {})
        max_fps = video_settings.get('max_fps', 0)
        if max_fps > 0 and max_fps > exposure_fps:
            notifications.warning(
                'Recording',
                'FPS budget exceeded',
                f'Requested {max_fps:.1f} FPS at {exposure:.0f} ms exposure exceeds '
                f"the camera's max {exposure_fps:.1f} FPS for that exposure. "
                f'Recording will run at {exposure_fps:.1f} FPS instead. '
                'Reduce exposure to hit the requested rate.',
            )
        # The effective-rate clamp: exposure bounds what the sensor can
        # produce, the user's cap applies when set, and the delivery
        # bound caps an uncapped fast-exposure config -- the frame budget
        # must reflect a rate the camera can actually deliver, never a
        # bare 1/exposure.
        effective_fps = effective_recording_fps(exposure_fps, max_fps)

        duration_s = get_manual_video_max_duration(settings)
        video_as_frames = settings['video_as_frames']
        capture_config = get_image_capture_config_from_settings(settings)
        identity = scope.imaging.camera_identity

        start_dt = datetime.datetime.now()
        start_time_str = start_dt.strftime('%Y-%m-%d_%H.%M.%S')
        if video_as_frames:
            save_folder = Path(settings['live_folder']) / 'Manual' / f'Video_{start_time_str}'
        else:
            save_folder = Path(settings['live_folder']) / 'Manual'

        # PR-2 floor semantics: manual recording is stop-when-I-say, so
        # the pre-flight is a floor check (can we start safely?), not a
        # whole-budget reservation; the rolling check in the write edge
        # carries the guarantee through the recording. Probed on the
        # live folder -- the save subfolders may not exist yet.
        ok, free_mb = check_disk_space_ok(Path(settings['live_folder']), MIN_REQUIRED_DISK_MB)
        if not ok:
            raise RecordingRefusedError(
                reason='insufficient_disk',
                title='Insufficient Disk Space',
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
                    title='Cannot Save Recording',
                    message=(
                        f'{save_folder.parent.parent} is not an accessible save '
                        'location. Check that it exists and any external drive '
                        'is connected, then try again.'
                    ),
                ) from exc
            except path_utils.CaptureLocationError as exc:
                raise RecordingRefusedError(
                    reason='capture_location_unusable',
                    title='Cannot Save Recording',
                    message=str(exc),
                ) from exc

        resolved_layer = self._resolve_channel_identity(layer)

        frame_size = scope.imaging.camera_frame_size
        manifest_extra = {
            # RENDERING, not identity: the video builder reads this back to
            # decide whether to false-color a rebuild, so a mono recording
            # must leave it null or the rebuild colorizes frames the user
            # deliberately saved gray.
            'channel_color': resolved_layer if false_color_on else None,
            'camera': {
                'model': identity['model'],
                'serial': identity['serial'],
                'timestamp_tick_hz': identity['timestamp_tick_frequency_hz'],
            },
            'provenance': {
                'host': gather_host_provenance(),
                'software': {'lvp_version': lvp_version},
            },
        }
        config = RecordingConfig(
            fps=effective_fps,
            duration_s=duration_s,
            width=frame_size['width'],
            height=frame_size['height'],
            bit_depth=capture_config.capture_depth,
            output_dir=save_folder,
            filename_template=manual_frame_filename_template(),
            timestamp_overlay=video_settings.get('timestamp_overlay', True),
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

        hyperstack = (
            video_as_frames
            and capture_config.output_format_sequenced == image_mode.OUTPUT_FORMAT_HYPERSTACK
        )
        plan = _RecordingPlan(
            video_as_frames=video_as_frames,
            save_folder=save_folder,
            layer=resolved_layer,
            false_color_on=false_color_on,
            save_encoding=capture_config.save_encoding,
            capture_depth=capture_config.capture_depth,
            tick_freq_hz=identity['timestamp_tick_frequency_hz'],
            hyperstack=hyperstack,
            # One position snapshot for the whole recording: the stage
            # does not move during a manual record, and the writer lane
            # must never query hardware per frame.
            stage_position=(scope.motion.get_current_position() if hyperstack else None),
            pixel_size_um=resolve_recording_pixel_size(scope),
        )

        writer = None
        if not video_as_frames:
            save_folder.mkdir(exist_ok=True, parents=True)
            writer = VideoWriter(
                output_path=save_folder / f'Video_{start_time_str}.mp4',
                fps=effective_fps,
                width=frame_size['width'],
                height=frame_size['height'],
                # Rendering: a null here keeps the encoder gray, which is
                # what a recording with the toggle off must produce.
                color=resolved_layer if false_color_on else None,
                include_timestamp_overlay=config.timestamp_overlay,
                vfr=True,
            )

        engine = VideoRecordingEngine(
            write_frame=self._write_frame,
            claim=self._claim,
            clock=self._clock,
            notify=notifications,
        )
        # Engine start is the commit point: it acquires the claim or
        # raises. Assign controller state only after it succeeds.
        engine.start(config)
        try:
            self._engine = engine
            self._config = config
            self._plan = plan
            self._writer = writer
            self._start_ts = self._clock()
            self._stall_watch = StallWatch(stall_threshold_s(effective_fps, exposure / 1000.0))
            self._rebaser = CameraTickRebaser(identity['timestamp_tick_frequency_hz'], self._clock)
            self._hyperstack_rows = [] if hyperstack else None
            self._last_disk_check_ts = 0.0
            self._on_complete = on_complete
            # The frames folder was reserved before the commit point; nothing
            # to create here.

            scope.imaging.add_frame_listener(self._on_camera_frame, name='manual_recording')
            self._finish_thread = threading.Thread(
                target=self._finish_after_drain, name='ManualRecordingFinish', daemon=True
            )
            self._finish_thread.start()
        except BaseException:
            # Past the commit point the engine holds the claim, and this is
            # the only frame holding the writer -- the engine is handed a
            # write_frame callable and can never close it.
            self._unwind_failed_start(engine, writer)
            raise
        logger.info(
            f'[ManualRecord] Recording started: {effective_fps:.2f} fps, '
            f'max {duration_s:.0f} s, {"frames" if video_as_frames else "mp4"} '
            f'-> {save_folder}'
        )

    def _unwind_failed_start(self, engine: VideoRecordingEngine, writer: Any) -> None:
        """Undo a start that raised after the engine committed.

        Order matters: stop delivering frames, then end the recording so
        the writer lane drains and frees the claim, and only then dispose
        the artifact -- disposing first would leave the lane writing into
        a closed encoder and count those frames as saved.
        """
        try:
            self._scope.imaging.remove_frame_listener(self._on_camera_frame)
        except Exception as e:
            # The listener registers late in start(), so an earlier failure
            # unwinds without one ever having been added.
            logger.debug(f'[ManualRecord] listener removal during start unwind: {e}')

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

        self._engine = None
        self._writer = None
        self._finish_thread = None

    def stop(self, reason: str = 'user_stop') -> None:
        """Close selection; the drain and finish continue on their own.

        Args:
            reason: The manifest's ``end_reason``. The default names the
                public surface's own meaning -- external callers (the
                Record button, app close, an L2 client) ARE the user
                stop; internal enders (duration cap, disk floor) pass
                their own reason.
        """
        engine = self._engine
        if engine is None:
            return
        try:
            self._scope.imaging.remove_frame_listener(self._on_camera_frame)
        except Exception as e:
            logger.warning(f'[ManualRecord] remove_frame_listener failed: {e}')
        engine.stop(reason)

    def tick(self) -> None:
        """Watch the recording's health; the caller polls this.

        Owns the wall-clock duration cap (the engine's frame budget is
        frame-driven, so a dead feed would never fill it) and both
        camera-death checks: the disconnect latch, and the stall watch
        for a feed that dies without an event -- delivery stops while
        the camera still reads active. Detection lives in the poll on
        purpose, so it is armed exactly while a host polls; a host that
        never ticks gets neither the cap nor the detector, and the
        recording runs until its frame budget or an explicit stop.
        """
        if self.is_recording and self._start_ts is not None and self._config is not None:
            if self._clock() - self._start_ts >= self._config.duration_s:
                self.stop(reason='duration_elapsed')
                return
            if not self._scope.imaging.camera_active:
                self._stop_for_camera_loss(reason='camera_disconnected')
                return
            if self._stall_watch is not None and self._stall_watch.stalled(
                self._engine.frames_selected, self._clock()
            ):
                self._stop_for_camera_loss(reason='camera_stalled')

    def _stop_for_camera_loss(self, reason: str) -> None:
        logger.error(f'[ManualRecord] Camera feed lost mid-recording ({reason}); stopping')
        self.stop(reason=reason)
        notifications.error(
            'Recording',
            'Recording Stopped',
            'The camera stopped delivering frames, so the recording was '
            'stopped. Frames captured so far are saved; check the camera '
            'connection before recording again.',
        )

    def discard_pending(self) -> None:
        """Drop the unwritten backlog loudly (the app-close discard path)."""
        if self._engine is not None:
            self._engine.discard_pending()

    # ------------------------------------------------------------------
    # Camera-thread ingest
    # ------------------------------------------------------------------

    def _on_camera_frame(self, image, timestamp, chunks) -> None:
        """SDK-thread listener: rebase the timestamp, offer to the engine."""
        engine = self._engine
        if engine is None or not engine.is_recording:
            return
        engine.ingest_frame(image, self._rebaser.frame_time_s(timestamp, chunks), chunks)

    # ------------------------------------------------------------------
    # Writer-lane edge
    # ------------------------------------------------------------------

    def _write_frame(self, image, timestamp_s, frame_number, config, chunks) -> Path:
        """Write one kept frame as its final artifact (runs on the lane)."""
        self._check_disk_floor(config)

        image = orient_and_fit(image, config.width, config.height)

        plan = self._plan
        if plan.video_as_frames:
            return self._write_tiff_frame(image, timestamp_s, frame_number, config, chunks)
        return self._write_mp4_frame(image, timestamp_s)

    def _write_tiff_frame(self, image, timestamp_s, frame_number, config, chunks) -> Path:
        plan = self._plan
        if config.bit_depth == 8 and image.dtype != np.uint8:
            image = image_utils.convert_to_8bit(image, config.bit_depth)

        metadata, ts_filename = tiff_frame_metadata(
            timestamp_s, frame_number, chunks, plan.tick_freq_hz, plan.pixel_size_um
        )
        file_loc = config.output_dir / config.filename_template.format(
            n=frame_number, ts=ts_filename
        )

        image_save.write_video_frame(
            frame=image,
            file_loc=file_loc,
            metadata=metadata,
            layer_color=plan.layer,
            false_color_on=plan.false_color_on,
            save_encoding=plan.save_encoding,
            capture_depth=plan.capture_depth,
        )

        if self._hyperstack_rows is not None:
            position = plan.stage_position or {}
            # 'Scan Count' is the T-axis ordinal per the execution-record
            # contract; within one recording the temporal ordinal IS the
            # frame number (this dataframe never mixes with scan-indexed
            # rows -- it feeds only the per-recording hyperstack build).
            self._hyperstack_rows.append(
                {
                    'Filepath': file_loc.name,
                    'Scan Count': frame_number,
                    # Channel identity: what was imaged. Independent of the
                    # false-color toggle, which governs display only, so it
                    # is recorded on every frame regardless of rendering.
                    'Color': plan.layer,
                    'Z-Slice': 0,
                    'X': position.get('X'),
                    'Y': position.get('Y'),
                    'Z': position.get('Z'),
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
        now = self._clock()
        if now - self._last_disk_check_ts < DISK_FLOOR_CHECK_INTERVAL_S:
            return
        self._last_disk_check_ts = now
        try:
            ok, free_mb = check_disk_space_ok(config.output_dir, MIN_REQUIRED_DISK_MB)
        except Exception as e:
            logger.warning(f'[ManualRecord] Disk-floor probe failed: {e}')
            return
        if not ok and self._engine is not None and self._engine.is_recording:
            logger.error(
                f'[ManualRecord] Free disk fell to {free_mb:.0f} MB (floor '
                f'{MIN_REQUIRED_DISK_MB} MB); stopping the recording'
            )
            notifications.error(
                'Recording',
                'Recording Stopped -- Disk Almost Full',
                'Free disk space fell below the safety floor, so the recording '
                'was stopped early. Frames captured so far are saved; free up '
                'space before recording again.',
            )
            self.stop(reason='disk_floor')

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
        engine.wait_for_drain()
        try:
            self._scope.imaging.remove_frame_listener(self._on_camera_frame)
        except Exception as e:
            logger.debug(f'[ManualRecord] listener removal at finish: {e}')

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
        except Exception:
            logger.exception('[ManualRecord] Post-drain finish failed')
            notifications.error(
                'Recording',
                'Recording Finalize Failed',
                'The recording finished but its output could not be fully '
                'assembled. Frames already written are on disk; check the log.',
            )
        finally:
            # Reporting needs the measured truth; completing the recording
            # does not. Keeping them apart is what lets the callback fire --
            # and the UI leave its recording state -- after a failed finish.
            if result is not None:
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
                    notifications.warning(
                        'Recording',
                        'Video Frames Dropped',
                        f'{dropped} of {result.frames_selected} frame(s) could not '
                        'be written, so the saved video is shorter than the '
                        'recording. Check the log for the cause.',
                    )
                logger.info(
                    f'[ManualRecord] Finished: {result.frames_written} written, '
                    f'{result.write_failures} failed, measured '
                    f'{result.measured_fps:.2f} fps over {result.measured_duration_s:.2f} s'
                )
            if self._on_complete is not None:
                try:
                    self._on_complete()
                except Exception:
                    logger.exception('[ManualRecord] on_complete callback failed')

    def _build_hyperstack(self) -> None:
        plan = self._plan
        df = pd.DataFrame(self._hyperstack_rows)
        output = plan.save_folder / MANUAL_HYPERSTACK_FILENAME
        result = StackBuilder(
            has_turret=self._scope.motion.has_turret()
        ).create_single_recording_stack(
            df=df,
            path=plan.save_folder,
            output_file_loc=output,
        )
        # The builder reports refusal in its return value rather than by
        # raising. Re-raise it so the finish handler's existing failure
        # notification carries it to the user; announcing a hyperstack the
        # builder declined to write would name a file that does not exist.
        if not result['status']:
            raise RuntimeError(f'hyperstack refused: {result["error"]}')
        logger.info(f'[ManualRecord] Hyperstack created at {output}')


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
    save_encoding: str
    capture_depth: int
    tick_freq_hz: float | None
    hyperstack: bool
    stage_position: dict | None
    # Resolved once at start(), not per frame: the objective cannot change
    # mid-recording, and a per-frame resolve would let one stack hold frames
    # that disagree about their own scale.
    pixel_size_um: float | None
