# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Image/video capture and file-write orchestration for protocol execution.

Runs on the protocol-executor thread (_capture) and file-IO thread
(_write_capture).  Extracted from ``sequenced_capture_executor.py``
during the protocol-decomposition refactor.
"""

from __future__ import annotations

import datetime
import pathlib
import threading
from typing import TYPE_CHECKING

from lvp_logger import logger

import modules.common_utils as common_utils
from modules.protocol import Protocol
from modules.video_capture import VideoCaptureSession, write_video
from modules.sequential_io_executor import IOTask

if TYPE_CHECKING:
    from modules.lumascope_api import Lumascope
    from modules.protocol_callbacks import ProtocolCallbacks
    from modules.protocol_execution_record import ProtocolExecutionRecord
    from modules.sequential_io_executor import SequentialIOExecutor


class ProtocolImageWriter:
    """Handles image/video capture and file writing during protocol runs.

    Created by SequencedCaptureExecutor at the start of each run with
    the references it needs.  All state is borrowed from the executor —
    this class owns no mutable state of its own.
    """

    LOGGER_NAME = "SequencedCaptureExecutor"

    def __init__(
        self,
        *,
        scope: Lumascope,
        callbacks: ProtocolCallbacks,
        protocol_ended: threading.Event,
        video_write_finished: threading.Event,
        file_io_executor: SequentialIOExecutor,
        protocol_executor,  # SequentialIOExecutor (protocol queue)
        execution_record: ProtocolExecutionRecord,
        # Functions borrowed from the parent executor
        leds_off_fn,
        led_on_fn,
        is_run_in_progress_fn,
    ):
        self._scope = scope
        self._callbacks = callbacks
        self._protocol_ended = protocol_ended
        self._video_write_finished = video_write_finished
        self._file_io_executor = file_io_executor
        self._protocol_executor = protocol_executor
        self._execution_record = execution_record
        self._leds_off = leds_off_fn
        self._led_on = led_on_fn
        self._is_run_in_progress = is_run_in_progress_fn
        self._consecutive_capture_failures = 0
        self._MAX_CONSECUTIVE_CAPTURE_FAILURES = 3

    def capture(
        self,
        save_folder,
        step,
        output_format: str,
        protocol,
        *,
        scan_count=None,
        sum_count: int = 1,
        enable_image_saving: bool = True,
        image_capture_config: dict | None = None,
        autogain_settings: dict | None = None,
        video_as_frames: bool = False,
        separate_folder_per_channel: bool = False,
        curr_step: int = 0,
        keep_led_on: bool = False,
    ):
        """Orchestrate image/video acquisition for a single protocol step.

        Runs on the protocol-executor thread.
        """
        if self._protocol_ended.is_set():
            return
        if not self._is_run_in_progress():
            return
        if not self._protocol_executor.is_protocol_running():
            return

        is_video = step['Acquire'] == "video"

        # #610 diagnostic: trace camera settings decision at each capture
        _ag = step['Auto_Gain']
        _curr_gain = self._scope.get_gain()
        _curr_exp = self._scope.get_exposure_time()
        logger.info(
            f"[CAPTURE DIAG] step={step.get('Name','?')} color={step['Color']} "
            f"Auto_Gain={_ag!r} (type={type(_ag).__name__}) "
            f"step_gain={step['Gain']} step_exp={step['Exposure']} "
            f"camera_gain={_curr_gain} camera_exp={_curr_exp}"
        )

        if not step['Auto_Gain']:
            logger.info(f"[CAPTURE DIAG] Applying step camera settings: gain={step['Gain']}, exp={step['Exposure']}")
            with self._scope.update_camera_config():
                self._scope.set_gain(step['Gain'])
                self._scope.set_exposure_time(step['Exposure'])
        else:
            logger.warning(f"[CAPTURE DIAG] SKIPPING camera settings — Auto_Gain is truthy: {_ag!r}")

        # Objective short name for filename
        objective_short_name = None
        if self._scope.has_turret():
            obj_info = self._scope.get_objective_info(objective_id=step["Objective"])
            if obj_info is not None:
                objective_short_name = obj_info.get('short_name')
            else:
                logger.warning(
                    f"[PROTOCOL] Turret available but no objective info for ID "
                    f"'{step['Objective']}' — using None for filename"
                )

        # Build base name from protocol's custom root + step name
        try:
            capture_root = protocol.capture_root()
        except Exception:
            capture_root = ''

        if capture_root not in (None, ''):
            combined_prefix = f"{capture_root}_{step['Name']}"
        else:
            combined_prefix = step['Name']

        # In engineering mode, include turret position in filename
        turret_pos = None
        if self._scope.engineering_mode and self._scope.has_turret():
            try:
                turret_pos = int(self._scope.get_current_position('T'))
            except Exception:
                pass

        name = common_utils.generate_default_step_name(
            well_label=step['Well'],
            color=step['Color'],
            z_height_idx=step['Z-Slice'],
            scan_count=scan_count,
            custom_name_prefix=combined_prefix,
            objective_short_name=objective_short_name,
            tile_label=step['Tile'],
            video=is_video,
            turret_position=turret_pos,
        )
        # Ensure the filename base has no invalid path characters
        try:
            name = Protocol.sanitize_step_name(input=name)
        except Exception:
            pass

        # Illuminate
        if self._scope.led_connected:
            self._led_on(color=step['Color'], illumination=step['Illumination'], block=True)
            logger.info(f"[{self.LOGGER_NAME} ] scope.led_on({step['Color']}, {step['Illumination']})")
        else:
            logger.warning('LED controller not available.')

        sum_iteration_callback = None
        use_color = step['Color'] if step['False_Color'] else 'BF'

        if enable_image_saving:
            use_full_pixel_depth = image_capture_config['use_full_pixel_depth']

            if is_video:
                session = VideoCaptureSession(
                    scope=self._scope,
                    step=step,
                    autogain_settings=autogain_settings,
                    is_protocol_running_fn=self._protocol_executor.is_protocol_running,
                    callbacks=self._callbacks.to_dict(),
                    leds_off_fn=self._leds_off,
                )
                video_result = session.capture()

                if video_result is None:
                    # Cancelled or zero frames — skip write
                    self._video_write_finished.set()
                    self._leds_off()
                    return

                self._leds_off()

                self._file_io_executor.protocol_put(IOTask(
                    action=self.write_capture,
                    kwargs={
                        "is_video": is_video,
                        "video_as_frames": video_as_frames,
                        "video_result": video_result,
                        "save_folder": save_folder,
                        "use_color": use_color,
                        "name": name,
                        "output_format": output_format,
                        "step": step,
                        "captured_image": None,
                        "step_index": curr_step,
                        "scan_count": scan_count,
                        "capture_time": datetime.datetime.now(),
                        "enable_image_saving": enable_image_saving,
                        "separate_folder_per_channel": separate_folder_per_channel,
                    }
                ))
                return  # Video: leds_off already called at line 181

            else:
                # Frame validity drains stale frames, then grabs a valid one
                captured_image = self._scope.capture_and_wait(
                    force_to_8bit=not use_full_pixel_depth,
                    all_ones_check=True,
                    timeout=datetime.timedelta(seconds=1.0),
                    sum_count=sum_count,
                    sum_delay_s=step["Exposure"] / 1000,
                    sum_iteration_callback=sum_iteration_callback,
                )

                if captured_image is False:
                    self._consecutive_capture_failures += 1
                    logger.error(f"[PROTOCOL] Capture failed for step {curr_step} ({step.get('Name', '?')}), scan {scan_count} — camera inactive or frame drain failed (failure {self._consecutive_capture_failures}/{self._MAX_CONSECUTIVE_CAPTURE_FAILURES})")
                    # Still record the step with "capture_failed" so the record isn't silently missing
                    self._file_io_executor.protocol_put(IOTask(
                        action=self.write_capture,
                        kwargs={
                            "step": step,
                            "step_index": curr_step,
                            "scan_count": scan_count,
                            "capture_time": datetime.datetime.now(),
                            "enable_image_saving": enable_image_saving,
                            "separate_folder_per_channel": separate_folder_per_channel,
                        }
                    ))
                    self._leds_off()
                    if self._consecutive_capture_failures >= self._MAX_CONSECUTIVE_CAPTURE_FAILURES:
                        from modules.notification_center import notifications
                        notifications.critical("Protocol", "Camera Failure",
                            f"Camera failed {self._consecutive_capture_failures} consecutive captures. Aborting protocol.")
                        self._protocol_ended.set()
                    return

                self._consecutive_capture_failures = 0  # Reset on success
                logger.info(f"Protocol Image Captured: {name}")

                self._file_io_executor.protocol_put(IOTask(
                    action=self.write_capture,
                    kwargs={
                        "save_folder": save_folder,
                        "use_color": use_color,
                        "name": name,
                        "output_format": output_format,
                        "step": step,
                        "captured_image": captured_image,
                        "step_index": curr_step,
                        "scan_count": scan_count,
                        "capture_time": datetime.datetime.now(),
                        "enable_image_saving": enable_image_saving,
                        "separate_folder_per_channel": separate_folder_per_channel,
                    }
                ))

        else:
            self._file_io_executor.protocol_put(IOTask(
                action=self.write_capture,
                kwargs={
                    "step": step,
                    "enable_image_saving": enable_image_saving,
                    "separate_folder_per_channel": separate_folder_per_channel,
                }
            ))

        if not keep_led_on:
            self._leds_off()

    def write_capture(
        self,
        is_video=False,
        video_as_frames=False,
        video_result=None,
        save_folder=None,
        use_color=None,
        name=None,
        output_format=None,
        step=None,
        captured_image=None,
        step_index=None,
        scan_count=None,
        capture_time=None,
        enable_image_saving=True,
        separate_folder_per_channel=False,
    ):
        """Write captured image/video to disk and record in execution log.

        Runs on the file-IO thread.
        """
        captured_frames = 0
        duration_sec = 0.0

        # M8: Check disk space before writing — long protocols can fill disk.
        if save_folder is not None:
            try:
                import shutil
                free_mb = shutil.disk_usage(str(save_folder)).free / (1024 * 1024)
                if free_mb < 500:  # 500 MB floor
                    from modules.notification_center import notifications
                    notifications.critical("FileIO", "Disk Space Critical",
                        f"Only {free_mb:.0f} MB free. Aborting protocol to prevent data loss.")
                    self._protocol_ended.set()
                    return
            except Exception:
                pass  # If we can't check, proceed anyway

        if enable_image_saving:
            if is_video:
                try:
                    capture_result = write_video(
                        result=video_result,
                        save_folder=save_folder,
                        name=name,
                        video_as_frames=video_as_frames,
                        step=step,
                        callbacks=self._callbacks.to_dict(),
                    )
                finally:
                    self._video_write_finished.set()

                captured_frames = video_result.captured_frames
                duration_sec = video_result.duration_sec

            else:
                if captured_image is False:
                    logger.warning(f"[PROTOCOL] _write_capture: captured_image is False for step {step_index} ({step.get('Name', '?') if step else '?'}), scan {scan_count}, recording as capture_failed")
                    if self._execution_record is not None:
                        self._execution_record.add_step(
                            capture_result_file_name="capture_failed",
                            step_name=name if name else "unknown",
                            step_index=step_index,
                            scan_count=scan_count,
                            timestamp=capture_time,
                            frame_count=0,
                            duration_sec=0.0,
                        )
                    return

                capture_result = self._scope.save_image(
                    array=captured_image,
                    save_folder=save_folder,
                    file_root=None,
                    append=name,
                    color=use_color,
                    tail_id_mode=None,
                    output_format=output_format,
                    true_color=step['Color'],
                    x=step['X'],
                    y=step['Y'],
                    z=step['Z']
                )

                del captured_image

            if capture_result is None:
                capture_result_filepath_name = "unsaved"
            elif isinstance(capture_result, dict):
                capture_result_filepath_name = capture_result['metadata']['file_loc']
            elif separate_folder_per_channel:
                capture_result_filepath_name = pathlib.Path(step["Color"]) / capture_result.name
            else:
                capture_result_filepath_name = capture_result.name

        else:
            capture_result_filepath_name = "unsaved"

        if self._execution_record is not None:
            try:
                self._execution_record.add_step(
                    capture_result_file_name=capture_result_filepath_name,
                    step_name=name,
                    step_index=step_index,
                    scan_count=scan_count,
                    timestamp=capture_time,
                    frame_count=captured_frames if is_video else 1,
                    duration_sec=duration_sec if is_video else 0.0
                )
                logger.info(f"Protocol-Writer] Added step to protocol execution record")
            except Exception as ex:
                logger.error(f"[Protocol-Writer] Failed to add step to protocol execution record: {ex}")
