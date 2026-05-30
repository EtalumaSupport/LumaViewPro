# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Main protocol run loop -- scan timing, hardware checks, completion detection.

Runs on the **protocol-executor** thread.  Extracted from
``sequenced_capture_runner.py`` during the protocol-decomposition refactor.
"""

from __future__ import annotations

import datetime
import time
from typing import TYPE_CHECKING

from lvp_logger import logger

from modules.common_utils import check_disk_space_ok
from modules.protocol_state_machine import ProtocolState

if TYPE_CHECKING:
    from modules.sequenced_capture_runner import SequencedCaptureRunner

from modules.kivy_utils import schedule_ui as _schedule_ui

# --- Disk-space estimation constants ---
ESTIMATED_VIDEO_STEP_MB = 50  # MP4 compressed, ~10-50 MB typical
ESTIMATED_IMAGE_STEP_MB = 8  # 1900x1900 16-bit TIFF ~7.2 MB + metadata
MIN_REQUIRED_DISK_MB = 2048  # Minimum free disk space to start a scan (2 GB)

# --- Hardware health check ---
HW_CHECK_INTERVAL_S = 30  # Seconds between hardware connection checks


class ProtocolRunLoop:
    """Manages scan timing and the outer run loop for protocol execution."""

    def __init__(self, parent: SequencedCaptureRunner):
        self._p = parent

    def run_loop(self):
        """Main entry point -- wraps inner loop with crash recovery."""
        try:
            self._run_loop_inner()
        except Exception as ex:
            logger.error(f'[PROTOCOL] Unhandled exception in run loop: {ex}', exc_info=True)
        finally:
            # Safety net: ensure cleanup always runs so LEDs are turned off,
            # protocol state is reset, and resources are released even if an
            # unhandled exception occurs.  _cleanup() is idempotent (guarded
            # by _cleanup_lock and _run_in_progress check) so duplicate calls
            # from the normal path are harmless.
            self._p._cleanup()

    def _return_to_first_step_between_scans(self):
        """Pre-position the stage at the first step during the inter-scan wait.

        Returning between scans (rather than at the next scan's start) means the
        period wait elapses with the stage already at step 0, so the next scan
        begins on time instead of after a long last-step -> first-step move.
        Pure stage move only -- not go_to_step, which would also power the first
        step's LED during the idle wait. No-op on the final scan, so the stage
        is left where the last scan ended.
        """
        p = self._p
        if p.remaining_scans() <= 0 or p._aborted.is_set():
            return
        try:
            first_step = p._protocol.step(idx=0)
            p._step_executor.default_move(
                px=first_step['X'], py=first_step['Y'], z=first_step['Z']
            )
        except Exception as ex:
            logger.warning(f'[PROTOCOL] Inter-scan return-to-first-step move failed: {ex}')

    def _run_loop_inner(self):
        """Inner run loop body."""
        p = self._p
        last_connection_check = time.monotonic()

        while p._run_in_progress_event.is_set() and not p._aborted.is_set():
            try:
                # Periodic hardware connection check (every 30 seconds)
                now = time.monotonic()
                if now - last_connection_check > HW_CHECK_INTERVAL_S:
                    last_connection_check = now
                    try:
                        if not p._scope.are_all_connected():
                            logger.error(
                                '[PROTOCOL] Hardware disconnected during run -- aborting protocol'
                            )
                            from modules.notification_center import notifications

                            notifications.error(
                                'Protocol',
                                'Protocol Aborted',
                                'Hardware disconnected during protocol run.',
                            )
                            if p._state not in (ProtocolState.COMPLETING, ProtocolState.IDLE):
                                p._set_state(ProtocolState.ERROR)
                            p._cleanup()
                            break
                    except Exception as ex:
                        logger.warning(f'[PROTOCOL] Connection check failed: {ex}')

                # Check if we've completed all scans
                remaining_scans = p.remaining_scans()
                if remaining_scans <= 0:
                    p._cleanup()
                    break

                # Check if enough time has elapsed for the next scan
                # Skip this check for the first scan (scan_count == 0)
                if p._scan_count > 0:
                    current_time = datetime.datetime.now()
                    elapsed_time = current_time - p._start_t

                    if elapsed_time < p._protocol.period():
                        time.sleep(0.1)
                        continue

                    p._start_t = current_time

                # Time for next scan
                if p._callbacks.protocol_iterate_pre:
                    _schedule_ui(
                        lambda dt: p._callbacks.protocol_iterate_pre(
                            remaining_scans=remaining_scans, interval=p._protocol.period()
                        )
                    )

                # Initialize scan variables
                p._curr_step = 0
                # Reset the AF state pointer so the first step's
                # kick-off check sees None.
                p._af_future = None
                if p._callbacks.run_scan_pre:
                    _schedule_ui(lambda dt: p._callbacks.run_scan_pre(), 0)

                # Check disk space once per scan
                try:
                    if p._parent_dir is not None:
                        estimated_mb = 0
                        num_steps = p._protocol.num_steps()
                        for i in range(num_steps):
                            step = p._protocol.step(idx=i)
                            if step.get('Acquire') == 'video':
                                estimated_mb += ESTIMATED_VIDEO_STEP_MB
                            else:
                                estimated_mb += ESTIMATED_IMAGE_STEP_MB
                        required_mb = max(MIN_REQUIRED_DISK_MB, estimated_mb)
                        ok, free_mb = check_disk_space_ok(p._parent_dir, required_mb)
                        if not ok:
                            msg = (
                                f'Insufficient disk space: {free_mb:.0f} MB free, '
                                f'need ~{required_mb:.0f} MB for {num_steps} steps.'
                            )
                            logger.error(f'[PROTOCOL] {msg} -- aborting protocol')
                            from modules.notification_center import notifications

                            notifications.error('Protocol', 'Protocol Aborted', msg)
                            # p._aborted IS protocol_thread.aborted; setting
                            # it from inside the run loop signals the next
                            # iteration to exit and triggers cleanup-on-exit.
                            p._aborted.set()
                            break
                except Exception as e:
                    logger.debug(f'[PROTOCOL] Disk space check failed (proceeding anyway): {e}')

                # Clean LED state before step 0 runs. Without this, a
                # Live-mode LED enabled by the user before pressing Scan
                # stays lit when step 0's led_on fires -- both channels
                # illuminate the sample simultaneously and the first
                # step's image is blown out. led_on is additive at the
                # API + driver layers; the leds_off-before-led_on
                # convention is documented at modules/step_navigation.py
                # and modules/composite_capture.py. Inter-step transitions
                # already do this in protocol_image_writer; step 0 had no
                # previous step, so the convention was silently skipped.
                p._step_executor.leds_off()

                p._step_executor.go_to_step(step_idx=p._curr_step)
                # Guard: if cleanup already ran (e.g. button spam), don't proceed
                if p._aborted.is_set() or p._state == ProtocolState.IDLE:
                    break
                p._scan_in_progress.set()
                p._set_state(ProtocolState.SCANNING)
                # _auto_gain_deadline is set at ARM time per step in
                # protocol_step_runner.scan_iterate (issue #673 fix).
                # No scan-start init: that produced a past-deadline
                # gate after AF and broke convergence. The default 0.0
                # init on SequencedCaptureRunner is safe -- non-AG
                # steps never read the deadline; AG steps overwrite it
                # at arm time before the gate is ever consulted.
                p._auto_gain_armed_step = -1

                start_scan_time = datetime.datetime.now()
                p._step_executor.scan_loop()
                end_scan_time = datetime.datetime.now()
                scan_duration = end_scan_time - start_scan_time

                logger.info(
                    f'Protocol scan {p._scan_count} completed in {scan_duration.total_seconds():.2f} seconds'
                )

                with p._protocol_state_lock:
                    p._scan_count += 1
                logger.debug(f'[{p.LOGGER_NAME}] Scan {p._scan_count}/{p._n_scans} completed')

                if p._callbacks.scan_iterate_post:
                    _schedule_ui(lambda dt: p._callbacks.scan_iterate_post(), 0)

                p._scan_in_progress.clear()
                if p._state == ProtocolState.SCANNING:
                    p._set_state(ProtocolState.RUNNING)

                self._return_to_first_step_between_scans()

            except Exception as ex:
                # Classify: hardware disconnected = fatal (abort +
                # notify); everything else = transient (silent log,
                # retry on next period). Only loss of one of the
                # connected boards (camera, LED, motor) blocks
                # the protocol from making progress; transient errors
                # (one bad LED ack, one frame timeout, etc.) are
                # exactly what a periodic protocol is supposed to ride
                # out. The 30s periodic are_all_connected() check
                # earlier in this loop covers between-scan
                # disconnects; this branch covers during-scan ones.
                try:
                    connected = p._scope.are_all_connected()
                except Exception:
                    # If the connection probe itself fails, assume
                    # the worst -- a disconnect that broke the probe
                    # is still a disconnect.
                    connected = False

                if not connected:
                    logger.error(
                        f'[Protocol] Hardware disconnect during scan: {ex}',
                        exc_info=True,
                    )
                    from modules.notification_center import notifications

                    notifications.error(
                        'Protocol',
                        'Protocol Aborted',
                        'Hardware disconnected during protocol run. '
                        'Check the camera, LED board, and motor board '
                        'connections, then restart the scan.',
                    )
                    if p._state not in (
                        ProtocolState.COMPLETING,
                        ProtocolState.IDLE,
                        ProtocolState.ERROR,
                    ):
                        try:
                            p._set_state(ProtocolState.ERROR)
                        except ValueError:
                            pass
                    p._cleanup()
                    break

                # Transient: log warning, do NOT increment scan_count,
                # do NOT break. The outer while loop's next iteration
                # waits the protocol period and re-runs the scan.
                logger.warning(
                    f'[Protocol] Transient scan failure (hardware '
                    f'still connected); will retry on next period: '
                    f'{ex}',
                    exc_info=True,
                )
                p._scan_in_progress.clear()
                if p._state == ProtocolState.SCANNING:
                    try:
                        p._set_state(ProtocolState.RUNNING)
                    except ValueError:
                        pass

        # Ensure cleanup runs when exiting the while loop
        p._cleanup()
