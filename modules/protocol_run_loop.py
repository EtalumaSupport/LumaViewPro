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

from modules.common_utils import check_disk_space_ok, estimate_step_write_mb
from modules.protocol_state_machine import ProtocolState

if TYPE_CHECKING:
    from modules.sequenced_capture_runner import SequencedCaptureRunner

from modules.kivy_utils import schedule_ui as _schedule_ui

# --- Disk-space estimation constants ---
MIN_REQUIRED_DISK_MB = 2048  # Minimum free disk space to start a scan (2 GB)

# --- Hardware health check ---
HW_CHECK_INTERVAL_S = 30  # Seconds between hardware connection checks

# --- Persistent-failure escalation ---
# Transient scan failures (one bad LED ack, one frame timeout) are retried
# on the next period by design -- a periodic protocol should ride out a
# blip. But "transient" failures that recur every period are not transient:
# without a ceiling, a persistently-failing overnight run retries forever,
# producing an empty dataset with only warning-level log lines. After this
# many CONSECUTIVE failed scans (no successful scan between), the run
# aborts loudly instead. Per PERFORMANCE_BUDGETS.md row
# protocol_consecutive_scan_failures.
MAX_CONSECUTIVE_SCAN_FAILURES = 3


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
            p._step_executor.default_move(px=first_step['X'], py=first_step['Y'], z=first_step['Z'])
        except Exception as ex:
            logger.warning(f'[PROTOCOL] Inter-scan return-to-first-step move failed: {ex}')

    def _run_loop_inner(self):
        """Inner run loop body."""
        p = self._p
        last_connection_check = time.monotonic()

        consecutive_scan_failures = 0

        # The per-step write estimate is a per-run invariant -- the protocol
        # steps and the video_as_frames mode do not change mid-run -- so compute
        # the required free space once on the first scan that checks and reuse
        # it, instead of re-walking every step on each scan's disk check. Stays
        # None (and unevaluated) until a scan with an output dir actually needs
        # it.
        run_required_mb = None
        num_steps = 0

        while p._run_in_progress_event.is_set() and not p._aborted.is_set():
            try:
                # ERROR is terminal for the run: a step-level failure (e.g.
                # motion timeout) already set it and notified the user, and
                # only cleanup may transition ERROR back to IDLE. Without
                # this gate the next period re-entered the scan path, the
                # ERROR->SCANNING transition raised, and the transient-
                # failure classifier below retried forever -- a wedged
                # multi-day run that delivers zero captures after one
                # timeout.
                if p._state == ProtocolState.ERROR:
                    logger.error(
                        '[PROTOCOL] Run is in ERROR state -- stopping protocol and cleaning up'
                    )
                    from modules.notification_center import notifications

                    notifications.error(
                        'Protocol',
                        'Protocol Stopped',
                        'The protocol stopped after an unrecoverable step '
                        'failure. Review the log for the cause, then restart '
                        'the scan.',
                    )
                    p._cleanup()
                    break

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
                        lambda dt, rs=remaining_scans: p._callbacks.protocol_iterate_pre(
                            remaining_scans=rs, interval=p._protocol.period()
                        )
                    )

                # Initialize per-scan state (curr_step, AF pointer).
                p._reset_scan_state()
                if p._callbacks.run_scan_pre:
                    _schedule_ui(lambda dt: p._callbacks.run_scan_pre(), 0)

                # Check disk space once per scan, against the per-run estimate
                # summed once on the first check and reused thereafter.
                try:
                    if p._parent_dir is not None:
                        if run_required_mb is None:
                            num_steps = p._protocol.num_steps()
                            run_required_mb = max(
                                MIN_REQUIRED_DISK_MB,
                                sum(
                                    estimate_step_write_mb(
                                        p._protocol.step(idx=i),
                                        video_as_frames=p._video_as_frames,
                                    )
                                    for i in range(num_steps)
                                ),
                            )
                        ok, free_mb = check_disk_space_ok(p._parent_dir, run_required_mb)
                        if not ok:
                            msg = (
                                f'Insufficient disk space: {free_mb:.0f} MB free, '
                                f'need ~{run_required_mb:.0f} MB for {num_steps} steps.'
                            )
                            logger.error(f'[PROTOCOL] {msg} -- aborting protocol')
                            from modules.notification_center import notifications

                            notifications.error('Protocol', 'Protocol Aborted', msg, fatal=True)
                            # p._aborted IS protocol_thread.aborted; setting
                            # it from inside the run loop signals the next
                            # iteration to exit and triggers cleanup-on-exit.
                            p._aborted.set()
                            break
                except Exception as e:
                    logger.debug(f'[PROTOCOL] Disk space check failed (proceeding anyway): {e}')

                # No nuclear leds_off before step 0: each step's capture makes
                # its channel exclusive (turns off every OTHER channel, leaves
                # an already-correct channel untouched). That still kills a
                # stray Live-mode LED so step 0 is not double-illuminated, but
                # WITHOUT clearing the LED-state cache. Clearing the cache here
                # forced the following same-color led_on to re-fire, blinking
                # the LED off->on at the start of every scan.
                p._step_executor.go_to_step(step_idx=p._curr_step)
                # Guard: if cleanup already ran (e.g. button spam), don't proceed
                if p._aborted.is_set() or p._state == ProtocolState.IDLE:
                    break
                p._scan_in_progress.set()
                p._set_state(ProtocolState.SCANNING)
                # Reset the per-step Auto_Gain arm guard so each scan arms AG
                # once per step (scan_iterate keys the one-shot on curr_step).
                p._auto_gain_armed_step = -1

                start_scan_time = datetime.datetime.now()
                p._step_executor.scan_loop()
                end_scan_time = datetime.datetime.now()
                scan_duration = end_scan_time - start_scan_time

                logger.info(
                    f'Protocol scan {p._scan_count} completed in {scan_duration.total_seconds():.2f} seconds'
                )

                new_count = p.advance_scan_count()
                logger.debug(f'[{p.LOGGER_NAME}] Scan {new_count}/{p._n_scans} completed')

                if p._callbacks.scan_iterate_post:
                    _schedule_ui(lambda dt: p._callbacks.scan_iterate_post(), 0)

                p._scan_in_progress.clear()
                if p._state == ProtocolState.SCANNING:
                    p._set_state(ProtocolState.RUNNING)

                self._return_to_first_step_between_scans()
                consecutive_scan_failures = 0

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
                # Handle-state check only (cached connected flags), NOT a
                # liveness round-trip: a camera whose handle is valid but whose
                # grab has died reads connected=True and so classifies as
                # transient. The consecutive-failure cap below bounds that case;
                # a true liveness probe (a hardware round-trip) is deferred --
                # it needs bench validation before it can change classification.
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
                    f'[Protocol] Scan failure with hardware handles still '
                    f'present (handle-state check, not a confirmed liveness '
                    f'probe); retrying on next period: {ex}',
                    exc_info=True,
                )
                p._scan_in_progress.clear()
                if p._state == ProtocolState.SCANNING:
                    try:
                        p._set_state(ProtocolState.RUNNING)
                    except ValueError:
                        pass

                consecutive_scan_failures += 1
                if consecutive_scan_failures >= MAX_CONSECUTIVE_SCAN_FAILURES:
                    logger.error(
                        f'[PROTOCOL] {consecutive_scan_failures} consecutive '
                        'scan failures with no successful scan between -- '
                        'aborting protocol'
                    )
                    from modules.notification_center import notifications

                    notifications.error(
                        'Protocol',
                        'Protocol Aborted',
                        f'The scan failed {consecutive_scan_failures} times '
                        'in a row. Check hardware connections and the log '
                        'for the cause, then restart the scan.',
                    )
                    p._cleanup()
                    break

        # Ensure cleanup runs when exiting the while loop
        p._cleanup()
