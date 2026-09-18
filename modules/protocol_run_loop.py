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

from modules.common_utils import MIN_REQUIRED_DISK_MB, check_disk_space_ok
from modules.lumascope_api.illumination import LedTransition, LedTransitionCtx
from modules.protocol_state_machine import ProtocolState
from modules.run_outcome import RunEnding

if TYPE_CHECKING:
    from modules.sequenced_capture_runner import SequencedCaptureRunner

from modules.kivy_utils import schedule_ui as _schedule_ui

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

# One object, because a normal ending is one fact: the loop reaches its end
# either having run every scan or having been told to stop, and both exits
# report through the same record.
RUN_COMPLETED = RunEnding(
    'completed', 'completed', 'Protocol Complete', 'The run finished normally.'
)
RUN_STOPPED = RunEnding('aborted', 'stopped', 'Protocol Stopped', 'Stopped')


class ProtocolRunLoop:
    """Manages scan timing and the outer run loop for protocol execution."""

    def __init__(self, parent: SequencedCaptureRunner):
        self._p = parent

    def run_loop(self) -> None:
        """Main entry point -- wraps inner loop with crash recovery."""
        crash = None
        try:
            self._run_loop_inner()
        except Exception as ex:
            logger.error(
                f'[PROTOCOL] Run loop aborted by exception; cleanup will run: {ex}',
                exc_info=True,
            )
            crash = RunEnding('failed', 'run_loop_crashed', 'Protocol Crashed', str(ex))
            self._p._ending.set_if_unset(crash)
        finally:
            # Safety net: ensure cleanup always runs so LEDs are turned off,
            # protocol state is reset, and resources are released even if an
            # unhandled exception occurs.  _cleanup() is idempotent (guarded
            # by _cleanup_lock and the run-phase check) so duplicate calls
            # from the normal path are harmless -- the inner loop's own
            # cleanup already ran and this no-ops, so the ending below only
            # ever reaches subscribers for a loop that died on the way out.
            self._p._cleanup(
                crash
                or RunEnding(
                    'failed',
                    'run_loop_crashed',
                    'Protocol Crashed',
                    'The run loop exited without cleaning up.',
                )
            )

    def _inter_scan_wait_follows(self) -> bool:
        """Whether the run is about to enter an inter-scan period wait.

        The shared skip condition for the idle-entry actions (LED darkening,
        stage pre-positioning): false after the final scan (no idle follows;
        the run-end transition owns that end state) and on abort (teardown
        owns it). One predicate so the two idle-entry behaviors cannot
        diverge.
        """
        p = self._p
        return p.remaining_scans() > 0 and not p._aborted.is_set()

    def _enter_inter_scan_idle(self):
        """Guarantee the sample is dark before the inter-scan period wait.

        The dark-idle guarantee lives HERE, at the one owner of the idle,
        rather than on the step machinery's success path: any path into the
        wait -- a normal scan end, a final-step write drop that skipped the
        step-boundary decision, or a mid-scan exception riding the
        transient retry -- passes through this epilogue, so no new early
        return or exception path in the step flow can leave a channel lit
        on the sample for a full period. The authority's diff makes it a
        no-op on a scan that already went dark (no blink, no extra serial
        traffic).

        A raise from the apply is contained here by design, not propagated:
        one call site is the transient-retry branch, where a propagated
        raise would escalate a single failed off-command into aborting a
        healthy multi-day timelapse -- the opposite of the ride-out-a-blip
        contract that branch implements. The failure is not silent: it is
        logged at error level, the LED driver's own command-failure path
        fires the user-facing sample-safety notification, and the channel
        is re-asserted exclusive by the next scan's step illumination.
        """
        p = self._p
        if not self._inter_scan_wait_follows():
            return
        try:
            p._step_executor.apply_led_transition(LedTransition.SCAN_IDLE, LedTransitionCtx())
        except Exception as ex:
            logger.error(f'[PROTOCOL] Scan-idle LED-off failed entering the idle wait: {ex}')

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
        if not self._inter_scan_wait_follows():
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

        while p._is_run_live() and not p._aborted.is_set():
            try:
                # Periodic hardware connection check (every 30 seconds)
                now = time.monotonic()
                if now - last_connection_check > HW_CHECK_INTERVAL_S:
                    last_connection_check = now
                    # The probe's handler computes a fact and nothing else: an
                    # abort raised or swallowed in here would leave the run
                    # going with the hardware gone.
                    try:
                        connected = p._scope.are_all_connected()
                    except Exception as ex:
                        logger.warning(f'[PROTOCOL] Connection check failed: {ex}')
                        connected = True
                    if not connected:
                        logger.error(
                            '[PROTOCOL] Hardware disconnected during run -- aborting protocol'
                        )
                        ending = RunEnding(
                            'failed',
                            'hardware_disconnected',
                            'Protocol Aborted',
                            'Hardware disconnected during protocol run. '
                            'Check the USB cable and power connections, '
                            'save the protocol, then restart LumaViewPro '
                            'and the protocol.',
                        )
                        if p._state not in (ProtocolState.COMPLETING, ProtocolState.IDLE):
                            p._set_state(ProtocolState.ERROR)
                        p.abort_run_fatal(ending.reason, ending.title, ending.message)
                        p._cleanup(ending)
                        break

                # Check if we've completed all scans
                remaining_scans = p.remaining_scans()
                if remaining_scans <= 0:
                    p._cleanup(RUN_COMPLETED)
                    break

                # Check if enough time has elapsed for the next scan
                # Skip this check for the first scan (scan_count == 0)
                if p._scan_count > 0:
                    # Monotonic pacing (see _start_t init): elapsed and period
                    # are both in seconds, immune to wall-clock jumps.
                    current_time = time.monotonic()
                    elapsed_time = current_time - p._start_t

                    if elapsed_time < p._protocol.period().total_seconds():
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
                # As with the connection probe: the handler computes the
                # numbers, the abort happens outside it. Inside, the
                # proceeding-anyway catch below would swallow the abort and
                # walk straight into the next step.
                disk_ok, free_mb = True, 0.0
                try:
                    if p._parent_dir is not None:
                        if run_required_mb is None:
                            num_steps = p._protocol.num_steps()
                            run_required_mb = max(
                                MIN_REQUIRED_DISK_MB,
                                p._protocol.estimate_write_mb(
                                    video_as_frames=p._video_as_frames,
                                    global_max_fps=p._video_max_fps,
                                ),
                            )
                        disk_ok, free_mb = check_disk_space_ok(p._parent_dir, run_required_mb)
                except Exception as e:
                    logger.debug(f'[PROTOCOL] Disk space check failed (proceeding anyway): {e}')
                if not disk_ok:
                    msg = (
                        f'Insufficient disk space: {free_mb:.0f} MB free, '
                        f'need ~{run_required_mb:.0f} MB for {num_steps} steps.'
                    )
                    logger.error(f'[PROTOCOL] {msg} -- aborting protocol')
                    # The funnel sets the abort this site used to set by hand,
                    # so the next iteration still exits and cleanup still runs.
                    p.abort_run_fatal('disk_space_critical', 'Protocol Aborted', msg)
                    break

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

                scan_aborted = p._aborted.is_set()
                scan_outcome = 'aborted after' if scan_aborted else 'completed in'
                logger.info(
                    f'Protocol scan {p._scan_count} {scan_outcome} '
                    f'{scan_duration.total_seconds():.2f} seconds'
                )

                # The pacing anchor has two lifecycle events: it is
                # ESTABLISHED at the run's first acquisition (here, once the
                # opening scan has captured), then MAINTAINED at gate-open for
                # every later scan (above). Before this point _start_t holds
                # the run()-entry fallback, which includes run setup + initial
                # motion + step-0 AF -- anchoring the period there shortens
                # the first interval by that lead-in (toward zero at short
                # periods). Deliberately NOT re-established from acquisition
                # on later scans: their gate-open anchor keeps start-to-start
                # cadence fixed instead of drifting by the per-scan lead-in.
                # The fallback remains the anchor only when the opening scan
                # completed no capture at all.
                if p._scan_count == 0 and p._scan_first_capture_t is not None:
                    p._start_t = p._scan_first_capture_t

                new_count = p.advance_scan_count()
                logger.debug(
                    f'[{p.LOGGER_NAME}] Scan {new_count}/{p._n_scans} '
                    f'{"aborted" if scan_aborted else "completed"}'
                )

                if p._callbacks.scan_iterate_post:
                    _schedule_ui(lambda dt: p._callbacks.scan_iterate_post(), 0)

                p._scan_in_progress.clear()
                if p._state == ProtocolState.SCANNING:
                    p._set_state(ProtocolState.RUNNING)

                # Dark before (and during) the pre-positioning move and the
                # period wait -- one of the two entries into the idle.
                self._enter_inter_scan_idle()
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
                    ending = RunEnding(
                        'failed',
                        'hardware_disconnected',
                        'Protocol Aborted',
                        'Hardware disconnected during protocol run. '
                        'Check the USB cable and power connections, save '
                        'the protocol, then restart LumaViewPro and the '
                        'protocol.',
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
                    p.abort_run_fatal(ending.reason, ending.title, ending.message)
                    p._cleanup(ending)
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

                    message = (
                        f'The scan failed {consecutive_scan_failures} times '
                        'in a row. Check the USB cable and power connections '
                        'and the log for the cause, save the protocol, then '
                        'restart LumaViewPro and the protocol.'
                    )
                    notifications.error('Protocol', 'Protocol Aborted', message, fatal=True)
                    ending = RunEnding(
                        'failed', 'consecutive_scan_failures', 'Protocol Aborted', message
                    )
                    p._ending.set_if_unset(ending)
                    p._cleanup(ending)
                    break

                # The failed scan may have died with a channel lit (an
                # exception between the step's illuminate and its boundary
                # decision) -- the other entry into the idle. Dark before
                # the period wait, or the sample stays lit for the full
                # period per retry. After the strike escalation above, so
                # the final strike goes straight to cleanup's run-end
                # decision without an extra darken that the restore-original
                # end state would immediately reverse (an off-then-on blink).
                self._enter_inter_scan_idle()

        # Ensure cleanup runs when exiting the while loop. The while
        # condition goes false on an abort (aborted set) or when the run
        # flag cleared; name which one so subscribers see the truth.
        p._cleanup(RUN_STOPPED if p._aborted.is_set() else RUN_COMPLETED)
