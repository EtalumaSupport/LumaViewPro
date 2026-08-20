# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import datetime
import logging
import pathlib
import threading
import time
from typing import TYPE_CHECKING

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from lvp_logger import logger

import modules.autofocus_functions as autofocus_functions
import modules.common_utils as common_utils
import modules.lumascope_api as lumascope_api
from modules.exceptions import AutofocusAborted
from modules.kivy_utils import schedule_ui as _schedule_ui
from modules.lumascope_api.illumination import (
    LedTransition,
    LedTransitionCtx,
    snapshot_lit_pairs,
)
from modules.notification_center import notifications
from modules.objectives_loader import ObjectiveLoader
from modules.sequential_io_executor import IOTask, SequentialIOExecutor

if TYPE_CHECKING:
    from modules.lumascope_api.illumination import LedLease

_af_log = logging.getLogger('LVP.autofocus')


class AutofocusRunner:
    def __init__(
        self,
        scope: lumascope_api.Lumascope,
        camera_executor: SequentialIOExecutor,
        io_executor: SequentialIOExecutor,
        file_io_executor: SequentialIOExecutor,
        ui_update_func=None,
    ):
        self._scope = scope
        self._camera_executor = camera_executor
        self._io_executor = io_executor
        self._file_io_executor = file_io_executor
        self.ui_update_func = ui_update_func

        # Set by run() before the loop starts; consulted by _iterate
        # each iteration. AutofocusThread owns the actual Event.
        self._abort_event: threading.Event | None = None

        # Guards _callbacks reads/writes -- the AF thread writes in
        # run(); UI dispatches read via _schedule_ui in _iterate.
        self._callbacks_lock = threading.Lock()

        self._af_in_progress = threading.Event()

        self._reset_state()

        if not self._scope.imaging.camera_active:
            return

        self._objective_loader = ObjectiveLoader()
        self._reset_state()

    def _notify_af_failure(self, title: str, message: str) -> None:
        """Fire an AF failure user-notification IF the trigger is
        interactive. Suppressed for unattended ('protocol') triggers
        because protocols continue with the prior Z position and
        should not block on modal popups -- per the
        "protocols are unattended" product contract. Log lines are
        emitted by the caller regardless of trigger source; this is
        the popup-gate only."""
        if self._run_trigger_source == 'protocol':
            return
        notifications.error('Autofocus', title, message)

    def reset(self):
        # Skip if a run is in flight: _reset_state() would wipe _params
        # while AFE.run() reads it on the AF thread. AFE.run()'s own
        # _reset_state() on entry covers deferred cleanup. Callers that
        # need to stop the in-flight run should call
        # autofocus_thread.abort() first; reset() succeeds once the AF
        # thread's finally block clears _af_in_progress.
        if self._af_in_progress.is_set():
            logger.debug(
                '[AF] reset() skipped: AF run in flight; '
                'call autofocus_thread.abort() to unwind first'
            )
            return
        # Restore Z precision mode to the resting default (ON). AF
        # temporarily disables precision during coarse passes for speed
        # and re-enables for the fine pass; this restores ON regardless
        # of which AF phase was interrupted so subsequent protocol Z
        # moves stop accurately.
        try:
            self._scope.motion.set_precision_mode('Z', True)
        except Exception as e:
            logger.debug(
                f'[AF] Could not restore precision mode during reset (scope may be unavailable): {e}'
            )
        self._reset_state()

    def set_scope(self, scope: lumascope_api.Lumascope):
        self._scope = scope

    def _calculate_params(self):
        center = self._scope.motion.get_current_position('Z')

        range = self._objective['AF_range']

        if range <= 0:
            raise ValueError(f'AF_range must be positive, got {range}')

        z_min = max(0, center - range)
        z_max = center + range
        resolution = self._objective['AF_max']
        exposure = self._scope.imaging.get_exposure_time()

        self._params = {
            'center': center,
            'range': range,
            'z_min': z_min,
            'z_max': z_max,
            'resolution': resolution,
            'exposure': exposure,
        }

    def _led_off(self):
        """Turn off only the LED(s) that AF owns (not all LEDs)."""
        if self._led_color is not None and self._scope.led_connected:
            self._scope.illumination.leds_off_owned('autofocus')

    def run(
        self,
        objective_id: str,
        callbacks: dict | None = None,
        save_results_to_file: bool = False,
        run_trigger_source: str | None = None,
        results_dir: pathlib.Path | None = None,
        led_color: str | None = None,
        led_illumination: float = 0,
        camera_gain: float | None = None,
        camera_exposure: float | None = None,
        abort_event: threading.Event | None = None,
        keep_led_on: bool = False,
        led_lease: 'LedLease | None' = None,
    ) -> float | None:
        """Run autofocus to completion synchronously on the caller's thread.

        Intended to be called by AutofocusThread; the abort_event ties
        the AF loop to the thread's per-run abort signal.

        Args:
            objective_id: which objective profile to use.
            callbacks: optional dict of UI hooks. Recognized keys:
                'move_position' -- called per Z move on the UI thread.
                The 'complete' hook from the prior API has been retired;
                completion is signalled via the AutofocusThread Future.
            save_results_to_file: if True, queue a save of AF results
                to results_dir on file_io_executor at AF end.
            run_trigger_source: free-form string recorded in saved data.
            results_dir: required when save_results_to_file=True.
            led_color, led_illumination, camera_gain, camera_exposure:
                AF-scan settings applied at start, restored at end.
            abort_event: signalled by caller to abort the run. Required.
            led_lease: the caller's LED lease when AF runs inside a
                protocol step -- AF takes a child lease under it. None for
                an interactive run, where AF takes a top-level lease itself.

        Returns:
            best_focus_position (float) on success, or None when the AF
            curve was degenerate.

        Raises:
            AutofocusAborted: abort_event was set during the run.
            Exception: any other AF failure; logged + user-notified
                before re-raising.
        """
        if self._af_in_progress.is_set():
            raise RuntimeError('Autofocus already in progress')

        if abort_event is None:
            raise ValueError('abort_event is required')

        if save_results_to_file and results_dir is None:
            raise ValueError('Cannot save autofocus results to file if results_dir is None')

        self._reset_state()
        with self._callbacks_lock:
            self._callbacks = callbacks if callbacks is not None else {}
        self._abort_event = abort_event
        self._run_trigger_source = run_trigger_source
        self._led_color = led_color
        self._led_illumination = led_illumination
        self._camera_gain = camera_gain
        self._camera_exposure = camera_exposure
        # When the protocol step that triggered AF will capture on the
        # same channel + illumination as the AF scan, skip the AF-end
        # LED off + state restore so the capture inherits the LED state
        # already established by AF (#612). Saves the redundant
        # off-on cycle (~50-200 ms + an extra LED mechanical cycle per
        # AF-every-N step). Caller (protocol_step_runner) sets this;
        # interactive AF runs default to False so pre-AF state is
        # restored as before.
        self._keep_led_on = keep_led_on
        self._last_progress_ts = time.monotonic()

        self._save_results_to_file = save_results_to_file
        # Per-run timestamped subdir under the caller's results_dir.
        # Eager mkdir so aborted runs still leave a directory marker
        # and successive runs do not collide on filenames.
        if save_results_to_file:
            self._results_dir = self._allocate_results_dir(results_dir)
        else:
            self._results_dir = None
        self._is_focusing_event.set()
        self._af_in_progress.set()
        # Mirror to the public ImagingAPI surface so callers can ask
        # scope.imaging.is_focusing and get the right answer during AF.
        # Tracks _af_in_progress (cleared LAST in the finally block) rather
        # than _is_focusing_event (cleared mid-flight in _iterate before
        # camera-state restore).
        self._scope.imaging.is_focusing = True

        self._objective = self._objective_loader.get_objective_info(objective_id=objective_id)

        self._calculate_params()
        self._af_start_time = time.monotonic()
        self._af_pass_num = 0
        _af_log.info(
            f'--- AF START objective={objective_id} '
            f'center={self._params["center"]:.1f} '
            f'range={self._params["range"]:.1f} '
            f'step={self._params["resolution"]:.1f} '
            f'z=[{self._params["z_min"]:.1f}, {self._params["z_max"]:.1f}] ---'
        )
        # Snapshot Z so abort / exception exits can restore the user's
        # pre-AF position. On success the fine-pass move overrides this
        # with best_focus_position.
        try:
            self._saved_z_position = self._scope.motion.get_current_position('Z')
        except Exception as e:
            logger.debug(f'[AF] Could not snapshot pre-AF Z position: {e}')
            self._saved_z_position = None
        self._saved_led_state = self._scope.illumination.save_led_state('autofocus')
        self._saved_camera_state = self._scope.imaging.save_camera_state('autofocus')
        _af_log.info(
            f'[AF DIAG] Saved pre-AF camera state: '
            f'gain={self._saved_camera_state.get("gain_db", "?")} '
            f'exp={self._saved_camera_state.get("exposure_ms", "?")} '
            f'(step wants gain={self._camera_gain} exp={self._camera_exposure})'
        )
        # Apply the step's camera settings so AF scans with correct gain
        # and exposure rather than inheriting the prior step's values.
        if self._camera_gain is not None:
            self._scope.imaging._set_gain_impl(self._camera_gain)
        if self._camera_exposure is not None:
            self._scope.imaging._set_exposure_time_impl(self._camera_exposure)
        last_gc_time = time.monotonic()
        completed_successfully = False
        try:
            # Acquire the LED lease BEFORE driving illumination below. AF
            # illuminates by calling apply(AF_ENTER) ON this lease; issued
            # before AF holds a lease, a protocol's already-held lease would
            # refuse the out-of-turn write and the AF channel never lights --
            # AF would then scan an unlit field. Inside a protocol step the
            # protocol passes its lease and AF nests as a child it must
            # outlive; an interactive run takes a top-level lease. The alive
            # probe is _af_in_progress (set above, cleared LAST in the
            # finally), so a contender can prove this run dead but never
            # steal from it live. The acquire sits inside the try so a
            # refused acquire unwinds through the finally (camera/Z restore,
            # in-progress flags cleared) instead of latching is_focusing.
            if led_lease is not None:
                self._led_lease = led_lease.acquire_child(
                    'autofocus', alive=self._af_in_progress.is_set
                )
            else:
                self._led_lease = self._scope.illumination.acquire_led_lease(
                    'autofocus', alive=self._af_in_progress.is_set
                )
            if self._led_lease is None:
                # A live owner holds illumination authority. AF without the
                # lease would sweep an unlit field and commit a garbage Z --
                # refuse the run loudly instead. error severity: the
                # operation ABORTED, and the likeliest contention (a running
                # protocol) suppresses non-fatal popups, which would
                # otherwise swallow exactly this message.
                holder = self._scope.illumination.led_lease_owner
                holder_desc = f'Another operation ({holder})' if holder else 'Another operation'
                logger.error(f'[AF] LED lease refused (held live by {holder!r}); aborting run')
                notifications.error(
                    'Autofocus',
                    'Autofocus Did Not Start',
                    f'{holder_desc} is controlling the microscope '
                    'illumination. Let it finish, then run autofocus.',
                )
                raise AutofocusAborted(f'LED authority held live by {holder!r}')
            # Make the AF channel the only lit one before scanning, confirmed
            # on (AF_ENTER blocks) so the focus metric never reads a dark or
            # mixed-illumination frame: a Live-mode LED on another channel
            # would otherwise stay lit alongside the AF channel and bias the
            # metric. The authority diff offs every non-target channel and
            # leaves an AF channel already at target untouched (no off->on
            # blink). No AF color means an empty target, so ambient AF clears
            # every channel. Pre-AF state was snapshotted into
            # self._saved_led_state above; the exit restores it via
            # AF_TO_CAPTURE.
            af_channel = (
                self._scope.illumination.color2ch(self._led_color)
                if self._led_color is not None
                else None
            )
            self._led_lease.apply(
                LedTransition.AF_ENTER,
                LedTransitionCtx(channel=af_channel, mA=self._led_illumination),
            )
            # Drop Z precision for the coarse passes; the fine pass restores
            # precision ON, and all exit paths (success, abort, exception)
            # also restore ON via the finally block and reset().
            try:
                self._scope.motion.set_precision_mode('Z', False)
            except Exception as e:
                logger.debug(f'[AF] Could not drop precision mode for coarse passes: {e}')
            self._move_absolute_position(pos=self._params['z_min'])

            while (
                self._af_in_progress.is_set()
                and self._is_focusing_event.is_set()
                and not abort_event.is_set()
            ):
                # Periodic maintenance: GC every 60 seconds
                if time.monotonic() - last_gc_time > 60:
                    import gc

                    gc.collect()
                    last_gc_time = time.monotonic()

                self._iterate()

                # Small inter-iteration delay; wake early on abort.
                if abort_event.wait(timeout=0.01):
                    break

            if abort_event.is_set() and not self._is_complete_event.is_set():
                _af_log.info(f'--- AF ABORTED by caller (source={self._run_trigger_source}) ---')
                raise AutofocusAborted('autofocus aborted by caller')

            completed_successfully = True
            return self._best_focus_position

        except AutofocusAborted:
            raise
        except Exception as ex:
            # Restore Z precision ON before propagating so the next
            # protocol Z move stops accurately even if AF threw mid-
            # coarse-pass with precision OFF.
            try:
                self._scope.motion.set_precision_mode('Z', True)
            except Exception:
                logger.debug('[AF] precision restore in error path failed', exc_info=True)
            self._is_focusing_event.clear()
            self._is_complete_event.clear()
            params_repr = repr(getattr(self, '_params', None))[:500]
            logger.exception(
                f'[AF] Error during loop: {type(ex).__name__}: {ex} | _params={params_repr}'
            )
            _af_log.exception(f'AF loop raised: {type(ex).__name__}: {ex} | _params={params_repr}')
            self._notify_af_failure(
                'Autofocus Failed',
                f'Unexpected error during autofocus: {ex}',
            )
            raise

        finally:
            # Save AF characterization data on EVERY exit path (success,
            # abort, exception, degenerate-curve). Queued before the
            # restore chain so the partial-pass data isn't lost if any
            # restore step raises. _save_autofocus_data early-returns
            # when both data lists are empty (true no-data abort).
            if self._save_results_to_file:
                # Promote any unpromoted in-pass samples so a mid-pass
                # abort still leaves diagnostic data on disk.
                if self._af_data_pass:
                    self._af_data_full.extend(self._af_data_pass)
                    self._af_data_pass = []
                try:
                    self._file_io_executor.protocol_put(IOTask(action=self._save_autofocus_data))
                except Exception as ex:
                    logger.warning(f'[AF] Failed to queue autofocus data save: {ex}')

            # Restore LED + camera + Z precision regardless of exit path
            # so the invariant "Z precision ON + pre-AF camera + LED off
            # outside of AF" holds for abort, exception, and success.
            # On non-success exits, also restore Z to the pre-AF position
            # so the user / protocol sees the state they started from.
            # _af_in_progress clears LAST so any caller polling
            # AFE.in_progress() does not race ahead before restoration
            # finishes.
            try:
                self._scope.motion.set_precision_mode('Z', True)
            except Exception:
                logger.debug('[AF] precision restore in finally failed', exc_info=True)
            if not completed_successfully and self._saved_z_position is not None:
                try:
                    # The non-dispatching body: AF runs while the executors
                    # are held by the run, so the public dispatcher would
                    # refuse this restore.
                    self._scope.motion._move_absolute_position_impl('Z', self._saved_z_position)
                    _af_log.info(
                        f'[AF DIAG] Non-success exit: restored Z to '
                        f'pre-AF position {self._saved_z_position:.2f}'
                    )
                except Exception:
                    logger.warning(
                        '[AF] pre-AF Z restore in finally failed; the stage '
                        'may be left at the last AF search position',
                        exc_info=True,
                    )
                    notifications.warning(
                        'Autofocus',
                        'Z Position Not Restored',
                        'Could not restore Z position after autofocus stopped. '
                        'Move Z manually if needed.',
                    )
            # The AF-end LED state is the authority's AF_TO_CAPTURE decision:
            # hold the AF channel for the following capture, or restore the
            # pre-AF snapshot. Hold only on success -- on abort or error the
            # capture never runs, so inheriting would leave the LED lit with no
            # owner to turn it off (overnight sample damage); a non-success
            # exit always restores. The authority's diff is idempotent (a
            # channel already at its target is left untouched, so no off->on
            # blink) and offs whatever is lit but not in the target.
            keep_for_capture = self._keep_led_on and completed_successfully
            illumination = self._scope.illumination
            if self._led_lease is not None:
                af_channel = (
                    illumination.color2ch(self._led_color) if self._led_color is not None else None
                )
                snapshot_lit = (
                    snapshot_lit_pairs(
                        self._saved_led_state.get('states', {}), illumination.color2ch
                    )
                    if self._saved_led_state
                    else frozenset()
                )
                self._led_lease.apply(
                    LedTransition.AF_TO_CAPTURE,
                    LedTransitionCtx(
                        channel=af_channel,
                        mA=self._led_illumination,
                        keep_led_on=keep_for_capture,
                        snapshot_lit=snapshot_lit,
                    ),
                )
            # No lease means the acquire was refused and the run aborted
            # before AF lit anything: there is no AF LED state to restore,
            # and writing here would fight the live holder's lease.
            if self._saved_camera_state:
                restore = self._camera_state_to_restore()
                _af_log.info(
                    f'[AF DIAG] Post-AF camera: keeping step targets '
                    f'gain={self._camera_gain} exp={self._camera_exposure}; '
                    f'restoring {restore or "nothing"} from pre-AF snapshot'
                )
                self._scope.imaging.restore_camera_state(restore)
            _af_log.info(
                f'[AF DIAG] Clearing _af_in_progress -- '
                f'camera now at gain={self._scope.imaging.get_gain()} '
                f'exp={self._scope.imaging.get_exposure_time()}'
            )
            self._af_in_progress.clear()
            # Clear the public ImagingAPI mirror AFTER camera/LED/Z restore
            # finishes, matching _af_in_progress lifecycle.
            self._scope.imaging.is_focusing = False
            # Release the LED lease last. leave_on: the lease does not drive
            # the LEDs yet -- the restore chain above already set the
            # end-state -- so releasing must not turn anything off here.
            if self._led_lease is not None:
                self._led_lease.release(leave_on=True)
                self._led_lease = None
            self._abort_event = None

    def _camera_state_to_restore(self) -> dict:
        """Pre-AF snapshot minus the fields this run explicitly targeted.

        The gain/exposure targets handed to AF are the committed layer or
        step values; the camera must keep them after AF ends. Reverting
        them to the pre-AF snapshot silently undid an exposure the user
        committed by clicking away from the text box just before starting
        AF, leaving the widget and the camera disagreeing. Only fields AF
        never explicitly set fall back to the snapshot.
        """
        restore = dict(self._saved_camera_state or {})
        if self._camera_gain is not None:
            restore.pop('gain_db', None)
        if self._camera_exposure is not None:
            restore.pop('exposure_ms', None)
        return restore

    def get_status(self) -> dict:
        """Get current autofocus status.

        Returns:
            dict with keys: 'state' (idle/focusing/complete), 'best_position',
                  'in_progress'.
        """
        if self._is_complete_event.is_set():
            state = 'complete'
        elif self._is_focusing_event.is_set():
            state = 'focusing'
        else:
            state = 'idle'

        return {
            'state': state,
            'in_progress': self._af_in_progress.is_set(),
            'best_position': self._best_focus_position,
        }

    def run_in_progress(self) -> bool:
        return self._af_in_progress.is_set()

    def _iterate(self):
        if not self._is_focusing_event.is_set():
            return

        if not self._af_in_progress.is_set():
            return

        # Check if Z is still moving (in-memory state check, zero serial I/O
        # when IDLE). Covers both target arrival and overshoot.
        if self._scope.motion.is_moving():
            return

        if self._abort_event is not None and self._abort_event.is_set():
            self._is_focusing_event.clear()
            return

        image = False
        num_retries = 5
        count = 0
        while True:
            # dark_floor_check stays False: AF consumes focus scores, not
            # saved truth, and a hard dark-reject mid-sweep would stall the
            # scan; the mean-intensity retry below handles dark frames.
            # The non-dispatching body, not the public capture_and_wait: AF
            # runs while the camera executor is disabled by the run, so the
            # dispatcher would refuse every grab; the body must run on this
            # thread.
            image = self._scope.imaging._capture_and_wait_impl(
                dark_floor_check=False, exclude_sources=('z_move',)
            )
            count += 1
            if isinstance(image, np.ndarray):
                break

            if count >= num_retries:
                raise Exception('Unable to grab image for autofocusing after max retries')

        height, width = image.shape

        if self._abort_event is not None and self._abort_event.is_set():
            self._is_focusing_event.clear()
            return

        # Detect dark/blank frames -- would score 0, corrupting the curve.
        # Retry once; if still dark, accept (may be genuinely dark sample).
        # capture_and_wait's required dark_floor_check is False here for the
        # same reason as the grab loop above: AF must accept a genuinely dark
        # sample rather than reject the frame.
        mean_intensity = float(np.mean(image))
        if mean_intensity < 1.0:
            _af_log.warning(f'  DARK FRAME: mean={mean_intensity:.2f}, retrying')
            # Non-dispatching body for the same reason as the grab loop
            # above: the camera executor is disabled during the run, so the
            # public dispatcher would refuse this retry.
            retry = self._scope.imaging._capture_and_wait_impl(
                dark_floor_check=False, exclude_sources=('z_move',)
            )
            if isinstance(retry, np.ndarray):
                image = retry

        # Use center quarter of image for focusing
        height, width = image.shape
        image = image[int(height / 4) : int(3 * height / 4), int(width / 4) : int(3 * width / 4)]

        focus_score = autofocus_functions.focus_function(image=image)
        current_pos = round(
            self._scope.motion.get_current_position('Z'), common_utils.max_decimal_precision('z')
        )

        if self.ui_update_func is not None:
            _schedule_ui(lambda dt: self.ui_update_func(pos=current_pos), 0)

        self._af_data_pass.append(
            {
                'position': current_pos,
                'score': focus_score,
            }
        )
        _af_log.info(f'  Z={current_pos:.2f} score={focus_score:.1f}')

        if self._abort_event is not None and self._abort_event.is_set():
            self._is_focusing_event.clear()
            return

        resolution = self._params['resolution']
        next_target = self._scope.motion.get_target_position('Z') + resolution

        if self._abort_event is not None and self._abort_event.is_set():
            self._is_focusing_event.clear()
            self._last_progress_ts = time.monotonic()
            return

        # INTENTIONAL: No early termination on the coarse pass. Real
        # samples have multiple focal planes (cells + debris, thick
        # tissue). Early stop could miss the global peak. The full
        # range must always be swept.

        # Extend scan if peak is at the edge -- we need both sides
        # of the peak for a reliable Gaussian fit. Keep going until
        # we see 2 consecutive drops below 50% of peak.
        if next_target > self._params['z_max'] and len(self._af_data_pass) >= 3:
            pass_scores = [d['score'] for d in self._af_data_pass if np.isfinite(d['score'])]
            if pass_scores:
                pass_max = max(pass_scores)
                peak_idx = pass_scores.index(pass_max)
                # Peak is in the last 2 positions -- extend the scan
                if peak_idx >= len(pass_scores) - 2 and pass_max > 0:
                    recent = pass_scores[-2:]
                    if not all(s < pass_max * 0.5 for s in recent):
                        self._params['z_max'] += resolution
                        _af_log.info(
                            f'  EXTEND: peak at edge, extending z_max to {self._params["z_max"]:.1f}'
                        )
                        self._move_relative_position(pos=resolution)
                        return

        # Measure next step?
        if next_target <= self._params['z_max']:
            self._move_relative_position(pos=resolution)
            return

        # Pass is complete
        self._af_pass_num += 1
        n_pts = len(self._af_data_pass)
        pass_scores = [d['score'] for d in self._af_data_pass]
        peak = max(pass_scores) if pass_scores else 0
        _af_log.info(
            f'  PASS {self._af_pass_num} complete: {n_pts} pts, '
            f'step={resolution:.2f}, peak={peak:.1f}'
        )

        # Adjust the resolution
        prev_resolution = self._params['resolution']
        next_resolution = prev_resolution / 3

        # Bound the resolution to AF_min
        af_min = self._objective['AF_min']
        self._params['resolution'] = max(af_min, next_resolution)

        # Add the scores for the pass to the full dataset and then reset
        # the pass list
        self._af_data_full.extend(self._af_data_pass)
        self._af_data_pass = []

        df = pd.DataFrame(self._af_data_full)

        # Detect degenerate focus curve (all zeros, all NaN, or flat)
        scores = df['score']
        if scores.max() == 0 or scores.isna().all():
            logger.warning(
                'Autofocus: degenerate focus curve (all scores zero or NaN) -- aborting, keeping current Z position'
            )
            _af_log.warning('--- AF ABORT: degenerate curve (all scores zero/NaN) ---')
            self._notify_af_failure(
                'Autofocus Failed',
                'Focus curve is flat or invalid -- check sample and illumination',
            )
            # Restore Z precision ON before bailing so the held
            # current-Z position is reached accurately on any
            # subsequent move.
            self._scope.motion.set_precision_mode('Z', True)
            self._is_focusing_event.clear()
            self._is_complete_event.set()
            self._best_focus_position = self._params['center']
            return

        best_focus_position = self._find_best(df=df)

        if self._last_pass:
            # Move just below the best position so the final approach
            # is upward; this side of the curve is the one the fine
            # pass measured most densely.
            self._move_absolute_position(pos=(best_focus_position - self._params['resolution']))

            af_elapsed = (time.monotonic() - self._af_start_time) * 1000
            _af_log.info(
                f'--- AF DONE best={best_focus_position:.2f}um '
                f'passes={self._af_pass_num} '
                f'total={len(self._af_data_full)} pts '
                f'({af_elapsed:.0f}ms) ---'
            )

            self._move_absolute_position(pos=best_focus_position)

            if self.ui_update_func is not None:
                _schedule_ui(lambda dt: self.ui_update_func(pos=float(best_focus_position)), 0)

            # Data save is queued from run()'s finally block so abort,
            # exception, and degenerate-curve exits also produce a CSV
            # + plot in the per-run subdir. AF Characterization is a
            # diagnostic tool; the failure data is the whole point.

            # Restore Z precision ON as the explicit AF-exit handoff
            # so the invariant "Z precision ON outside of AF" holds
            # regardless of exit path. Idempotent when the fine pass
            # already set it.
            self._scope.motion.set_precision_mode('Z', True)

            self._is_focusing_event.clear()
            self._is_complete_event.set()

            # _af_in_progress is cleared in run()'s finally block
            # after camera state is restored, so callers polling
            # in_progress() do not race ahead before restoration.

            self._best_focus_position = best_focus_position
            return

        self._params['z_min'] = best_focus_position - prev_resolution
        self._params['z_max'] = best_focus_position + prev_resolution

        self._move_absolute_position(pos=self._params['z_min'])
        self._last_progress_ts = time.monotonic()

        if self._params['resolution'] == af_min:
            self._last_pass = True
            # Enable precision mode for the fine pass -- accurate
            # motor stopping for reliable focus measurements.
            self._scope.motion.set_precision_mode('Z', True)
            _af_log.info('  PRECISION MODE ON for fine pass')

    def best_focus_position(self) -> float | None:
        return self._best_focus_position

    def _move_absolute_position(self, pos):
        # Internal-caller contract of the motion API: the public members are
        # dispatchers that serialize EXTERNAL callers onto the io worker,
        # and every internal caller already on a managed thread binds the
        # body directly -- the same contract the AF camera grabs above and
        # the protocol writer follow.
        self._scope.motion._move_absolute_position_impl('Z', pos)
        with self._callbacks_lock:
            cb = self._callbacks.get('move_position')
        if cb is not None:
            _schedule_ui(lambda dt: cb('Z'))

    def _move_relative_position(self, pos):
        # Internal-caller contract of the motion API -- see
        # _move_absolute_position above.
        self._scope.motion._move_relative_position_impl('Z', pos)
        with self._callbacks_lock:
            cb = self._callbacks.get('move_position')
        if cb is not None:
            _schedule_ui(lambda dt: cb('Z'))

    def in_progress(self) -> bool:
        # Use _af_in_progress, not _is_focusing_event. _is_focusing_event
        # is cleared in _iterate() when AF finds the best focus, but the
        # finally block still needs to restore camera state. _af_in_progress
        # is cleared at the END of the finally block, so callers (protocol
        # capture) won't proceed until restore is done.
        return self._af_in_progress.is_set()

    def complete(self) -> bool:
        return self._is_complete_event.is_set()

    def _save_autofocus_data(self):
        if len(self._af_data_full) == 0:
            # No data to save
            return

        ts = self._init_results_dir_and_ts(results_dir=self._results_dir)
        results_file_loc = self._results_dir / f'autofocus_data_{ts}.csv'

        df = pd.DataFrame(self._af_data_full)
        df.to_csv(results_file_loc, header=True, index=False)

        plot_filename = f'autofocus_plot_{ts}.png'
        plot_outfile_loc = self._results_dir / plot_filename

        fig = Figure(figsize=(12, 12))
        axs = fig.add_subplot(111)
        df.reset_index().plot.scatter(x='position', y='score', ax=axs)

        axs.set_title(
            f"""
            Autofocus Characterization
            {plot_filename}
        """,
            fontsize=10,
        )

        axs.set_xlabel('Position (um)')
        axs.set_ylabel('Focus Score')
        axs.grid()

        try:
            fig.savefig(str(plot_outfile_loc), backend='agg')
        except Exception as ex:
            logger.warning(f'[AF] Failed to save autofocus plot: {ex}')
        finally:
            fig.clear()
            del fig

    @staticmethod
    def _find_best(df: pd.DataFrame) -> float:
        # Drop NaN/infinite scores before finding best
        valid = df[df['score'].apply(lambda x: np.isfinite(x))]
        if valid.empty:
            logger.warning(
                'Autofocus: all focus scores are NaN/infinite -- returning first position'
            )
            return df['position'].iloc[0]
        max_score_idx = valid['score'].idxmax()
        raw_best = valid['position'].loc[max_score_idx]

        # Gaussian peak fitting for sub-step interpolation.
        # Fit ln(score) = a*z^2 + b*z + c to points above 50% of peak.
        # Peak of the Gaussian is at z = -b/(2a), giving sub-step resolution.
        try:
            z_vals = valid['position'].values.astype(float)
            scores = valid['score'].values.astype(float)
            peak_score = scores.max()

            if peak_score > 0:
                threshold = peak_score * 0.5
                mask = scores > threshold
                if np.sum(mask) >= 5:
                    z_fit = z_vals[mask]
                    s_fit = scores[mask]
                    s_fit_safe = np.clip(s_fit, 1.0, None)
                    log_s = np.log(s_fit_safe)
                    coeffs = np.polyfit(z_fit, log_s, 2)
                    a, b, _c = coeffs
                    if a < 0:  # concave-down = valid Gaussian peak
                        fit_z = -b / (2 * a)
                        # Sanity: fit peak must be within the measured range
                        z_min, z_max = z_vals.min(), z_vals.max()
                        if z_min <= fit_z <= z_max:
                            shift = abs(fit_z - raw_best)
                            # Sanity: fit shift must be less than the step
                            # spacing between measured points. A larger shift
                            # means the fit is extrapolating beyond the data
                            # -- likely an asymmetric curve fooling the Gaussian.
                            z_diffs = np.diff(np.sort(z_vals))
                            max_shift = np.median(z_diffs) * 2 if len(z_diffs) > 0 else float('inf')
                            if shift <= max_shift:
                                _af_log.info(
                                    f'  FIT: {fit_z:.2f}um '
                                    f'(raw max: {raw_best:.2f}, shift: {shift:.2f}um)'
                                )
                                return float(fit_z)
                            else:
                                _af_log.info(
                                    f'  FIT: {fit_z:.2f}um shift {shift:.2f}um '
                                    f'exceeds max {max_shift:.2f}um, using raw max'
                                )
                        else:
                            _af_log.info(
                                f'  FIT: {fit_z:.2f}um outside range '
                                f'[{z_min:.2f}, {z_max:.2f}], using raw max'
                            )
        except Exception as ex:
            _af_log.info(f'  FIT: failed ({ex}), using raw max')

        return raw_best

    def _reset_state(self):
        self._objective = None
        # Events are recreated each reset so a fresh AF run starts from
        # a known-clear state regardless of how the prior run exited.
        self._is_focusing_event = threading.Event()
        self._is_complete_event = threading.Event()
        self._saved_led_state = None
        self._led_lease = None
        self._saved_camera_state = None
        self._saved_z_position = None
        self._camera_gain = None
        self._camera_exposure = None
        self._af_in_progress.clear()
        self._af_data_pass = []
        self._af_data_full = []
        self._best_focus_position = None
        self._last_pass = False
        self._params = {}
        self._run_trigger_source = None
        self._led_color = None
        self._led_illumination = 0
        with self._callbacks_lock:
            self._callbacks = {}

    def _init_results_dir_and_ts(self, results_dir: pathlib.Path) -> str:
        results_dir.mkdir(exist_ok=True, parents=True)
        now = datetime.datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

    def _allocate_results_dir(self, parent_dir: pathlib.Path) -> pathlib.Path:
        """Allocate a per-run timestamped subdir under parent_dir.

        Eager mkdir so aborted runs still leave a directory marker.
        Same-second collisions retry with `_001`...`_999` suffixes so
        two AF runs in the same wall-clock second do not collide.
        """
        parent_dir = pathlib.Path(parent_dir)
        parent_dir.mkdir(parents=True, exist_ok=True)
        base = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        candidates = [base] + [f'{base}_{i:03d}' for i in range(1, 1000)]
        for candidate in candidates:
            run_dir = parent_dir / candidate
            try:
                run_dir.mkdir(exist_ok=False)
                return run_dir
            except FileExistsError:
                continue
        raise RuntimeError(
            f'Could not allocate an autofocus results folder under '
            f'{parent_dir} (1000 same-second collisions). Try running '
            f'again; if the problem persists, free disk space or restart.'
        )
