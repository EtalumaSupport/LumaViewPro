# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import datetime
import logging
import pathlib
import time
import typing

from lvp_logger import logger

from modules.kivy_utils import schedule_ui as _schedule_ui
from modules.notification_center import notifications

import threading

_af_log = logging.getLogger('LVP.autofocus')

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from modules.sequential_io_executor import SequentialIOExecutor, IOTask

import modules.lumascope_api as lumascope_api
import modules.autofocus_functions as autofocus_functions
import modules.common_utils as common_utils
from modules.objectives_loader import ObjectiveLoader

class AutofocusExecutor:

    def __init__(
        self,
        scope: lumascope_api.Lumascope,
        camera_executor: SequentialIOExecutor,
        io_executor: SequentialIOExecutor,
        file_io_executor: SequentialIOExecutor,
        autofocus_executor: SequentialIOExecutor,
        clock_unschedule_fn: typing.Callable | None = None,
        clock_schedule_interval_fn: typing.Callable | None = None,
        ui_update_func = None
    ):
        # Callback inversion (Architecture Rule 1, LV-31): caller passes
        # the Kivy Clock primitives; this executor never imports kivy.
        # Headless callers pass None — schedule_interval raises and
        # unschedule is a no-op (the executor is then driven directly).
        self._scope = scope
        self._camera_executor = camera_executor
        self._io_executor = io_executor
        self._file_io_executor = file_io_executor
        self._autofocus_executor = autofocus_executor
        self._iterator_scheduled = None
        self.ui_update_func = ui_update_func

        self._af_in_progress = threading.Event()
        self._clock_unschedule_fn = clock_unschedule_fn
        self._clock_schedule_interval_fn = clock_schedule_interval_fn

        self._reset_state()

        if not self._scope.camera_active:
            return

        self._objective_loader = ObjectiveLoader()
        self._reset_state()


    def reset(self):
        if (self._clock_unschedule_fn is not None
                and hasattr(self, '_iterator_scheduled')
                and self._iterator_scheduled is not None):
            self._clock_unschedule_fn(self._iterator_scheduled)
            self._iterator_scheduled = None
        # Restore Z precision mode to the resting default (ON). AF
        # temporarily disables precision during coarse passes for speed
        # and re-enables for the fine pass; this restores ON regardless
        # of which AF phase was interrupted so subsequent protocol Z
        # moves stop accurately.
        try:
            self._scope.set_motor_precision_mode('Z', True)
        except Exception as e:
            logger.debug(f"[AF] Could not restore precision mode during reset (scope may be unavailable): {e}")
        self._reset_state()

    def set_scope(self, scope: lumascope_api.Lumascope):
        self._scope = scope


    def _schedule_interval_func(
        self,
        func: typing.Callable,
        interval_sec: float
    ):
        if self._clock_schedule_interval_fn is not None:
            # Create wrapper method to avoid lambda closure
            def wrapper(dt):
                self._autofocus_executor.protocol_put(IOTask(action=func))
            return self._clock_schedule_interval_fn(wrapper, interval_sec)
        else:
            raise NotImplementedError(
                "AutofocusExecutor was constructed without "
                "clock_schedule_interval_fn; cannot schedule "
                "interval-driven AF (headless mode)"
            )


    def _unschedule_func(
        self,
        func: typing.Callable,
    ):
        if self._clock_unschedule_fn is not None:
            self._clock_unschedule_fn(func)
        else:
            raise NotImplementedError(
                "AutofocusExecutor was constructed without "
                "clock_unschedule_fn; cannot unschedule "
                "interval-driven AF (headless mode)"
            )


    def _calculate_params(self):
        center = self._scope.get_current_position('Z')

        range = self._objective['AF_range']

        if range <= 0:
            raise ValueError(f"AF_range must be positive, got {range}")

        z_min = max(0, center-range)
        z_max = center+range
        resolution = self._objective['AF_max']
        exposure = self._scope.get_exposure_time()

        self._params = {
            'center': center,
            'range': range,
            'z_min': z_min,
            'z_max': z_max,
            'resolution': resolution,
            'exposure': exposure,
        }


    def _led_on(self):
        """Turn on LED for autofocus illumination (if configured).

        Uses owner='autofocus' so only AF can turn this LED off.
        """
        if self._led_color is not None and self._scope.led_connected:
            ch = self._scope.color2ch(self._led_color)
            self._scope.led_on(channel=ch, mA=self._led_illumination,
                               block=True, owner='autofocus')

    def _led_off(self):
        """Turn off only the LED(s) that AF owns (not all LEDs)."""
        if self._led_color is not None and self._scope.led_connected:
            self._scope.leds_off_owned('autofocus')

    def run(
        self,
        objective_id: str,
        callbacks: dict = {},
        save_results_to_file: bool = False,
        run_trigger_source: str = None,
        results_dir: pathlib.Path | None = None,
        led_color: str | None = None,
        led_illumination: float = 0,
        camera_gain: float | None = None,
        camera_exposure: float | None = None,
    ):
        if self._af_in_progress.is_set():
            return

        self._reset_state()
        self._callbacks = callbacks
        self._run_trigger_source = run_trigger_source
        self._led_color = led_color
        self._led_illumination = led_illumination
        self._camera_gain = camera_gain
        self._camera_exposure = camera_exposure
        self._autofocus_executor.protocol_start()
        self._last_progress_ts = time.monotonic()

        if save_results_to_file and results_dir is None:
            raise Exception(f"Cannot save autofocus results to file if results_dir is None")

        self._save_results_to_file = save_results_to_file
        self._results_dir = results_dir
        self._is_focusing_event.set()
        self._af_in_progress.set()

        self._objective = self._objective_loader.get_objective_info(
            objective_id=objective_id
        )

        self._calculate_params()
        self._af_start_time = time.monotonic()
        self._af_pass_num = 0
        _af_log.info(f'--- AF START objective={objective_id} '
                     f'center={self._params["center"]:.1f} '
                     f'range={self._params["range"]:.1f} '
                     f'step={self._params["resolution"]:.1f} '
                     f'z=[{self._params["z_min"]:.1f}, {self._params["z_max"]:.1f}] ---')
        # Save LED + camera state before AF so we can restore after (#608, #610)
        self._saved_led_state = self._scope.save_led_state('autofocus')
        self._saved_camera_state = self._scope.save_camera_state('autofocus')
        # #610 diagnostic: what state did AF just save?
        _af_log.info(f'[AF DIAG] Saved pre-AF camera state: '
                     f'gain={self._saved_camera_state.get("gain", "?")} '
                     f'exp={self._saved_camera_state.get("exposure", "?")} '
                     f'(step wants gain={self._camera_gain} exp={self._camera_exposure})')
        # Apply the step's camera settings so AF scans with correct gain/exposure.
        # Without this, AF inherits whatever the previous protocol step left behind
        # (e.g., Green's gain=12.8/exp=100ms when AF needs BF's gain=0/exp=2ms).
        if self._camera_gain is not None:
            self._scope.set_gain(self._camera_gain)
        if self._camera_exposure is not None:
            self._scope.set_exposure_time(self._camera_exposure)
        # Turn on LED for AF illumination with ownership (#602)
        self._led_on()
        # Drop Z precision for the coarse passes -- the looser stop
        # threshold (VSTOP=1000) saves ~tens of ms per move at the
        # cost of overshoot tolerance, which is fine for the coarse
        # search. The fine pass restores precision ON at line 526
        # below; all exit paths (success, cancel, exception, abort)
        # also restore ON via reset() / explicit setters.
        try:
            self._scope.set_motor_precision_mode('Z', False)
        except Exception as e:
            logger.debug(f"[AF] Could not drop precision mode for coarse passes: {e}")
        self._move_absolute_position(pos=self._params['z_min'])

        # Queue single IOTask that runs the entire autofocus loop
        self._autofocus_executor.protocol_put(IOTask(action=self._autofocus_loop))

    def _autofocus_loop(self):
        """Main autofocus loop - runs continuously until AF completes or is cancelled"""
        last_gc_time = time.monotonic()

        try:
            self._autofocus_loop_inner(last_gc_time)
        finally:
            # Restore LED and camera state to pre-AF values (#602/#608/#610)
            self._led_off()
            if self._saved_led_state:
                self._scope.restore_led_state(self._saved_led_state,
                                              owner='autofocus')
            if self._saved_camera_state:
                # #610 diagnostic: what state is AF about to restore?
                _af_log.info(f'[AF DIAG] Restoring pre-AF camera state: '
                             f'gain={self._saved_camera_state.get("gain", "?")} '
                             f'exp={self._saved_camera_state.get("exposure", "?")}')
                self._scope.restore_camera_state(self._saved_camera_state)
            # Signal AF complete AFTER all state is restored. Previously
            # this was in _iterate() before the finally block ran, creating
            # a race where capture() read stale cached gain while this
            # thread was still restoring the camera to pre-AF settings.
            # (#610 race fix)
            _af_log.info(f'[AF DIAG] Clearing _af_in_progress — '
                         f'camera now at gain={self._scope.get_gain()} '
                         f'exp={self._scope.get_exposure_time()}')
            self._af_in_progress.clear()

    def _autofocus_loop_inner(self, last_gc_time):
        while self._af_in_progress.is_set() and self._is_focusing_event.is_set():
            try:
                # Periodic maintenance: GC every 60 seconds
                if time.monotonic() - last_gc_time > 60:
                    import gc
                    gc.collect()
                    last_gc_time = time.monotonic()

                    # Log queue depths for monitoring
                    try:
                        af_queue_size = self._autofocus_executor.protocol_queue_size()
                        logger.debug(f"[AF Watchdog] AF protocol queue: {af_queue_size}")
                    except Exception:
                        logger.debug("[AF Watchdog] Failed to read protocol queue size", exc_info=True)

                # Run one iteration
                self._iterate()

                # Small delay to prevent CPU throttling
                time.sleep(0.01)

            except Exception as ex:
                # Any unexpected AF error: cleanup so UI is not stuck.
                # Restore Z precision ON so subsequent protocol Z moves
                # aren't left in the low-precision state from the
                # coarse passes that were running when AF threw.
                self._scope.set_motor_precision_mode('Z', True)
                self._autofocus_executor.protocol_end()
                self._autofocus_executor.clear_protocol_pending()
                self._is_focusing_event.clear()
                self._is_complete_event.clear()
                # _af_in_progress is cleared in _autofocus_loop() after
                # camera state is restored (#610 race fix).
                # Surface traceback in both the main log and the AF-
                # specific log so post-mortem readers find it from either
                # entry point. logger.exception emits at ERROR level with
                # the full stack frame; the bare repr of self._params is
                # included so KeyError-on-params bugs identify the missing
                # key + the surrounding state in one read.
                params_repr = repr(getattr(self, '_params', None))[:500]
                logger.exception(
                    f"[AF] Error during loop: {type(ex).__name__}: {ex} "
                    f"| _params={params_repr}"
                )
                _af_log.exception(
                    f"AF loop raised: {type(ex).__name__}: {ex} "
                    f"| _params={params_repr}"
                )
                notifications.error("Autofocus", "Autofocus Failed",
                                    f"Unexpected error during autofocus: {ex}")
                if 'complete' in self._callbacks:
                    _schedule_ui(lambda dt: self._callbacks['complete']())
                break

    def cancel(self):
        """Cancel an in-progress autofocus run."""
        if not self._af_in_progress.is_set():
            return
        _af_log.info('--- AF CANCELLED ---')
        # Restore Z precision ON so subsequent protocol moves stop
        # accurately (coarse passes may have left it OFF).
        self._scope.set_motor_precision_mode('Z', True)
        self._led_off()
        if self._saved_led_state:
            self._scope.restore_led_state(self._saved_led_state,
                                          owner='autofocus')
        if self._saved_camera_state:
            self._scope.restore_camera_state(self._saved_camera_state)
        self._af_in_progress.clear()
        self._is_focusing_event.clear()
        self._autofocus_executor.protocol_end()
        self._autofocus_executor.clear_protocol_pending()

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

    def _iterate(self, dt=None):
            if not self._is_focusing_event.is_set():
                return

            if not self._af_in_progress.is_set():
                return

            # Check if Z is still moving (in-memory state check, zero serial I/O
            # when IDLE). Covers both target arrival and overshoot.
            if self._scope.is_moving():
                return

            if not self._autofocus_executor.is_protocol_running():
                self._is_focusing_event.clear()
                return

            image = False
            num_retries = 5
            count = 0
            while True:
                image = self._scope.capture_and_wait(exclude_sources=('z_move',))
                count += 1
                if isinstance(image, np.ndarray):
                    break

                if count >= num_retries:
                    raise Exception(f"Unable to grab image for autofocusing after max retries")

            height, width = image.shape

            if not self._autofocus_executor.is_protocol_running():
                self._is_focusing_event.clear()
                return

            # Detect dark/blank frames — would score 0, corrupting the curve.
            # Retry once; if still dark, accept (may be genuinely dark sample).
            mean_intensity = float(np.mean(image))
            if mean_intensity < 1.0:
                _af_log.warning(f'  DARK FRAME: mean={mean_intensity:.2f}, retrying')
                retry = self._scope.capture_and_wait(exclude_sources=('z_move',))
                if isinstance(retry, np.ndarray):
                    image = retry

            # Use center quarter of image for focusing
            height, width = image.shape
            image = image[int(height/4):int(3*height/4),int(width/4):int(3*width/4)]

            focus_score = autofocus_functions.focus_function(image=image)
            current_pos = round(self._scope.get_current_position('Z'), common_utils.max_decimal_precision('z'))

            if self.ui_update_func is not None:
                _schedule_ui(lambda dt: self.ui_update_func(pos=current_pos), 0)

            self._af_data_pass.append(
                {
                    'position': current_pos,
                    'score': focus_score,
                }
            )
            _af_log.info(f'  Z={current_pos:.2f} score={focus_score:.1f}')

            if not self._autofocus_executor.is_protocol_running():
                self._is_focusing_event.clear()
                return

            resolution = self._params['resolution']
            next_target = self._scope.get_target_position('Z') + resolution

            if not self._autofocus_executor.is_protocol_running():
                self._is_focusing_event.clear()
                self._last_progress_ts = time.monotonic()
                return

            # INTENTIONAL: No early termination on the coarse pass. Real
            # samples have multiple focal planes (cells + debris, thick tissue).
            # Early stop could miss the global peak. The full range must
            # always be swept. See AF_OPTIMIZATION_PLAN.md "Full Range Sweep Required".

            # Extend scan if peak is at the edge — we need both sides
            # of the peak for a reliable Gaussian fit. Keep going until
            # we see 2 consecutive drops below 50% of peak.
            if next_target > self._params['z_max'] and len(self._af_data_pass) >= 3:
                pass_scores = [d['score'] for d in self._af_data_pass
                               if np.isfinite(d['score'])]
                if pass_scores:
                    pass_max = max(pass_scores)
                    peak_idx = pass_scores.index(pass_max)
                    # Peak is in the last 2 positions — extend the scan
                    if peak_idx >= len(pass_scores) - 2 and pass_max > 0:
                        recent = pass_scores[-2:]
                        if not all(s < pass_max * 0.5 for s in recent):
                            self._params['z_max'] += resolution
                            _af_log.info(f'  EXTEND: peak at edge, extending z_max to {self._params["z_max"]:.1f}')
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
            _af_log.info(f'  PASS {self._af_pass_num} complete: {n_pts} pts, '
                         f'step={resolution:.2f}, peak={peak:.1f}')

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
                logger.warning("Autofocus: degenerate focus curve (all scores zero or NaN) — aborting, keeping current Z position")
                _af_log.warning('--- AF ABORT: degenerate curve (all scores zero/NaN) ---')
                notifications.error("Autofocus", "Autofocus Failed",
                                    "Focus curve is flat or invalid — check sample and illumination")
                # Restore Z precision ON before bailing so the held
                # current-Z position is reached accurately on any
                # subsequent move.
                self._scope.set_motor_precision_mode('Z', True)
                self._is_focusing_event.clear()
                self._is_complete_event.set()
                self._best_focus_position = self._params['center']
                return

            best_focus_position = self._find_best(df=df)

            if self._last_pass:
                if self._clock_unschedule_fn is not None:
                    try:
                        self._clock_unschedule_fn(self._iterator_scheduled)
                    except Exception:
                        logger.warning("[AF] Failed to unschedule Kivy Clock iterator", exc_info=True)

                # Move just underneath focus position to ensure we move UP to final position
                self._move_absolute_position(pos=(best_focus_position-self._params['resolution']))

                af_elapsed = (time.monotonic() - self._af_start_time) * 1000
                _af_log.info(f'--- AF DONE best={best_focus_position:.2f}um '
                             f'passes={self._af_pass_num} '
                             f'total={len(self._af_data_full)} pts '
                             f'({af_elapsed:.0f}ms) ---')

                self._move_absolute_position(pos=best_focus_position)

                # End protocol AFTER final move to prevent race condition (#563)
                self._autofocus_executor.protocol_end()
                self._autofocus_executor.clear_protocol_pending()
                if self.ui_update_func is not None:
                    _schedule_ui(lambda dt: self.ui_update_func(pos=float(best_focus_position)), 0)

                if self._save_results_to_file:
                    # Push file/plot work off the UI thread using the file IO executor
                    try:
                        self._file_io_executor.protocol_put(IOTask(action=self._save_autofocus_data))
                    except Exception as ex:
                        logger.warning(f"[AF] Failed to queue autofocus data save: {ex}")

                # Fine pass just set precision ON for the final move;
                # set it again here as the explicit AF-exit handoff so
                # the invariant "Z precision ON outside of AF" holds
                # regardless of which exit path AF took. Idempotent
                # when the fine-pass setter already ran.
                self._scope.set_motor_precision_mode('Z', True)

                self._is_focusing_event.clear()
                self._is_complete_event.set()

                # _af_in_progress is cleared in _autofocus_loop() after
                # camera state is restored (#610 race fix). Clearing it
                # here let the protocol worker race ahead into capture()
                # while the finally block was still restoring camera state.

                if 'complete' in self._callbacks:
                    _schedule_ui(lambda dt: self._callbacks['complete']())

                self._best_focus_position = best_focus_position
                return

            self._params['z_min'] = best_focus_position - prev_resolution
            self._params['z_max'] = best_focus_position + prev_resolution

            self._move_absolute_position(pos=self._params['z_min'])
            self._last_progress_ts = time.monotonic()

            if self._params['resolution'] == af_min:
                self._last_pass = True
                # Enable precision mode for the fine pass — accurate
                # motor stopping for reliable focus measurements
                self._scope.set_motor_precision_mode('Z', True)
                _af_log.info('  PRECISION MODE ON for fine pass')


    def _tick_iterate(self, dt=None):
        """Callback-based iteration - triggers next iteration without Clock.schedule_interval"""
        # Don't queue if AF is done or stopped
        if not self._af_in_progress.is_set() or not self._is_focusing_event.is_set():
            return

        # Guard against queue buildup
        try:
            if hasattr(self._autofocus_executor, 'protocol_queue_size') and self._autofocus_executor.protocol_queue_size() > 3:
                return
        except Exception:
            logger.debug("[AF] Failed to check protocol queue size", exc_info=True)

        # Periodic maintenance: GC and watchdog logging every 60 seconds
        if not hasattr(self, '_last_gc_time'):
            self._last_gc_time = time.monotonic()

        if time.monotonic() - self._last_gc_time > 60:
            import gc
            gc.collect()
            self._last_gc_time = time.monotonic()

            # Log queue depths for monitoring
            try:
                af_queue_size = self._autofocus_executor.protocol_queue_size()
                logger.debug(f"[AF Watchdog] AF protocol queue: {af_queue_size}")
            except Exception:
                logger.debug("[AF Watchdog] Failed to read protocol queue size", exc_info=True)

        # Queue next iteration with callback to continue the loop
        self._autofocus_executor.protocol_put(IOTask(
            action=self._iterate,
            callback=self._tick_iterate
        ))

    def best_focus_position(self) -> float | None:
        return self._best_focus_position


    def _move_absolute_position(self, pos):
        self._scope.move_absolute_position('Z', pos)
        if 'move_position' in self._callbacks:
            _schedule_ui(lambda dt: self._callbacks['move_position']('Z'))


    def _move_relative_position(self, pos):
        self._scope.move_relative_position('Z', pos)
        if 'move_position' in self._callbacks:
            _schedule_ui(lambda dt: self._callbacks['move_position']('Z'))


    def in_progress(self) -> bool:
        # Use _af_in_progress, not _is_focusing_event. _is_focusing_event
        # is cleared in _iterate() when AF finds the best focus, but the
        # finally block in _autofocus_loop() still needs to restore camera
        # state. _af_in_progress is cleared at the END of the finally block,
        # so callers (protocol capture) won't proceed until restore is done.
        # (#610 race fix)
        return self._af_in_progress.is_set()


    def complete(self) -> bool:
        return self._is_complete_event.is_set()


    def _save_autofocus_data(self):
        if len(self._af_data_full) == 0:
            # No data to save
            return

        ts = self._init_results_dir_and_ts(results_dir=self._results_dir)
        results_file_loc = self._results_dir / f"autofocus_data_{ts}.csv"

        df = pd.DataFrame(self._af_data_full)
        df.to_csv(results_file_loc, header=True, index=False)

        plot_filename = f"autofocus_plot_{ts}.png"
        plot_outfile_loc = self._results_dir / plot_filename
        from matplotlib.figure import Figure
        fig = Figure(figsize=(12, 12))
        axs = fig.add_subplot(111)
        df.reset_index().plot.scatter(x="position", y="score", ax=axs)

        axs.set_title(f"""
            Autofocus Characterization
            {plot_filename}
        """, fontsize=10)

        axs.set_xlabel("Position (um)")
        axs.set_ylabel("Focus Score")
        axs.grid()

        try:
            fig.savefig(str(plot_outfile_loc), backend='agg')
        except Exception as ex:
            logger.warning(f"[AF] Failed to save autofocus plot: {ex}")
        finally:
            fig.clear()
            del fig


    @staticmethod
    def _find_best(df: pd.DataFrame) -> float:
        # Drop NaN/infinite scores before finding best
        valid = df[df['score'].apply(lambda x: np.isfinite(x))]
        if valid.empty:
            logger.warning("Autofocus: all focus scores are NaN/infinite — returning first position")
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
                    a, b, c = coeffs
                    if a < 0:  # concave-down = valid Gaussian peak
                        fit_z = -b / (2 * a)
                        # Sanity: fit peak must be within the measured range
                        z_min, z_max = z_vals.min(), z_vals.max()
                        if z_min <= fit_z <= z_max:
                            shift = abs(fit_z - raw_best)
                            # Sanity: fit shift must be less than the step
                            # spacing between measured points. A larger shift
                            # means the fit is extrapolating beyond the data
                            # — likely an asymmetric curve fooling the Gaussian.
                            z_diffs = np.diff(np.sort(z_vals))
                            max_shift = np.median(z_diffs) * 2 if len(z_diffs) > 0 else float('inf')
                            if shift <= max_shift:
                                _af_log.info(f'  FIT: {fit_z:.2f}um '
                                            f'(raw max: {raw_best:.2f}, shift: {shift:.2f}um)')
                                return float(fit_z)
                            else:
                                _af_log.info(f'  FIT: {fit_z:.2f}um shift {shift:.2f}um '
                                            f'exceeds max {max_shift:.2f}um, using raw max')
                        else:
                            _af_log.info(f'  FIT: {fit_z:.2f}um outside range '
                                        f'[{z_min:.2f}, {z_max:.2f}], using raw max')
        except Exception as ex:
            _af_log.info(f'  FIT: failed ({ex}), using raw max')

        return raw_best


    def _reset_state(self):
        self._objective = None
        self._is_focusing_event = threading.Event()   # thread-safe (#607)
        self._is_complete_event = threading.Event()    # thread-safe (#607)
        self._saved_led_state = None
        self._saved_camera_state = None
        self._camera_gain = None
        self._camera_exposure = None
        self._af_in_progress.clear()
        self._af_data_pass = []
        self._af_data_full = []
        self._best_focus_position = None # Last / Previous focus score
        self._last_pass = False         # Are we on the last scan for autofocus?
        self._params = {}
        self._run_trigger_source = None
        self._led_color = None
        self._led_illumination = 0
        self._autofocus_executor.protocol_end()
        self._autofocus_executor.clear_protocol_pending()
        try:
            if self._clock_unschedule_fn is not None:
                self._clock_unschedule_fn(self._iterator_scheduled)
        except Exception:
            logger.warning("[AF] Failed to unschedule Kivy Clock iterator during stop", exc_info=True)

    def _init_results_dir_and_ts(self, results_dir: pathlib.Path) -> str:
        results_dir.mkdir(exist_ok=True, parents=True)
        now = datetime.datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")

