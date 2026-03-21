# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import datetime
import importlib
import logging
import pathlib
import time
import typing

from lvp_logger import logger

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
        use_kivy_clock: bool = False,
        ui_update_func = None
    ):
        self._scope = scope
        self._use_kivy_clock = use_kivy_clock
        self._camera_executor = camera_executor
        self._io_executor = io_executor
        self._file_io_executor = file_io_executor
        self._autofocus_executor = autofocus_executor
        self._iterator_scheduled = None
        self.ui_update_func = ui_update_func

        self._af_in_progress = threading.Event()
        self._kivy_clock_module = None

        if self._use_kivy_clock:
            self._kivy_clock_module = importlib.import_module('kivy.clock')

        self._reset_state()

        if not self._scope.camera_active:
            return

        self._objective_loader = ObjectiveLoader()
        self._reset_state()


    def reset(self):
        if self._use_kivy_clock and self._kivy_clock_module and hasattr(self, '_iterator_scheduled') and self._iterator_scheduled is not None:
            self._kivy_clock_module.Clock.unschedule(self._iterator_scheduled)
            self._iterator_scheduled = None
        self._reset_state()

    def set_scope(self, scope: lumascope_api.Lumascope):
        self._scope = scope


    def _schedule_interval_func(
        self,
        func: typing.Callable,
        interval_sec: float
    ):
        if self._use_kivy_clock:
            # Create wrapper method to avoid lambda closure
            def wrapper(dt):
                self._autofocus_executor.protocol_put(IOTask(action=func))
            return self._kivy_clock_module.Clock.schedule_interval(wrapper, interval_sec)
        else:
            raise NotImplementedError(f"Not implemented for support outside Kivy")


    def _unschedule_func(
        self,
        func: typing.Callable,
    ):
        if self._use_kivy_clock:
            self._kivy_clock_module.Clock.unschedule(func)
        else:
            raise NotImplementedError(f"Not implemented for support outside Kivy")


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


    def run(
        self,
        objective_id: str,
        callbacks: dict = {},
        save_results_to_file: bool = False,
        run_trigger_source: str = None,
        results_dir: pathlib.Path | None = None,
    ):
        if self._af_in_progress.is_set():
            return

        #logger.error(f"AF RUN")
        self._reset_state()
        self._callbacks = callbacks
        self._run_trigger_source = run_trigger_source
        self._autofocus_executor.protocol_start()
        self._last_progress_ts = time.monotonic()

        if save_results_to_file and results_dir is None:
            raise Exception(f"Cannot save autofocus results to file if results_dir is None")

        self._save_results_to_file = save_results_to_file
        self._results_dir = results_dir
        self._is_focusing = True
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
        self._move_absolute_position(pos=self._params['z_min'])

        # Queue single IOTask that runs the entire autofocus loop
        self._autofocus_executor.protocol_put(IOTask(action=self._autofocus_loop))

    def _autofocus_loop(self):
        """Main autofocus loop - runs continuously until AF completes or is cancelled"""
        last_gc_time = time.monotonic()

        while self._af_in_progress.is_set() and self._is_focusing:
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
                # Any unexpected AF error: cleanup so UI is not stuck
                self._scope.set_motor_precision_mode('Z', False)
                self._autofocus_executor.protocol_end()
                self._autofocus_executor.clear_protocol_pending()
                self._is_focusing = False
                self._is_complete = False
                self._af_in_progress.clear()
                # Surface error in logs; UI callback (if present) will clear button
                import logging as _logging
                _logging.getLogger().error(f"[AF] Error during loop: {ex}", exc_info=True)
                if 'complete' in self._callbacks:
                    if self._use_kivy_clock:
                        self._kivy_clock_module.Clock.schedule_once(lambda dt: self._callbacks['complete'](), 0)
                    else:
                        self._callbacks['complete']()
                break

    def cancel(self):
        """Cancel an in-progress autofocus run."""
        if not self._af_in_progress.is_set():
            return
        _af_log.info('--- AF CANCELLED ---')
        self._scope.set_motor_precision_mode('Z', False)
        self._af_in_progress.clear()
        self._is_focusing = False
        self._autofocus_executor.protocol_end()
        self._autofocus_executor.clear_protocol_pending()

    def get_status(self) -> dict:
        """Get current autofocus status.

        Returns:
            dict with keys: 'state' (idle/focusing/complete), 'best_position',
                  'in_progress'.
        """
        if self._is_complete:
            state = 'complete'
        elif self._is_focusing:
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
        # # Progress timeout: if AF does not advance for a while, cancel gracefully
        # if hasattr(self, '_last_progress_ts') and (time.monotonic() - self._last_progress_ts > 15):
        #     try:
        #         self._kivy_clock_module.Clock.unschedule(self._iterator_scheduled)
        #     except Exception:
        #         pass
        #     self._autofocus_executor.protocol_end()
        #     self._autofocus_executor.clear_protocol_pending()
        #     self._is_focusing = False
        #     self._is_complete = False
        #     if 'complete' in self._callbacks:
        #         # Use UI thread to reset button if caller wired it
        #         self._kivy_clock_module.Clock.schedule_once(lambda dt: self._callbacks['complete'](), 0)
        #     return

            if not self._is_focusing:
                return

            if not self._af_in_progress.is_set():
                return

            # Check if Z is still moving (in-memory state check, zero serial I/O
            # when IDLE). Covers both target arrival and overshoot.
            if self._scope.is_moving():
                return

            if not self._autofocus_executor.is_protocol_running():
                self._is_focusing = False
                return

            # Wait for mechanical settle after motor reports arrival.
            # With Vstop=1000 the motor may still be decelerating through
            # the last microsteps when target_reached fires. Drain 2 frames
            # to let the stage physically stop before scoring.
            # TODO: reduce/remove after Vstop is lowered in firmware
            self._scope.get_image(force_new_capture=True)  # drain frame 1
            self._scope.get_image(force_new_capture=True)  # drain frame 2

            image = False
            num_retries = 5
            count = 0
            while True:
                image = self._scope.get_image(force_new_capture=True)
                count += 1
                if isinstance(image, np.ndarray):
                    break

                if count >= num_retries:
                    raise Exception(f"Unable to grab image for autofocusing after max retries")

            height, width = image.shape

            if not self._autofocus_executor.is_protocol_running():
                self._is_focusing = False
                return

            # Use center quarter of image for focusing
            image = image[int(height/4):int(3*height/4),int(width/4):int(3*width/4)]

            focus_score = autofocus_functions.focus_function(image=image)
            current_pos = round(self._scope.get_current_position('Z'), common_utils.max_decimal_precision('z'))

            if self._use_kivy_clock:
                self._kivy_clock_module.Clock.schedule_once(lambda dt: self.ui_update_func(pos=current_pos), 0)
            elif self.ui_update_func:
                self.ui_update_func(pos=current_pos)

            self._af_data_pass.append(
                {
                    'position': current_pos,
                    'score': focus_score,
                }
            )
            _af_log.info(f'  Z={current_pos:.2f} score={focus_score:.1f}')

            if not self._autofocus_executor.is_protocol_running():
                self._is_focusing = False
                return

            resolution = self._params['resolution']
            next_target = self._scope.get_target_position('Z') + resolution

            if not self._autofocus_executor.is_protocol_running():
                self._is_focusing = False
                self._last_progress_ts = time.monotonic()
                return

            # Early termination: if score has dropped well below the pass
            # peak for 2+ consecutive positions, we've passed the focus and
            # can skip the rest of this sweep.
            early_stop = False
            if len(self._af_data_pass) >= 4:
                pass_scores = [d['score'] for d in self._af_data_pass
                               if np.isfinite(d['score'])]
                if pass_scores:
                    pass_max = max(pass_scores)
                    if pass_max > 0:
                        recent = pass_scores[-2:]
                        if all(s < pass_max * 0.5 for s in recent):
                            early_stop = True
                            _af_log.info(f'  EARLY STOP: score {recent[-1]:.0f} '
                                        f'({recent[-1]/pass_max*100:.0f}% of peak {pass_max:.0f})')

            # Measure next step?
            if next_target <= self._params['z_max'] and not early_stop:
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
                self._scope.set_motor_precision_mode('Z', False)
                self._is_focusing = False
                self._is_complete = True
                self._best_focus_position = self._params['center']
                return

            best_focus_position = self._find_best(df=df)

            if self._last_pass:
                if self._use_kivy_clock:
                    try:
                        self._kivy_clock_module.Clock.unschedule(self._iterator_scheduled)
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
                if self._use_kivy_clock:
                    self._kivy_clock_module.Clock.schedule_once(lambda dt: self.ui_update_func(pos=float(best_focus_position)), 0)
                elif self.ui_update_func:
                    self.ui_update_func(pos=float(best_focus_position))

                if self._save_results_to_file:
                    # Push file/plot work off the UI thread using the file IO executor
                    try:
                        self._file_io_executor.protocol_put(IOTask(action=self._save_autofocus_data))
                    except Exception as ex:
                        logger.warning(f"[AF] Failed to queue autofocus data save: {ex}")

                # Restore normal motor mode after fine pass
                self._scope.set_motor_precision_mode('Z', False)

                self._is_focusing = False
                self._is_complete = True

                self._af_in_progress.clear()

                if 'complete' in self._callbacks:
                    self._callbacks['complete']()

                self._best_focus_position = best_focus_position
                return

            self._params['z_min'] = best_focus_position - prev_resolution
            self._params['z_max'] = best_focus_position + prev_resolution



            #logger.error(f"[AF] Moving to z_min: {self._params['z_min']} um")
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
        if not self._af_in_progress.is_set() or not self._is_focusing:
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
            self._callbacks['move_position']('Z')


    def _move_relative_position(self, pos):
        self._scope.move_relative_position('Z', pos)
        if 'move_position' in self._callbacks:
            self._callbacks['move_position']('Z')


    def in_progress(self) -> bool:
        return self._is_focusing


    def complete(self) -> bool:
        return self._is_complete


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
                            _af_log.info(f'  FIT: {fit_z:.2f}um '
                                        f'(raw max: {raw_best:.2f}, shift: {shift:.2f}um)')
                            return float(fit_z)
                        else:
                            _af_log.info(f'  FIT: {fit_z:.2f}um outside range '
                                        f'[{z_min:.2f}, {z_max:.2f}], using raw max')
        except Exception as ex:
            _af_log.info(f'  FIT: failed ({ex}), using raw max')

        return raw_best


    def _reset_state(self):
        self._objective = None
        self._is_focusing = False
        self._is_complete = False
        self._af_in_progress.clear()
        self._af_data_pass = []
        self._af_data_full = []
        self._best_focus_position = None # Last / Previous focus score
        self._last_pass = False         # Are we on the last scan for autofocus?
        self._params = {}
        self._run_trigger_source = None
        self._autofocus_executor.protocol_end()
        self._autofocus_executor.clear_protocol_pending()
        try:
            if self._use_kivy_clock:
                self._kivy_clock_module.Clock.unschedule(self._iterator_scheduled)
        except Exception:
            logger.warning("[AF] Failed to unschedule Kivy Clock iterator during stop", exc_info=True)



    def _init_results_dir_and_ts(self, results_dir: pathlib.Path) -> str:
        results_dir.mkdir(exist_ok=True, parents=True)
        now = datetime.datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")

