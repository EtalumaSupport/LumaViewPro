
import datetime
import importlib
import pathlib
import time
import typing

from lvp_logger import logger

import threading

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from modules.sequential_io_executor import SequentialIOExecutor, IOTask

import lumascope_api
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

        self._reset_state()

        if not self._scope.camera.active:
            return

        if self._use_kivy_clock:
            self._kivy_clock_module = importlib.import_module('kivy.clock')

        self._objective_loader = ObjectiveLoader()
        self._reset_state()


    def reset(self):
        if hasattr(self, '_iterator_scheduled') and self._iterator_scheduled is not None:
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
                        pass

                # Run one iteration
                self._iterate()

                # Small delay to prevent CPU throttling
                time.sleep(0.01)

            except Exception as ex:
                # Any unexpected AF error: cleanup so UI is not stuck
                self._autofocus_executor.protocol_end()
                self._autofocus_executor.clear_protocol_pending()
                self._is_focusing = False
                self._is_complete = False
                # Surface error in logs; UI callback (if present) will clear button
                import logging as _logging
                _logging.getLogger().error(f"[AF] Error during loop: {ex}", exc_info=True)
                if 'complete' in self._callbacks:
                    self._kivy_clock_module.Clock.schedule_once(lambda dt: self._callbacks['complete'](), 0)
                break

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

            if not self._scope.get_target_status('Z'):
                return

            if self._scope.get_overshoot():
                return

            if not self._autofocus_executor.is_protocol_running():
                self._is_focusing = False
                return

            # Sleep for at least 75ms to ensure that the camera is ready for the next capture
            #time.sleep(max(self._params['exposure']/1000, 0.075))

            image = False
            num_retries = 5
            count = 0
            while True:
                image = self._scope.get_image(force_new_capture=True)
                count += 1
                if type(image) == np.ndarray:
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

            logger.info(f"[AF] _iterate scheduling ui update to: {current_pos}")
            self._kivy_clock_module.Clock.schedule_once(lambda dt: self.ui_update_func(pos=current_pos), 0)

            self._af_data_pass.append(
                {
                    'position': current_pos,
                    'score': focus_score,
                }
            )

            if not self._autofocus_executor.is_protocol_running():
                self._is_focusing = False
                return

            resolution = self._params['resolution']
            next_target = self._scope.get_target_position('Z') + resolution

            if not self._autofocus_executor.is_protocol_running():
                self._is_focusing = False
                self._last_progress_ts = time.monotonic()
                return

            # Measure next step?
            if next_target <= self._params['z_max']:
                self._move_relative_position(pos=resolution)
                return

            # Pass is complete

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
            best_focus_position = self._find_best(df=df)

            if self._last_pass == True:
                try:
                    self._kivy_clock_module.Clock.unschedule(self._iterator_scheduled)
                except Exception:
                    pass

                # Move just underneath focus position to ensure we move UP to final position
                self._move_absolute_position(pos=(best_focus_position-self._params['resolution']))

                self._autofocus_executor.protocol_end()
                self._autofocus_executor.clear_protocol_pending()

                logger.info(f"[AF] Autofocus complete. Best focus position: {best_focus_position} um")

                self._move_absolute_position(pos=best_focus_position)
                logger.info(f"[AF] _iterate last_pass scheduling ui update to: {float(best_focus_position)}")
                self._kivy_clock_module.Clock.schedule_once(lambda dt: self.ui_update_func(pos=float(best_focus_position)), 0)

                if self._save_results_to_file:
                    # Push file/plot work off the UI thread using the file IO executor
                    try:
                        self._file_io_executor.protocol_put(IOTask(action=self._save_autofocus_data))
                    except Exception:
                        pass

                self._is_focusing = False
                self._is_complete = True

                #self._af_in_progress.clear()

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
            pass

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
                pass

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

        fig.savefig(str(plot_outfile_loc), backend='agg')
        try:
            fig.clear()
        except Exception:
            pass


    @staticmethod
    def _find_best(df: pd.DataFrame) -> float:
        max_score_idx = df['score'].idxmax()
        max_position = df['position'].loc[max_score_idx]
        return max_position


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
            pass



    def _init_results_dir_and_ts(self, results_dir: pathlib.Path) -> str:
        results_dir.mkdir(exist_ok=True, parents=True)
        now = datetime.datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")

