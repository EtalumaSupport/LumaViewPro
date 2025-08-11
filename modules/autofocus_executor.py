
import datetime
import importlib
import pathlib
import time
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.sequential_io_executor import SequentialIOExecutor, IOTask

import lumascope_api
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
            return self._kivy_clock_module.Clock.schedule_interval(lambda dt: self._autofocus_executor.protocol_put(IOTask(action=func)), interval_sec)
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
        results_dir: pathlib.Path | None = None,
    ):
        self._reset_state()
        self._callbacks = callbacks
        self._autofocus_executor.protocol_start()

        if save_results_to_file and results_dir is None:
            raise Exception(f"Cannot save autofocus results to file if results_dir is None")
        
        self._save_results_to_file = save_results_to_file
        self._results_dir = results_dir
        self._is_focusing = True

        self._objective = self._objective_loader.get_objective_info(
            objective_id=objective_id
        )

        self._calculate_params()
        self._move_absolute_position(pos=self._params['z_min'])

        self._iterator_scheduled = self._kivy_clock_module.Clock.schedule_interval(lambda dt: self._autofocus_executor.protocol_put(IOTask(action=self._iterate)), 0.01)
        # self._iterator_scheduled = self._schedule_interval_func(
        #     func=self._iterate,
        #     interval_sec=0.01,
        # )


    def _iterate(self, dt=None):

        if not self._is_focusing:
            return

        if not self._scope.get_target_status('Z'):
            return
        
        if self._scope.get_overshoot():
            return
        
        if not self._autofocus_executor.is_protocol_running():
            self._is_focusing = False
            return
        
        
        exposure_delay = 2*self._params['exposure']/1000+0.2
        time.sleep(exposure_delay)

        image = False
        num_retries = 5
        count = 0
        while True:
            image = self._scope.get_image()
            count += 1
            if type(image) == np.ndarray:
                break

            if count >= num_retries:
                # Failed to grab image after max retries
                # TODO
                raise Exception(f"Unable to grab image for autofocusing after max retries")
            
        height, width = image.shape

        if not self._autofocus_executor.is_protocol_running():
            self._is_focusing = False
            return
        
        # Use center quarter of image for focusing
        image = image[int(height/4):int(3*height/4),int(width/4):int(3*width/4)]

        focus = self.focus_function(image=image)
        current_pos = round(self._scope.get_current_position('Z'), common_utils.max_decimal_precision('z'))

        self._kivy_clock_module.Clock.schedule_once(lambda dt: self.ui_update_func(pos=current_pos), 0)

        self._af_data_pass.append(
            {
                'position': current_pos,
                'score': focus,
            }
        )

        if not self._autofocus_executor.is_protocol_running():
            self._is_focusing = False
            return
        
        resolution = self._params['resolution']
        next_target = self._scope.get_target_position('Z') + resolution

        if not self._autofocus_executor.is_protocol_running():
            self._is_focusing = False
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

        df = pd.DataFrame(self._af_data_pass)
        best_focus_position = self._find_best(df=df)

        if self._last_pass == True:
            #self._unschedule_func(func=self._iterator_scheduled)
            self._kivy_clock_module.Clock.unschedule(self._iterator_scheduled)
            self._autofocus_executor.protocol_end()
            self._autofocus_executor.clear_protocol_pending()
            self._move_absolute_position(pos=best_focus_position)
            self._kivy_clock_module.Clock.schedule_once(lambda dt: self.ui_update_func(pos=float(best_focus_position)), 0)

            if self._save_results_to_file:
                self._save_autofocus_data()

            self._is_focusing = False
            self._is_complete = True
            if 'complete' in self._callbacks:
                self._callbacks['complete']()

            self._best_focus_position = best_focus_position
            return

        self._params['z_min'] = best_focus_position - prev_resolution
        self._params['z_max'] = best_focus_position + prev_resolution

        # Add the scores for the pass to the full dataset and then reset
        # the pass list
        self._af_data_full.extend(self._af_data_pass)
        self._af_data_pass = []

        self._move_absolute_position(pos=self._params['z_min'])

        if self._params['resolution'] == af_min:
            self._last_pass = True


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
        fig, axs = plt.subplots(figsize=(12,12))
        df.reset_index().plot.scatter(x="position", y="score", ax=axs)
        
        axs.set_title(f"""
            Autofocus Characterization
            {plot_filename}
        """, fontsize=10)

        axs.set_xlabel("Position (um)")
        axs.set_ylabel("Focus Score")
        axs.grid()

        fig.savefig(str(plot_outfile_loc))
        plt.close()


    @staticmethod
    def _find_best(df: pd.DataFrame) -> float:
        max_score_idx = df['score'].idxmax()
        max_position = df['position'].loc[max_score_idx]
        return max_position


    def focus_function(
        self,
        image: np.ndarray,
        algorithm: str = 'vollath4',
    ):
        # TODO the w/h seem swapped, but this is how the original code was written.
        # Needs further investigation to clarify.
        w, h = image.shape

        # Journal of Microscopy, Vol. 188, Pt 3, December 1997, pp. 264–272
        if algorithm == 'vollath4': # pg 266
            image = np.double(image)
            sum_one = np.sum(np.multiply(image[:w-1,:h], image[1:w,:h])) # g(i, j).g(i+1, j)
            sum_two = np.sum(np.multiply(image[:w-2,:h], image[2:w,:h])) # g(i, j).g(i+2, j)
            return sum_one - sum_two

        elif algorithm == 'skew':
            hist = np.histogram(image, bins=256,range=(0,256))
            hist = np.asarray(hist[0], dtype='int')
            max_index = hist.argmax()

            edges = np.histogram_bin_edges(image, bins=1)
            white_edge = edges[1]

            skew = white_edge-max_index
            return skew

        elif algorithm == 'pixel_variation':
            sum = np.sum(image)
            ssq = np.sum(np.square(image))
            var = ssq*w*h-sum**2
            return var
      
        else:
            return 0


    def _reset_state(self):
        self._objective = None
        self._is_focusing = False
        self._is_complete = False
        self._af_data_pass = []
        self._af_data_full = []
        self._best_focus_position = None # Last / Previous focus score
        self._last_pass = False         # Are we on the last scan for autofocus?
        self._params = {}
        self._autofocus_executor.clear_protocol_pending()



    def _init_results_dir_and_ts(self, results_dir: pathlib.Path) -> str:
        results_dir.mkdir(exist_ok=True, parents=True)
        now = datetime.datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")
    
