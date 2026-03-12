# Copyright Etaluma, Inc.
import json
import logging
import os
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')  # Must be set before pyplot import to avoid Tk/macOS conflict
import matplotlib.pyplot as plt
from matplotlib.dates import ConciseDateFormatter
import numpy as np
import pandas as pd

from kivy.clock import Clock
from kivy.properties import BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup

from ui.progress_popup import show_popup
from modules.sequential_io_executor import IOTask
from modules.stitcher import Stitcher
from modules.composite_generation import CompositeGeneration
from modules.video_builder import VideoBuilder
from modules.json_helper import CustomJSONizer
import modules.zprojector as zprojector
import modules.post_processing as post_processing
import modules.image_utils as image_utils
import modules.image_utils_kivy as image_utils_kivy
import modules.app_context as _app_ctx

logger = logging.getLogger('LVP.ui.post_processing')


class StitchControls(BoxLayout):

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import lumaviewpro
        lumaviewpro.stitch_controls = self


    def set_button_enabled_state(self, state: bool):
        self.ids['stitch_apply_btn'].disabled = not state


    @show_popup
    def run_stitcher(self, popup, path):
        ctx = _app_ctx.ctx
        status_map = {
            True: "Success",
            False: "FAILED"
        }
        popup.title = "Stitcher"
        popup.text = "Generating stitched images..."
        popup.progress = 0
        popup.auto_dismiss = False

        stitcher = Stitcher(
            has_turret=ctx.lumaview.scope.has_turret(),
        )
        # result = stitcher.load_folder(
        #     path=pathlib.Path(path),
        #     tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json"
        # )
        ctx.file_io_executor.put(IOTask(action=stitcher.load_folder,
                             args=(pathlib.Path(path),
                                    pathlib.Path(ctx.source_path) / "data" / "tiling.json",
                                    popup
                                    ),
                             callback=self.stitcher_callback,
                             cb_args=(popup, status_map),
                             pass_result=True))


    def stitcher_callback(self, popup, status_map, result=None, exception=None):
        if result is None:
            popup.text = "Stitching images - FAILED"
            Clock.schedule_once(lambda dt: popup.dismiss(), 5)
            return

        final_text = f"Stitching images - {status_map[result['status']]}"
        if result['status'] is False:
            final_text += f"\n{result['message']}"
            popup.text = final_text
            Clock.schedule_once(lambda dt: popup.dismiss(), 5)
            return

        popup.text = final_text
        Clock.schedule_once(lambda dt: popup.dismiss(), 2)


class ZProjectionControls(BoxLayout):

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import lumaviewpro
        lumaviewpro.zprojection_controls = self
        Clock.schedule_once(self._init_ui, 0)
        self.ij_initialized = False
        self.ij_buffer_event = None
        self.ij_buffer_count = 0
        self.ij_buffer_interval = 0.5


    def _init_ui(self, dt=0):
        self.ids['zprojection_method_spinner'].values = zprojector.ZProjector.methods()
        self.ids['zprojection_method_spinner'].text = zprojector.ZProjector.methods()[1]


    @show_popup
    def run_zprojection(self, popup, path):
        ctx = _app_ctx.ctx
        popup.title = "Z-Projection"
        popup.progress = 0
        popup.auto_dismiss = False

        if ctx.ij_helper is None:
            popup.text = "     ImageJ is not initialized.\n" + \
                         "Please wait for ImageJ to initialize.\n" + \
                         "   Note: This may take some time.\n" + \
                         "                  \n" + \
                         "               "
            self.ij_initialized = False
            # Run imagej initialization in a separate thread
            # Callback to finish zprojection when imagej is initialized
            from modules.imagej_helper import init_ij
            _app_ctx.ctx.file_io_executor.put(IOTask(action=init_ij, callback=self.zprojection_with_imagej, cb_args=(popup, path)))
            self.ij_buffer_event = Clock.schedule_interval(lambda dt: self.waiting_for_imagej(popup), self.ij_buffer_interval)
            return

        self.ij_initialized = True
        # Imagej already initialized, run zprojection
        self.zprojection_with_imagej(popup, path)

    def waiting_for_imagej(self, popup):
        if self.ij_initialized:
            Clock.unschedule(self.ij_buffer_event)
            self.ij_buffer_event = None
            self.ij_buffer_count = 0
            return

        popup.text =     "ImageJ is not initialized. Please wait for ImageJ to initialize.\n" + \
                         "                Note: This may take some time.\n" + \
                         "                  \n" + \
                         "                      " + "o   "*self.ij_buffer_count
        self.ij_buffer_count += 1
        if self.ij_buffer_count > 3:
            self.ij_buffer_count = 0

        return

    def zprojection_with_imagej(self, popup, path):
        ctx = _app_ctx.ctx
        status_map = {
            True: "Success",
            False: "FAILED"
        }

        if ctx.ij_helper is not None:
            self.ij_initialized = True
            Clock.unschedule(self.ij_buffer_event)
            self.ij_buffer_event = None
            self.ij_buffer_count = 0

        if self.ij_buffer_event is not None:
            Clock.unschedule(self.ij_buffer_event)
            self.ij_buffer_event = None
            self.ij_buffer_count = 0

        if ctx.ij_helper is None:
            popup.text = "Failed to initialize ImageJ. Please try again."
            Clock.schedule_once(lambda dt: popup.dismiss(), 5)
            return

        popup.text = "Generating Z-Projection images..."

        zproj = zprojector.ZProjector(
            has_turret=ctx.lumaview.scope.has_turret(),
            ij_helper=ctx.ij_helper
        )
        # result = zproj.load_folder(
        #     path=pathlib.Path(path),
        #     tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
        #     method=self.ids['zprojection_method_spinner'].text
        # )

        ctx.file_io_executor.put(IOTask(action=zproj.load_folder,
                             args=(pathlib.Path(path),
                                    pathlib.Path(ctx.source_path) / "data" / "tiling.json",
                                    popup
                                    ),
                             kwargs={
                                "method": self.ids['zprojection_method_spinner'].text
                             },
                             callback=self.zprojection_callback,
                             cb_args=(popup, status_map),
                             pass_result=True))

    def zprojection_callback(self, popup, status_map, result=None, exception=None):
        popup.progress = 100
        if result is None:
            popup.text = "Generating Z-Projection images - FAILED"
            Clock.schedule_once(lambda dt: popup.dismiss(), 5)
            return

        final_text = f"Generating Z-Projection images - {status_map[result['status']]}"
        if result['status'] is False:
            final_text += f"\n{result['message']}"
            popup.text = final_text
            Clock.schedule_once(lambda dt: popup.dismiss(), 5)
            return

        popup.text = final_text
        Clock.schedule_once(lambda dt: popup.dismiss(), 2)
        return

class CompositeGenControls(BoxLayout):

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import lumaviewpro
        lumaviewpro.composite_gen_controls = self


    @show_popup
    def run_composite_gen(self, popup, path):
        ctx = _app_ctx.ctx
        status_map = {
            True: "Success",
            False: "FAILED"
        }
        popup.title = "Composite Image Generation"
        popup.text = "Generating composite images..."
        popup.progress = 0
        popup.auto_dismiss = False

        composite_gen = CompositeGeneration(
            has_turret=ctx.lumaview.scope.has_turret(),
        )

        # For now, progress is only updated on the generation of each composite image, not each image that is used to generate the composite
        # May want to update this in the future
        ctx.file_io_executor.put(IOTask(action=composite_gen.load_folder,
                             args=(pathlib.Path(path),
                                    pathlib.Path(ctx.source_path) / "data" / "tiling.json",
                                    popup
                                    ),
                             callback=self.composite_gen_callback,
                             cb_args=(popup, status_map),
                             pass_result=True))

        # result = composite_gen.load_folder(
        #     path=pathlib.Path(path),
        #     tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json"
        # )
        # final_text = f"Generating composite images - {status_map[result['status']]}"
        # if result['status'] is False:
        #     final_text += f"\n{result['message']}"
        #     popup.text = final_text
        #     time.sleep(5)
        #     self.done = True
        #     return

        # popup.text = final_text
        # time.sleep(2)
        # self.done = True

    def composite_gen_callback(self, popup, status_map, result=None, exception=None):
        if result is None:
            popup.text = "Generating composite images - FAILED"
            Clock.schedule_once(lambda dt: popup.dismiss(), 5)
            return

        final_text = f"Generating composite images - {status_map[result['status']]}"
        if result['status'] is False:
            final_text += f"\n{result['message']}"
            popup.text = final_text
            Clock.schedule_once(lambda dt: popup.dismiss(), 5)
            return

        popup.text = final_text
        Clock.schedule_once(lambda dt: popup.dismiss(), 2)
        return


class VideoCreationControls(BoxLayout):

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import lumaviewpro
        lumaviewpro.video_creation_controls = self


    @show_popup
    def run_video_gen(self, popup, path) -> None:
        ctx = _app_ctx.ctx
        status_map = {
            True: "Success",
            False: "FAILED"
        }

        popup.title = "Video Builder"
        popup.text = "Generating video(s)..."
        popup.progress = 0
        popup.auto_dismiss = False

        try:
            fps = int(self.ids['video_gen_fps_id'].text)
        except Exception:
            fps = 5
            logger.error(f"Could not retrieve valid FPS for video generation. Using {fps} fps.")

        ts_overlay_btn = self.ids['enable_timestamp_overlay_btn']
        enable_timestamp_overlay = True if ts_overlay_btn.state == 'down' else False

        if fps < 1:
            msg = "Video generation frames/second must be >= 1 fps"
            final_text = f"Generating video(s) - {status_map[False]}"
            final_text += f"\n{msg}"
            popup.text = final_text
            logger.error(f"{msg}")
            Clock.schedule_once(lambda dt: popup.dismiss(), 5)
            return
            #self.done = True

        video_builder = VideoBuilder(
            has_turret=ctx.lumaview.scope.has_turret(),
        )

        ctx.file_io_executor.put(IOTask(action=video_builder.load_folder,
                             args=(pathlib.Path(path),
                                    pathlib.Path(ctx.source_path) / "data" / "tiling.json",
                                    popup
                                    ),
                             kwargs={
                                "frames_per_sec": fps,
                                "enable_timestamp_overlay": enable_timestamp_overlay,
                             },
                             callback=self.video_builder_callback,
                             cb_args=(popup, status_map),
                             pass_result=True))

        # result = video_builder.load_folder(
        #     path=pathlib.Path(path),
        #     tiling_configs_file_loc=pathlib.Path(source_path) / "data" / "tiling.json",
        #     frames_per_sec=fps,
        #     enable_timestamp_overlay=enable_timestamp_overlay,
        #     popup=popup
        # )

    def video_builder_callback(self, popup, status_map, result=None, exception=None):
        if result is None:
            popup.text = "Generating video(s) - FAILED"
            Clock.schedule_once(lambda dt: popup.dismiss(), 5)
            return

        final_text = f"Generating video(s) - {status_map[result['status']]}"
        if result['status'] is False:
            final_text += f"\n{result['message']}"
            popup.text = final_text
            Clock.schedule_once(lambda dt: popup.dismiss(), 5)
            return

        final_text = f"Generating video(s) - {status_map[result['status']]}"
        popup.text = final_text
        Clock.schedule_once(lambda dt: popup.dismiss(), 2)
        return
        # self._launch_video()


    # def _launch_video(self) -> None:
    #     try:
    #         os.startfile(self._output_file_loc)
    #     except Exception as e:
    #         logger.error(f"Unable to launch video {self._output_file_loc}:\n{e}")

# ============================================================================
# GraphingControls — Data Plotting and Trendlines
# ============================================================================

class GraphingControls(BoxLayout):
    x_axis_label = "X-Axis"
    y_axis_label = "Y-Axis"
    graph_title = ""
    available_axes = ['No Data Loaded']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('LVP Main: GraphingControls.__init__()')
        self._source_csv = None
        self.fig = None
        self._post = post_processing.PostProcessing()
        self.graphing_area = self.ids.graphing_area
        self.graph_widget = None
        self.x_axis_data = []
        self.y_axis_data = []
        self.selected_x_axis = None
        self.selected_y_axis = None
        self.trendline_enabled = False
        self.graph_df = None
        self.initialize_graph()

    def set_x_axis(self):
        if self._source_csv:
            self.selected_x_axis = self.ids['graphing_x_axis_spinner'].text
            self.ids.x_axis_label_input.text = self.selected_x_axis

            sorted_graph_df = self.graph_df.sort_values(by=self.selected_x_axis)
            self.x_axis_data = sorted_graph_df[self.selected_x_axis]
            self.update_x_axis_label()
            if self.selected_y_axis is None:
                return

            self.initialize_graph()
            self.update_x_axis_label()
            if "TIME" in self.selected_x_axis.upper():
                self.ax.xaxis.set_major_formatter(ConciseDateFormatter(self.ax.xaxis.get_major_locator()))
                self.ids.trendline_spinner.values = ('None', 'Linear', 'Quadratic', 'Exponential')
            elif "TIME" not in self.selected_y_axis.upper():
                self.ids.trendline_spinner.values = ('None', 'Linear', 'Quadratic', 'Exponential', 'Power', 'Logarithmic')
            self.ax.scatter(self.x_axis_data, self.y_axis_data)
            if self.trendline_enabled:
                self.update_trendline(axis=True)
            self.update_graph()

    def set_y_axis(self):
        if self._source_csv:
            self.selected_y_axis = self.ids['graphing_y_axis_spinner'].text
            self.ids.y_axis_label_input.text = self.selected_y_axis

            if self.selected_x_axis is None:
                self.y_axis_data = self.graph_df[self.selected_y_axis]
                self.update_y_axis_label()
                return

            sorted_graph_df = self.graph_df.sort_values(by=self.selected_x_axis)
            self.y_axis_data = sorted_graph_df[self.selected_y_axis]
            self.update_y_axis_label()

            self.initialize_graph()
            self.update_y_axis_label()
            if "TIME" in self.selected_y_axis.upper():
                self.ax.yaxis.set_major_formatter(ConciseDateFormatter(self.ax.yaxis.get_major_locator()))
                self.ids.trendline_spinner.values = ('None', 'Linear', 'Quadratic', 'Exponential')
            elif "TIME" not in self.selected_x_axis.upper():
                self.ids.trendline_spinner.values = ('None', 'Linear', 'Quadratic', 'Exponential', 'Power', 'Logarithmic')
            self.ax.scatter(self.x_axis_data, self.y_axis_data)
            if self.trendline_enabled:
                self.update_trendline(axis=True)
            self.update_graph()

    def update_x_axis_label(self):
        self.ax.set_xlabel(self.ids.x_axis_label_input.text)
        self.x_axis_label = self.ids.x_axis_label_input.text
        self.update_graph()

    def update_y_axis_label(self):
        self.ax.set_ylabel(self.ids.y_axis_label_input.text)
        self.y_axis_label = self.ids.y_axis_label_input.text
        self.update_graph()

    def update_available_axes(self):
        self.available_x_axes = list(self.available_axes)
        self.available_y_axes = list(self.available_axes)

        # Remove time from y-axis because it cannot be properly formatted at the moment and causes trendline issues
        if 'time' in self.available_y_axes:
            self.available_y_axes.remove('time')

        self.ids.graphing_x_axis_spinner.values = self.available_x_axes
        self.ids.graphing_y_axis_spinner.values = self.available_y_axes


    def update_graph_title(self):
        self.ax.set_title(self.ids.graph_title_input.text)
        self.graph_title = self.ids.graph_title_input.text
        self.update_graph()

    def update_trendline(self, axis: bool=False):
        if self.selected_x_axis is None or self.selected_y_axis is None:
            return

        trendline_type = self.ids.trendline_spinner.text
        if trendline_type == "None":
            self.trendline_enabled = False


        if not axis:
            self.initialize_graph()
            self.set_x_axis()
            self.set_y_axis()

        self.trendline_enabled = True

        #self.y_axis_data = self.graph_df[self.selected_y_axis]

        x_data = self.x_axis_data
        y_data = self.y_axis_data

        time_x = False
        time_y = False

        # If we are dealing with time, convert to an ordinal fomat for trendline creation
        if 'time' in self.selected_x_axis:
            x_time_data_original = x_data
            x_ref_time = x_data.min()

            # Normalize x-data for scaling purposes
            x_data = (x_data - x_ref_time).dt.total_seconds()
            x_data = x_data.to_numpy()
            time_x = True
        else:
            x_data = x_data.to_numpy()

        if 'time' in self.selected_y_axis:
            y_time_data_original = y_data
            y_ref_time = y_data.min()

            # Normalize y-data for scaling purposes
            y_data = (y_data - y_ref_time).dt.total_seconds()
            y_data = y_data.to_numpy()
            time_y = True
        else:
            y_data = y_data.to_numpy()


        if len(x_data) > 1 and len(y_data) > 1:

            if trendline_type == "Linear":
                try:
                    z = np.polyfit(x_data, y_data, 1)  # 1st degree polynomial (linear fit)
                    p = np.poly1d(z)

                    if time_x:
                        self.ax.plot(x_time_data_original, p(x_data), "r--")
                    else:
                        self.ax.plot(x_data, p(x_data), "r--")
                except Exception as e:
                    logger.exception(f"[Graphing  ] Could not fit linear trendline: {e}")
                    self.ids.trendline_spinner.text = "None"


            elif trendline_type == "Quadratic":
                try:
                    z = np.polyfit(x_data, y_data, 2)
                    p = np.poly1d(z)

                    if time_x:
                        self.ax.plot(x_time_data_original, p(x_data), "r--")
                    else:
                        self.ax.plot(x_data, p(x_data), "r--")
                except Exception as e:
                    logger.exception(f"[Graphing  ] Could not fit quadratic trendline: {e}")
                    self.ids.trendline_spinner.text = "None"

            elif trendline_type == "Exponential":
                try:
                    log_y_data = np.log(y_data)

                    # Calculate the exponential trendline
                    z = np.polyfit(x_data, log_y_data, 1)
                    p = np.poly1d(z)

                    # Convert back to original scale
                    exp_y_data = np.exp(p(x_data))

                    if time_x:
                        self.ax.plot(x_time_data_original, exp_y_data, "r--")
                    else:
                        self.ax.plot(x_data, exp_y_data, "r--")
                except Exception as e:
                    logger.exception(f"[Graphing  ] Could not fit exponential trendline: {e}")
                    self.ids.trendline_spinner.text = "None"

            elif trendline_type == "Power":
                try:
                    # Transform data for power fit
                    log_x_data = np.log(x_data)
                    log_y_data = np.log(y_data)

                    # Calculate the power trendline
                    z = np.polyfit(log_x_data, log_y_data, 1)
                    p = np.poly1d(z)

                    # Convert back to original scale
                    power_y_data = np.exp(p(np.log(x_data)))

                    try:
                        self.ax.plot(x_data, power_y_data, "r--")
                    except Exception as e:
                        logger.exception(f"Graphing ] Power trendline error: {e}")
                except Exception as e:
                    logger.exception(f"[Graphing  ] Could not fit power trendline: {e}")
                    self.ids.trendline_spinner.text = "None"

            elif trendline_type == "Logarithmic":
                try:
                    # Transform x_data for logarithmic fit
                    log_x_data = np.log(x_data)

                    # Calculate the logarithmic trendline
                    z = np.polyfit(log_x_data, y_data, 1)
                    p = np.poly1d(z)

                    try:
                        self.ax.plot(x_data, p(np.log(x_data)), "r--")
                    except Exception as e:
                        logger.exception(f"Graphing ] Logarithmic trendline error: {e}")
                except Exception as e:
                    logger.exception(f"[Graphing  ] Could not fit logarithmic trendline: {e}")
                    self.ids.trendline_spinner.text = "None"

            self.update_graph()


    def regenerate_graph(self):
        self.initialize_graph()
        self.set_x_axis()
        self.set_y_axis()
        if self.trendline_enabled:
            self.update_trendline()

    def initialize_graph(self):
        if plt:
            plt.clf()
        graphing_area = self.graphing_area
        self.fig, self.ax = plt.subplots()
        self.ax.scatter([], [])
        self.ax.set_xlabel(self.x_axis_label)
        self.ax.set_ylabel(self.y_axis_label)
        self.ax.set_title(self.graph_title)

        if self.graph_widget:
            graphing_area.remove_widget(self.graph_widget)

        from ui.figure_canvas import FigureCanvasKivyAgg
        self.graph_widget = FigureCanvasKivyAgg(plt.gcf())

        graphing_area.add_widget(self.graph_widget)


    def update_graph(self):
        self.graph_widget.draw()

    def save_graph(self, filepath):
        plt.savefig(filepath)

    def set_graphing_source(self, file):
        from datetime import datetime as date_time
        self._source_csv = file
        self.initialize_graph()
        try:
            self.graph_df = pd.read_csv(file)
            self.available_axes = list(self.graph_df.keys())
            if self.available_axes[0] == "file":
                self.available_axes = self.available_axes[1:]
            if "time" in self.available_axes:
                self.graph_df['time'] = [date_time.strptime(datetime_obj, '%c') for datetime_obj in self.graph_df['time']]

            self.update_available_axes()
            self.set_x_axis()
            self.set_y_axis()

        except Exception as e:
            logger.exception(f"Graph Generation | Set graphing source | {e}")



    def set_post_processing_module(self, postprocessingmodule):
        self._post = postprocessingmodule


# ============================================================================
# CellCountControls — Cell Counting and Analysis
# ============================================================================

class CellCountControls(BoxLayout):

    ENABLE_PREVIEW_AUTO_REFRESH = False

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('LVP Main: CellCountControls.__init__()')
        self._preview_source_image = None
        self._preview_image = None
        self._post = post_processing.PostProcessing()
        self._settings = self._get_init_settings()
        self._set_ui_to_settings(self._settings)


    def _get_init_settings(self):
        return {
            'context': {
                'pixels_per_um': 1.0,       # default; updated per objective/camera at runtime
                'fluorescent_mode': True
            },
            'segmentation': {
                'algorithm': 'initial',
                'parameters': {
                    'threshold': 20,
                }
            },
            'filters': {
                'area': {
                    'min': 0,
                    'max': 100
                },
                'perimeter': {
                    'min': 0,
                    'max': 100
                },
                'sphericity': {
                    'min': 0.0,
                    'max': 1.0
                },
                'intensity': {
                    'min': {
                        'min': 0,
                        'max': 100
                    },
                    'mean': {
                        'min': 0,
                        'max': 100
                    },
                    'max': {
                        'min': 0,
                        'max': 100
                    }
                }
            }
        }

    def apply_method_to_preview_image(self):
        self._regenerate_image_preview()


    # Decorate function to show popup and run the code below in a thread
    @show_popup
    def apply_method_to_folder(self, popup, path):
        popup.title = 'Processing Cell Count Method'
        pre_text = f'Applying method to folder: {path}'
        popup.text = pre_text

        popup.progress = 0
        popup.auto_dismiss = False

        _app_ctx.ctx.file_io_executor.put(IOTask(action=self.execute_apply_method_to_folder,
                             args=(popup, path),
                             callback=self.apply_method_to_folder_callback,
                             cb_args=(popup, path),
                             pass_result=True))

    def execute_apply_method_to_folder(self, popup, path):
        pre_text = f'Applying method to folder: {path}'
        total_images = self._post.get_num_images_in_folder(path=path)
        image_count = 0

        for image_process in self._post.apply_cell_count_to_folder(path=path, settings=self._settings):
            filename = image_process['filename']
            image_count += 1
            popup.progress = int(100 * image_count / total_images)
            popup.text = f"{pre_text}\n- {image_count}/{total_images}: {filename}"

    def apply_method_to_folder_callback(self, popup, path, result=None, exception=None):
        if result is None:
            popup.text = "Applying method to folder - FAILED"
            Clock.schedule_once(lambda dt: popup.dismiss(), 5)
            return

        popup.progress = 100
        popup.text = "Applying method to folder - Done"
        Clock.schedule_once(lambda dt: popup.dismiss(), 2)
        return

    def set_post_processing_module(self, post_processing_module):
        self._post = post_processing_module

    def get_current_settings(self):
        return self._settings


    @staticmethod
    def _validate_method_settings_metadata(settings):
        if 'metadata' not in settings:
            raise Exception(f"No valid metadata found")

        metadata = settings['metadata']

        for key in ('type', 'version'):
            if key not in metadata:
                raise Exception(f"No {key} found in metadata")


    def _add_method_settings_metadata(self):
        self._settings['metadata'] = {
            'type': 'cell_count_method',
            'version': '1'
        }


    def load_settings(self, settings):
        self._validate_method_settings_metadata(settings=settings)
        self._settings = settings
        self._set_ui_to_settings(settings)


    def _area_range_slider_values_to_physical(self, slider_values):
        if self._preview_source_image is None:
            return slider_values

        xp = [0, 30, 60, 100]
        max = self.calculate_area_filter_max(image=self._preview_source_image)
        if max < 10001:
            max = 10001

        fp = [0, 1000, 10000, max]
        fg = np.interp(slider_values, xp, fp)
        return fg[0], fg[1]

    def _area_range_slider_physical_to_values(self, physical_values):
        if self._preview_source_image is None:
            return physical_values

        max = self.calculate_area_filter_max(image=self._preview_source_image)
        if max < 10001:
            max = 10001

        xp = [0, 1000, 10000, max]
        fp = [0, 30, 60, 100]
        fg = np.interp(physical_values, xp, fp)
        return fg[0], fg[1]

    def _perimeter_range_slider_values_to_physical(self, slider_values):
        if self._preview_source_image is None:
            return slider_values

        xp = [0, 50, 100]

        max = self.calculate_perimeter_filter_max(image=self._preview_source_image)
        if max < 101:
            max = 101

        fp = [0, 100, max]
        fg = np.interp(slider_values, xp, fp)
        return fg[0], fg[1]

    def _perimeter_range_slider_physical_to_values(self, physical_values):
        if self._preview_source_image is None:
            return physical_values

        max = self.calculate_perimeter_filter_max(image=self._preview_source_image)
        if max < 101:
            max = 101

        xp = [0, 100, max]
        fp = [0, 50, 100]
        fg = np.interp(physical_values, xp, fp)
        return fg[0], fg[1]


    def _set_ui_to_settings(self, settings):
        self.ids.text_cell_count_pixels_per_um_id.text = str(settings['context']['pixels_per_um'])
        self.ids.cell_count_fluorescent_mode_id.active = settings['context']['fluorescent_mode']
        self.ids.slider_cell_count_threshold_id.value = settings['segmentation']['parameters']['threshold']
        self.ids.slider_cell_count_area_id.value = self._area_range_slider_physical_to_values(
            (settings['filters']['area']['min'], settings['filters']['area']['max'])
        )

        self.ids.slider_cell_count_perimeter_id.value = self._perimeter_range_slider_physical_to_values(
            (settings['filters']['perimeter']['min'], settings['filters']['perimeter']['max'])
        )
        self.ids.slider_cell_count_sphericity_id.value = (settings['filters']['sphericity']['min'], settings['filters']['sphericity']['max'])
        self.ids.slider_cell_count_min_intensity_id.value = (settings['filters']['intensity']['min']['min'], settings['filters']['intensity']['min']['max'])
        self.ids.slider_cell_count_mean_intensity_id.value = (settings['filters']['intensity']['mean']['min'], settings['filters']['intensity']['mean']['max'])
        self.ids.slider_cell_count_max_intensity_id.value = (settings['filters']['intensity']['max']['min'], settings['filters']['intensity']['max']['max'])

        self.slider_adjustment_area()
        self.slider_adjustment_perimeter()
        self._regenerate_image_preview()


    def set_preview_source_file(self, file) -> None:
        image = image_utils.image_file_to_image(image_file=file)
        if image is None:
            return

        self.set_preview_source(image=image)


    def calculate_area_filter_max(self, image):
        pixels_per_um = self._settings['context']['pixels_per_um']

        max_area_pixels = image.shape[0] * image.shape[1]
        max_area_um2 = max_area_pixels / (pixels_per_um**2)
        return max_area_um2


    def calculate_perimeter_filter_max(self, image):
        pixels_per_um = self._settings['context']['pixels_per_um']

        # Assume max perimeter will never need to be larger than 2x frame size border
        # The 2x is to provide margin for handling various curvatures
        max_perimeter_pixels = 2*((2*image.shape[0])+(2*image.shape[1]))
        max_perimeter_um = max_perimeter_pixels / pixels_per_um
        return max_perimeter_um


    def update_filter_max(self, image):
        max_area_um2 = self. calculate_area_filter_max(image=image)
        max_perimeter_um = self.calculate_perimeter_filter_max(image=image)

        self.ids.slider_cell_count_area_id.max = int(self._area_range_slider_physical_to_values(physical_values=(0,max_area_um2))[1])
        self.ids.slider_cell_count_perimeter_id.max = int(self._perimeter_range_slider_physical_to_values(physical_values=(0,max_perimeter_um))[1])

        self.slider_adjustment_area()
        self.slider_adjustment_perimeter()


    def set_preview_source(self, image) -> None:
        self._preview_source_image = image
        self._preview_image = image
        self.ids['cell_count_image_id'].texture = image_utils_kivy.image_to_texture(image=image)
        self.update_filter_max(image=image)
        self._regenerate_image_preview()


    # Save settings to JSON file
    def save_method_as(self, file="./data/cell_count_method.json"):
        logger.info(f'[LVP Main  ] CellCountContent.save_method_as({file})')
        # Resolve relative paths against source_path instead of relying on CWD
        if not os.path.isabs(file):
            file = os.path.join(_app_ctx.ctx.source_path, file)
        self._add_method_settings_metadata()
        with open(file, "w") as write_file:
            json.dump(self._settings, write_file, indent = 4, cls=CustomJSONizer)


    def load_method_from_file(self, file):
        logger.info(f'[LVP Main  ] CellCountContent.load_method_from_file({file})')
        with open(file, "r") as f:
            method_settings = json.load(f)

        self.load_settings(settings=method_settings)


    def _regenerate_image_preview(self):
        if self._preview_source_image is None:
            return

        image, _ = self._post.preview_cell_count(
            image=self._preview_source_image,
            settings=self._settings
        )

        self._preview_image = image

        _app_ctx.ctx.cell_count_content.ids['cell_count_image_id'].texture = image_utils_kivy.image_to_texture(image=image)


    def slider_adjustment_threshold(self):
        self._settings['segmentation']['parameters']['threshold'] = self.ids['slider_cell_count_threshold_id'].value

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def slider_adjustment_area(self):
        low, high = self._area_range_slider_values_to_physical(
            (self.ids['slider_cell_count_area_id'].value[0], self.ids['slider_cell_count_area_id'].value[1])
        )

        self._settings['filters']['area']['min'], self._settings['filters']['area']['max'] = low, high

        self.ids['label_cell_count_area_id'].text = f"{int(low)}-{int(high)} \u03bcm\u00b2"

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def slider_adjustment_perimeter(self):
        low, high = self._perimeter_range_slider_values_to_physical(
            (self.ids['slider_cell_count_perimeter_id'].value[0], self.ids['slider_cell_count_perimeter_id'].value[1])
        )

        self._settings['filters']['perimeter']['min'], self._settings['filters']['perimeter']['max'] = low, high

        self.ids['label_cell_count_perimeter_id'].text = f"{int(low)}-{int(high)} \u03bcm"

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_sphericity(self):
        self._settings['filters']['sphericity']['min'] = self.ids['slider_cell_count_sphericity_id'].value[0]
        self._settings['filters']['sphericity']['max'] = self.ids['slider_cell_count_sphericity_id'].value[1]

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_min_intensity(self):
        self._settings['filters']['intensity']['min']['min'] = self.ids['slider_cell_count_min_intensity_id'].value[0]
        self._settings['filters']['intensity']['min']['max'] = self.ids['slider_cell_count_min_intensity_id'].value[1]

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def slider_adjustment_mean_intensity(self):
        self._settings['filters']['intensity']['mean']['min'] = self.ids['slider_cell_count_mean_intensity_id'].value[0]
        self._settings['filters']['intensity']['mean']['max'] = self.ids['slider_cell_count_mean_intensity_id'].value[1]

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def slider_adjustment_max_intensity(self):
        self._settings['filters']['intensity']['max']['min'] = self.ids['slider_cell_count_max_intensity_id'].value[0]
        self._settings['filters']['intensity']['max']['max'] = self.ids['slider_cell_count_max_intensity_id'].value[1]

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def flourescent_mode_toggle(self):
        self._settings['context']['fluorescent_mode'] = self.ids['cell_count_fluorescent_mode_id'].active

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()


    def pixel_conversion_adjustment(self):

        def _validate(value_str):
            try:
                value = float(value_str)
            except Exception:
                return False, -1

            if value <= 0:
                return False, -1

            return True, value

        value_str = _app_ctx.ctx.cell_count_content.ids['text_cell_count_pixels_per_um_id'].text

        valid, value = _validate(value_str)
        if not valid:
            return

        if self._preview_image is None:
            return

        self._settings['context']['pixels_per_um'] = value
        self.update_filter_max(image=self._preview_image)


# ============================================================================
# PostProcessingAccordion — Post-Processing Panel Container
# ============================================================================

class PostProcessingAccordion(BoxLayout):

    def __init__(self, **kwargs):
        #super(PostProcessingAccordion,self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.name = self.__class__.__name__
        self.post = post_processing.PostProcessing()
        #stitching params (see more info in image_stitcher.py):
        #self.raw_images_folder = settings['save_folder'] # I'm guessing not ./capture/ because that would have frames over time already (to make video)
        self.raw_images_folder = './capture/' # I'm guessing not ./capture/ because that would have frames over time already (to make video)
        self.combine_colors = False #True if raw images are in separate red/green/blue channels and need to be first combined
        self.ext = "tiff" #or read it from settings?
        #self.stitching_method = "features" # "features" - Low method, "position" - based on position information
        self.stitching_method = "position" # "features" - Low method, "position" - based on position information
        self.stitched_save_name = "last_composite_img.tiff"
        #self.positions_file = None #relevant if stitching method is position, will read positions from that file
        self.positions_file = "./capture/2x2.tsv" #relevant if stitching method is position, will read positions from that file
        self.pos2pix = 2630 # relevant if stitching method is position. The scale conversion for pos info into pixels


        # self.tiling_target = []
        self.tiling_min = {
            "x": 120000,
            "y": 80000
        }

        self.tiling_max = {
            "x": 0,
            "y": 0
        }

        self.tiling_count = {
            "x": 1,
            "y": 1
        }

        self.accordion_item_states = {
            'cell_count_accordion_id': None,
            'stitch_accordion_id': None,
            'composite_gen_accordion_id': None,
            'zprojection_accordion_id': None,
            'create_avi_accordion_id': None
        }
        """print("===============================================================")
        print(self.ids)
        print("===============================================================")
        for id in self.accordion_item_states.keys():
            self.ids[id].background_color = [0.753, 0.816, 0.910, 1]"""

        self.init_cell_count()
        self._graphing_popup = None


    @staticmethod
    def accordion_item_state(accordion_item):
        if accordion_item.collapse:
            return 'closed'
        return 'open'

    def hide_stitch(self):
        #self.ids['stitch_accordion_id'].visible = False
        # sc = self.ids['stitch_controls_id']
        # sa = self.ids['stitch_accordion_id']
        # self.ids['stitch_controls_id'].visible = False
        # self.remove_widget(self.ids['stitch_accordion_id'])
        # self.remove_widget(self.ids['stitch_controls_id'])
        #self.ids['post_processing_accordion_id'].remove_widget(stitch_controls)
        stitch_accordion = None

        post_accordion = self.children[0]
        for child in post_accordion.children:
            if child.title == 'Stitch':
                stitch_accordion = child
                break

        if stitch_accordion:
            stitch_accordion.parent.remove_widget(stitch_accordion)


    def get_accordion_item_states(self):
        return {
            'cell_count_accordion_id': self.accordion_item_state(self.ids['cell_count_accordion_id']),
            'stitch_accordion_id': self.accordion_item_state(self.ids['stitch_accordion_id']),
            'composite_gen_accordion_id': self.accordion_item_state(self.ids['composite_gen_accordion_id']),
            'zprojection_accordion_id': self.accordion_item_state(self.ids['zprojection_accordion_id']),
            'create_avi_accordion_id': self.accordion_item_state(self.ids['create_avi_accordion_id']),
        }


    def accordion_collapse(self):

        new_accordion_item_states = self.get_accordion_item_states()

        changed_items = []
        for accordion_item_id, prev_accordion_item_state in self.accordion_item_states.items():
            if new_accordion_item_states[accordion_item_id] == prev_accordion_item_state:
                # No change
                continue

            # Update state and add state change to list
            self.accordion_item_states[accordion_item_id] = self.accordion_item_state(self.ids[accordion_item_id])
            changed_items.append(accordion_item_id)



    def init_cell_count(self):
        self._cell_count_popup = None


    def convert_to_avi(self):
        logger.debug('[LVP Main  ] PostProcessingAccordian.convert_to_avi() not yet implemented')


    def open_cell_count(self):
        ctx = _app_ctx.ctx
        if self._cell_count_popup is None:
            ctx.cell_count_content.set_post_processing_module(self.post)
            self._cell_count_popup = Popup(
                title="Post Processing - Object Analysis",
                content=ctx.cell_count_content,
                size_hint=(0.85,0.85),
                auto_dismiss=True
            )

        self._cell_count_popup.open()

    def open_graphing(self):
        ctx = _app_ctx.ctx
        if self._graphing_popup is None:
            ctx.graphing_controls.set_post_processing_module(self.post)
            self._graphing_popup = Popup(
                title="Post Processing - Object Plotting",
                content=ctx.graphing_controls,
                size_hint=(0.85,0.85),
                auto_dismiss=True
            )

        self._graphing_popup.open()


def open_last_save_folder():
    ctx = _app_ctx.ctx

    OS_FOLDER_MAP = {
        'win32': 'explorer',
        'darwin': 'open',
        'linux': 'xdg-open'
    }

    if sys.platform not in OS_FOLDER_MAP:
        logger.info(f'[LVP Main  ] PostProcessing.open_folder() not yet implemented for {sys.platform} platform')
        return

    command = OS_FOLDER_MAP[sys.platform]
    if ctx.last_save_folder is None:
        subprocess.Popen([command, str(pathlib.Path(ctx.settings['live_folder']).resolve())])
    else:
        subprocess.Popen([command, str(ctx.last_save_folder)])

# ============================================================================
# CellCountDisplay and ShaderEditor
# ============================================================================

class CellCountDisplay(FloatLayout):

    def __init__(self, **kwargs):
        super(CellCountDisplay,self).__init__(**kwargs)
