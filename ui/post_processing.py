# Copyright Etaluma, Inc.
import logging
import os
import pathlib
import subprocess
import sys
from typing import ClassVar

import matplotlib

matplotlib.use('Agg')  # Must be set before pyplot import to avoid Tk/macOS conflict
import matplotlib.pyplot as plt
from matplotlib.dates import ConciseDateFormatter
import numpy as np
import pandas as pd

from kivy.clock import Clock
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup

from ui.progress_popup import show_popup
from modules import gui_logger
from modules.stitcher import Stitcher
import modules.zprojector as zprojector
import modules.post_processing as post_processing
import modules.image_utils as image_utils
import ui.image_utils_kivy as image_utils_kivy
import modules.app_context as _app_ctx
from ui.ui_helpers import run_reported, submit_reported

logger = logging.getLogger('LVP.ui.post_processing')


def _run_build(build, popup, label: str, on_done=None) -> None:
    """Run a session post-processing build on its lane, showing its progress.

    *build* takes the progress callback and calls one
    ``session.post_processing`` member. The build runs on the
    post-processing lane through the GUI boundary, which reports a refusal
    or failure once, as the request of the person who pressed the button.
    The popup then shows the build's own words for what it made, or closes
    when the build raised: its outcome has already been told. *on_done*
    gets the result, or None, for a panel that shows more of it.
    """
    produced = {}

    def _progress(percent: float, text: str | None) -> None:
        popup.progress = percent
        if text is not None:
            popup.text = text

    def _build():
        produced['result'] = build(_progress)

    def _show():
        result = produced.get('result')
        if on_done is not None:
            on_done(result)
        if result is None:
            popup.dismiss()
            return
        popup.progress = 100
        popup.text = result['message']
        Clock.schedule_once(lambda dt: popup.dismiss(), 5 if result.get('degraded') else 2)

    submit_reported(_build, _show, label, lane=_app_ctx.ctx.session.post_processing.lane)


class QuickEnhanceControls(BoxLayout):
    """GUI shell for the non-destructive Quick Enhance derived-file path."""

    done = BooleanProperty(False)
    last_output_folder = StringProperty('')
    status_text = StringProperty('')
    busy = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _app_ctx.register_early('quick_enhance_controls', self)

    def set_source_file(self, file) -> None:
        self._start_export(pathlib.Path(file))

    def set_source_folder(self, path) -> None:
        self._start_export(pathlib.Path(path))

    def _start_export(self, target: pathlib.Path) -> None:
        self.busy = True
        self.status_text = ''
        self.export(target)

    @show_popup
    def export(self, popup, target: pathlib.Path) -> None:
        popup.title = 'Enhance'
        popup.text = ''
        popup.progress = 0
        popup.auto_dismiss = False

        def _build(progress):
            def _progress(percent: float, text: str | None) -> None:
                progress(percent, text)
                if text is not None:
                    Clock.schedule_once(lambda _dt: setattr(self, 'status_text', text), 0)

            return _app_ctx.ctx.session.post_processing.enhance(
                target, on_progress=_progress, on_derived_image=self._queue_derived_image
            )

        _run_build(_build, popup, 'ENHANCE', on_done=self._export_done)

    def _queue_derived_image(self, image: np.ndarray, significant_bits: int) -> None:
        display_image = image.copy()

        def _show(_dt):
            scope_display = getattr(_app_ctx.ctx, 'scope_display', None)
            if scope_display is not None:
                scope_display.hold_derived_image(display_image, significant_bits)

        Clock.schedule_once(_show, 0)

    def _export_done(self, result) -> None:
        self.busy = False
        if result is None:
            self.status_text = ''
            return
        self.last_output_folder = str(result['output_folder'])
        self.status_text = result['message']


class StitchControls(BoxLayout):
    done = BooleanProperty(False)
    stitching_mode = StringProperty('Quality')
    _MODE_VALUES: ClassVar[dict[str, str]] = {
        'Quality': Stitcher.QUALITY_MODE,
        'Fast Preview': Stitcher.FAST_PREVIEW_MODE,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _app_ctx.register_early('stitch_controls', self)

    def set_button_enabled_state(self, state: bool):
        disabled = not state
        self.ids['quality_stitch_btn'].disabled = disabled
        self.ids['fast_preview_stitch_btn'].disabled = disabled

    @show_popup
    def run_stitcher(self, popup, path):
        mode_label = self.stitching_mode
        stitching_mode = self._MODE_VALUES.get(mode_label, Stitcher.QUALITY_MODE)
        gui_logger.button('RUN_STITCHER', f'path={path} mode={stitching_mode}')
        ctx = _app_ctx.ctx
        popup.title = f'{mode_label} Stitch'
        popup.text = (
            f'Running {mode_label} Stitch.\n'
            'Estimating remaining time after the first tile group.\n'
            'Source pixels and channel colors are preserved.'
        )
        popup.progress = 0
        popup.auto_dismiss = False

        _run_build(
            lambda progress: ctx.session.post_processing.stitch(
                path, mode=stitching_mode, on_progress=progress
            ),
            popup,
            'RUN_STITCHER',
        )


class ZProjectionControls(BoxLayout):
    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _app_ctx.register_early('zprojection_controls', self)
        Clock.schedule_once(self._init_ui, 0)

    def _init_ui(self, dt=0):
        # Each assignment below dispatches the spinner's text-change event, so
        # populating this panel used to record two selections nobody made --
        # setting .values snaps the text to the first option, then .text moves
        # it to the default. Declared before each write, because the dispatch is
        # synchronous and the record would otherwise already be out.
        methods = zprojector.ZProjector.methods()
        gui_logger.note_write_back('ZPROJECTION_METHOD', methods[0])
        self.ids['zprojection_method_spinner'].values = methods
        gui_logger.note_write_back('ZPROJECTION_METHOD', methods[1])
        self.ids['zprojection_method_spinner'].text = methods[1]

    @show_popup
    def run_zprojection(self, popup, path):
        gui_logger.button('RUN_ZPROJECTION', f'path={path}')
        ctx = _app_ctx.ctx
        popup.title = 'Z-Projection'
        popup.progress = 0
        popup.auto_dismiss = False

        popup.text = 'Generating Z-Projection images...'

        method = self.ids['zprojection_method_spinner'].text
        _run_build(
            lambda progress: ctx.session.post_processing.zproject(
                path, method=method, on_progress=progress
            ),
            popup,
            'RUN_ZPROJECTION',
        )

    def log_zprojection_method(self) -> None:
        """Record a z-projection method selection.

        Bound on the spinner's text change, which is what every other logged
        spinner in this app uses. That event cannot see a re-selection of the
        value already shown, and it does fire on the programmatic write this
        panel makes while setting its options up, so a selection record
        appears at app start. Both are properties of the event rather than of
        this call, and both are fixed for every spinner at once when spinner
        selection moves to the dropdown's own event.
        """
        gui_logger.select('ZPROJECTION_METHOD', self.ids['zprojection_method_spinner'].text)


class CompositeGenControls(BoxLayout):
    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _app_ctx.register_early('composite_gen_controls', self)

    @show_popup
    def run_composite_gen(self, popup, path):
        gui_logger.button('RUN_COMPOSITE_GEN', f'path={path}')
        ctx = _app_ctx.ctx
        popup.title = 'Composite Image Generation'
        popup.text = 'Generating composite images...'
        popup.progress = 0
        popup.auto_dismiss = False

        _run_build(
            lambda progress: ctx.session.post_processing.composite(path, on_progress=progress),
            popup,
            'RUN_COMPOSITE_GEN',
        )


class VideoCreationControls(BoxLayout):
    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _app_ctx.register_early('video_creation_controls', self)

    @show_popup
    def run_video_gen(self, popup, path) -> None:
        gui_logger.button('RUN_VIDEO_GEN', f'path={path}')
        ctx = _app_ctx.ctx

        popup.title = 'Video Builder'
        popup.text = 'Generating video(s)...'
        popup.progress = 0
        popup.auto_dismiss = False

        # Blank (or 'auto') = the recording's own measured rate; anything
        # else is the user's playback-rate override, which the build judges.
        fps_text = self.ids['video_gen_fps_id'].text.strip()
        fps = None if fps_text.lower() in ('', 'auto') else fps_text
        enable_timestamp_overlay = self.ids['enable_timestamp_overlay_btn'].state == 'down'
        _run_build(
            lambda progress: ctx.session.post_processing.video(
                path,
                frames_per_sec=fps,
                timestamp_overlay=enable_timestamp_overlay,
                on_progress=progress,
            ),
            popup,
            'RUN_VIDEO_GEN',
        )

    def log_video_gen_fps(self) -> None:
        """Record a typed playback-rate commit.

        An empty box is meaningful here rather than absent: it means use the
        recording's own measured rate, which is what the field's hint says.
        The raw text is recorded so that choice is visible as the user left it.
        """
        gui_logger.text_input('VIDEO_GEN_FPS', self.ids['video_gen_fps_id'].text)

    def log_timestamp_overlay(self) -> None:
        """Record the timestamp-overlay toggle.

        This is a ToggleButton, whose state is the string 'normal' or 'down';
        both are truthy, so the comparison -- not the raw state -- is what
        carries the user's intent. The same conversion is how this panel reads
        the control when it builds a video.
        """
        state_down = self.ids['enable_timestamp_overlay_btn'].state == 'down'
        gui_logger.toggle('VIDEO_TIMESTAMP_OVERLAY_BTN', state_down)


# ============================================================================
# GraphingControls -- Data Plotting and Trendlines
# ============================================================================


class GraphingControls(BoxLayout):
    x_axis_label = 'X-Axis'
    y_axis_label = 'Y-Axis'
    graph_title = ''
    available_axes: ClassVar[list] = ['No Data Loaded']

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
        axis = self.ids['graphing_x_axis_spinner'].text
        gui_logger.select('GRAPHING_X_AXIS', axis)
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
            if 'TIME' in self.selected_x_axis.upper():
                self.ax.xaxis.set_major_formatter(
                    ConciseDateFormatter(self.ax.xaxis.get_major_locator())
                )
                self.ids.trendline_spinner.values = ('None', 'Linear', 'Quadratic', 'Exponential')
            elif 'TIME' not in self.selected_y_axis.upper():
                self.ids.trendline_spinner.values = (
                    'None',
                    'Linear',
                    'Quadratic',
                    'Exponential',
                    'Power',
                    'Logarithmic',
                )
            self.ax.scatter(self.x_axis_data, self.y_axis_data)
            if self.trendline_enabled:
                self.update_trendline(axis=True)
            self.update_graph()

    def set_y_axis(self):
        gui_logger.select('GRAPHING_Y_AXIS', self.ids['graphing_y_axis_spinner'].text)
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
            if 'TIME' in self.selected_y_axis.upper():
                self.ax.yaxis.set_major_formatter(
                    ConciseDateFormatter(self.ax.yaxis.get_major_locator())
                )
                self.ids.trendline_spinner.values = ('None', 'Linear', 'Quadratic', 'Exponential')
            elif 'TIME' not in self.selected_x_axis.upper():
                self.ids.trendline_spinner.values = (
                    'None',
                    'Linear',
                    'Quadratic',
                    'Exponential',
                    'Power',
                    'Logarithmic',
                )
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

    def log_text_commit(self, name: str, widget_id: str) -> None:
        """Record a typed commit in one of the graph's three label fields.

        Bound on the commit events rather than inside the live-update handlers
        those fields already carry: those run on every keystroke, and two of
        the three are also written programmatically when an axis is chosen, so
        logging from them would produce a line per character plus lines the
        user never typed.

        The name and widget id are passed in because all three fields share
        this method. A no-argument version could not say which field fired it,
        so all three would report under one name and a bundle could not tell
        which label the user edited.
        """
        gui_logger.text_input(name, self.ids[widget_id].text)

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

    def update_trendline(self, axis: bool = False):
        # Logged at entry, before the early return: choosing a trendline before
        # the axes are set is still a user action and would otherwise vanish.
        # `axis` is the discriminator -- the kv spinner calls this with no
        # argument, while set_x_axis/set_y_axis pass True, so a record is only
        # emitted for a real selection and not for an axis-driven refresh.
        if not axis:
            gui_logger.select('TRENDLINE', self.ids.trendline_spinner.text)

        if self.selected_x_axis is None or self.selected_y_axis is None:
            return

        trendline_type = self.ids.trendline_spinner.text
        if trendline_type == 'None':
            self.trendline_enabled = False

        if not axis:
            self.initialize_graph()
            self.set_x_axis()
            self.set_y_axis()

        self.trendline_enabled = True

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
            y_time_data_original = y_data  # noqa: F841 -- deferred
            y_ref_time = y_data.min()

            # Normalize y-data for scaling purposes
            y_data = (y_data - y_ref_time).dt.total_seconds()
            y_data = y_data.to_numpy()
            time_y = True  # noqa: F841 -- deferred
        else:
            y_data = y_data.to_numpy()

        if len(x_data) > 1 and len(y_data) > 1:
            if trendline_type == 'Linear':
                try:
                    z = np.polyfit(x_data, y_data, 1)  # 1st degree polynomial (linear fit)
                    p = np.poly1d(z)

                    if time_x:
                        self.ax.plot(x_time_data_original, p(x_data), 'r--')
                    else:
                        self.ax.plot(x_data, p(x_data), 'r--')
                except Exception as e:
                    logger.exception(f'[Graphing  ] Could not fit linear trendline: {e}')
                    self.ids.trendline_spinner.text = 'None'

            elif trendline_type == 'Quadratic':
                try:
                    z = np.polyfit(x_data, y_data, 2)
                    p = np.poly1d(z)

                    if time_x:
                        self.ax.plot(x_time_data_original, p(x_data), 'r--')
                    else:
                        self.ax.plot(x_data, p(x_data), 'r--')
                except Exception as e:
                    logger.exception(f'[Graphing  ] Could not fit quadratic trendline: {e}')
                    self.ids.trendline_spinner.text = 'None'

            elif trendline_type == 'Exponential':
                try:
                    log_y_data = np.log(y_data)

                    # Calculate the exponential trendline
                    z = np.polyfit(x_data, log_y_data, 1)
                    p = np.poly1d(z)

                    # Convert back to original scale
                    exp_y_data = np.exp(p(x_data))

                    if time_x:
                        self.ax.plot(x_time_data_original, exp_y_data, 'r--')
                    else:
                        self.ax.plot(x_data, exp_y_data, 'r--')
                except Exception as e:
                    logger.exception(f'[Graphing  ] Could not fit exponential trendline: {e}')
                    self.ids.trendline_spinner.text = 'None'

            elif trendline_type == 'Power':
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
                        self.ax.plot(x_data, power_y_data, 'r--')
                    except Exception as e:
                        logger.exception(f'Graphing ] Power trendline error: {e}')
                except Exception as e:
                    logger.exception(f'[Graphing  ] Could not fit power trendline: {e}')
                    self.ids.trendline_spinner.text = 'None'

            elif trendline_type == 'Logarithmic':
                try:
                    # Transform x_data for logarithmic fit
                    log_x_data = np.log(x_data)

                    # Calculate the logarithmic trendline
                    z = np.polyfit(log_x_data, y_data, 1)
                    p = np.poly1d(z)

                    try:
                        self.ax.plot(x_data, p(np.log(x_data)), 'r--')
                    except Exception as e:
                        logger.exception(f'Graphing ] Logarithmic trendline error: {e}')
                except Exception as e:
                    logger.exception(f'[Graphing  ] Could not fit logarithmic trendline: {e}')
                    self.ids.trendline_spinner.text = 'None'

            self.update_graph()

    def regenerate_graph(self):
        self.initialize_graph()
        self.set_x_axis()
        self.set_y_axis()
        if self.trendline_enabled:
            self.update_trendline()

    def initialize_graph(self):
        # plt.clf() only clears the figure contents; the figure object
        # itself (along with its GDI handle on Windows) leaks. Close the
        # previous figure explicitly before allocating a new one.
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
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
        self._source_csv = file
        self.initialize_graph()
        try:
            self.graph_df = pd.read_csv(file)
            self.available_axes = list(self.graph_df.keys())
            if self.available_axes[0] == 'file':
                self.available_axes = self.available_axes[1:]
            if 'time' in self.available_axes:
                # Parse to a pandas datetime64 column. A list comprehension of
                # datetime.strptime objects yields an object-dtype column, and
                # the .dt accessor (used by the time-axis trendline) rejects
                # object dtype -- that crashed update_trendline on a time axis.
                self.graph_df['time'] = pd.to_datetime(self.graph_df['time'], format='%c')

            self.update_available_axes()
            self.set_x_axis()
            self.set_y_axis()

        except Exception as e:
            logger.exception(f'Graph Generation | Set graphing source | {e}')

    def set_post_processing_module(self, postprocessingmodule):
        self._post = postprocessingmodule


# ============================================================================
# CellCountControls -- Cell Counting and Analysis
# ============================================================================


class CellCountControls(BoxLayout):
    ENABLE_PREVIEW_AUTO_REFRESH = False

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('LVP Main: CellCountControls.__init__()')
        self._preview_source_image = None
        self._preview_source_significant_bits = 16
        self._preview_image = None
        self._post = post_processing.PostProcessing()
        self._settings = post_processing.default_cell_count_method()
        self._set_ui_to_settings(self._settings)

    def apply_method_to_preview_image(self) -> None:
        gui_logger.button('APPLY_METHOD_TO_PREVIEW')
        run_reported(self._regenerate_image_preview, None, 'APPLY_METHOD_TO_PREVIEW')

    # Decorate function to show popup and run the code below in a thread
    @show_popup
    def apply_method_to_folder(self, popup, path):
        popup.title = 'Processing Cell Count Method'
        pre_text = f'Applying method to folder: {path}'
        popup.text = pre_text

        popup.progress = 0
        popup.auto_dismiss = False

        settings = self._settings
        _run_build(
            lambda progress: _app_ctx.ctx.session.post_processing.count_cells(
                path, method=settings, on_progress=progress
            ),
            popup,
            'APPLY_CELL_COUNT_TO_FOLDER',
        )

    def set_post_processing_module(self, post_processing_module):
        self._post = post_processing_module

    def get_current_settings(self):
        return self._settings

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
        self.ids.slider_cell_count_threshold_id.value = settings['segmentation']['parameters'][
            'threshold'
        ]
        self.ids.slider_cell_count_area_id.value = self._area_range_slider_physical_to_values(
            (settings['filters']['area']['min'], settings['filters']['area']['max'])
        )

        self.ids.slider_cell_count_perimeter_id.value = (
            self._perimeter_range_slider_physical_to_values(
                (settings['filters']['perimeter']['min'], settings['filters']['perimeter']['max'])
            )
        )
        self.ids.slider_cell_count_sphericity_id.value = (
            settings['filters']['sphericity']['min'],
            settings['filters']['sphericity']['max'],
        )
        self.ids.slider_cell_count_min_intensity_id.value = (
            settings['filters']['intensity']['min']['min'],
            settings['filters']['intensity']['min']['max'],
        )
        self.ids.slider_cell_count_mean_intensity_id.value = (
            settings['filters']['intensity']['mean']['min'],
            settings['filters']['intensity']['mean']['max'],
        )
        self.ids.slider_cell_count_max_intensity_id.value = (
            settings['filters']['intensity']['max']['min'],
            settings['filters']['intensity']['max']['max'],
        )

        self.slider_adjustment_area()
        self.slider_adjustment_perimeter()
        self._regenerate_image_preview()

    def set_preview_source_file(self, file) -> None:
        # One read returns pixels AND their payload depth, so the preview always
        # scales by the source's true depth and cannot read the two out of sync.
        try:
            image, significant_bits = image_utils.load_pixels(file)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f'[LVP Main  ] Cell-count preview could not load {file}: {e}')
            return
        self._preview_source_significant_bits = significant_bits
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
        max_perimeter_pixels = 2 * ((2 * image.shape[0]) + (2 * image.shape[1]))
        max_perimeter_um = max_perimeter_pixels / pixels_per_um
        return max_perimeter_um

    def update_filter_max(self, image):
        max_area_um2 = self.calculate_area_filter_max(image=image)
        max_perimeter_um = self.calculate_perimeter_filter_max(image=image)

        self.ids.slider_cell_count_area_id.max = int(
            self._area_range_slider_physical_to_values(physical_values=(0, max_area_um2))[1]
        )
        self.ids.slider_cell_count_perimeter_id.max = int(
            self._perimeter_range_slider_physical_to_values(physical_values=(0, max_perimeter_um))[
                1
            ]
        )

        self.slider_adjustment_area()
        self.slider_adjustment_perimeter()

    def set_preview_source(self, image) -> None:
        self._preview_source_image = image
        self._preview_image = image
        img_widget = self.ids['cell_count_image_id']
        # The stored source stays full-depth for the cell-count math; the display
        # blit gets an 8-bit copy scaled by the source's payload depth, so a
        # right-aligned 12-bit frame renders correctly instead of being blitted
        # as raw 16-bit bytes.
        display_image = image_utils.convert_to_8bit(image, self._preview_source_significant_bits)
        img_widget.texture = image_utils_kivy.image_to_texture(
            image=display_image, existing=img_widget.texture
        )
        self.update_filter_max(image=image)
        self._regenerate_image_preview()

    # Save settings to JSON file
    def save_method_as(self, file='./data/cell_count_method.json'):
        logger.info(f'[LVP Main  ] CellCountContent.save_method_as({file})')
        # Resolve relative paths against source_path instead of relying on CWD
        if not os.path.isabs(file):
            file = os.path.join(_app_ctx.ctx.source_path, file)
        method = self._settings
        run_reported(
            lambda: post_processing.save_cell_count_method(method, file),
            None,
            'SAVE_CELL_COUNT_METHOD',
        )

    def load_method_from_file(self, file):
        logger.info(f'[LVP Main  ] CellCountContent.load_method_from_file({file})')

        def _load():
            self._settings = post_processing.load_cell_count_method(file)

        run_reported(
            _load, lambda: self._set_ui_to_settings(self._settings), 'LOAD_CELL_COUNT_METHOD'
        )

    def _regenerate_image_preview(self):
        if self._preview_source_image is None:
            return

        image, _ = self._post.preview_cell_count(
            image=self._preview_source_image,
            settings=self._settings,
            significant_bits=self._preview_source_significant_bits,
        )

        self._preview_image = image

        img_widget = _app_ctx.ctx.cell_count_content.ids['cell_count_image_id']
        img_widget.texture = image_utils_kivy.image_to_texture(
            image=image, existing=img_widget.texture
        )

    def slider_adjustment_threshold(self):
        value = self.ids['slider_cell_count_threshold_id'].value
        gui_logger.slider('CELL_COUNT_THRESHOLD', value)
        self._settings['segmentation']['parameters']['threshold'] = value

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_area(self):
        low, high = self._area_range_slider_values_to_physical(
            (
                self.ids['slider_cell_count_area_id'].value[0],
                self.ids['slider_cell_count_area_id'].value[1],
            )
        )

        gui_logger.slider('CELL_COUNT_AREA_RANGE', f'{int(low)}-{int(high)}')
        self._settings['filters']['area']['min'], self._settings['filters']['area']['max'] = (
            low,
            high,
        )

        self.ids['label_cell_count_area_id'].text = f'{int(low)}-{int(high)} \u03bcm\u00b2'

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_perimeter(self):
        low, high = self._perimeter_range_slider_values_to_physical(
            (
                self.ids['slider_cell_count_perimeter_id'].value[0],
                self.ids['slider_cell_count_perimeter_id'].value[1],
            )
        )

        gui_logger.slider('CELL_COUNT_PERIMETER_RANGE', f'{int(low)}-{int(high)}')
        (
            self._settings['filters']['perimeter']['min'],
            self._settings['filters']['perimeter']['max'],
        ) = low, high

        self.ids['label_cell_count_perimeter_id'].text = f'{int(low)}-{int(high)} \u03bcm'

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_sphericity(self):
        lo = self.ids['slider_cell_count_sphericity_id'].value[0]
        hi = self.ids['slider_cell_count_sphericity_id'].value[1]
        gui_logger.slider('CELL_COUNT_SPHERICITY_RANGE', f'{lo}-{hi}')
        self._settings['filters']['sphericity']['min'] = lo
        self._settings['filters']['sphericity']['max'] = hi

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_min_intensity(self):
        lo = self.ids['slider_cell_count_min_intensity_id'].value[0]
        hi = self.ids['slider_cell_count_min_intensity_id'].value[1]
        gui_logger.slider('CELL_COUNT_MIN_INTENSITY_RANGE', f'{lo}-{hi}')
        self._settings['filters']['intensity']['min']['min'] = lo
        self._settings['filters']['intensity']['min']['max'] = hi

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_mean_intensity(self):
        lo = self.ids['slider_cell_count_mean_intensity_id'].value[0]
        hi = self.ids['slider_cell_count_mean_intensity_id'].value[1]
        gui_logger.slider('CELL_COUNT_MEAN_INTENSITY_RANGE', f'{lo}-{hi}')
        self._settings['filters']['intensity']['mean']['min'] = lo
        self._settings['filters']['intensity']['mean']['max'] = hi

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_max_intensity(self):
        lo = self.ids['slider_cell_count_max_intensity_id'].value[0]
        hi = self.ids['slider_cell_count_max_intensity_id'].value[1]
        gui_logger.slider('CELL_COUNT_MAX_INTENSITY_RANGE', f'{lo}-{hi}')
        self._settings['filters']['intensity']['max']['min'] = lo
        self._settings['filters']['intensity']['max']['max'] = hi

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def flourescent_mode_toggle(self):
        gui_logger.toggle(
            'CELL_COUNT_FLUORESCENT_MODE',
            bool(self.ids['cell_count_fluorescent_mode_id'].active),
        )
        self._settings['context']['fluorescent_mode'] = self.ids[
            'cell_count_fluorescent_mode_id'
        ].active

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def commit_pixels_per_um(self) -> None:
        """Hand a typed pixels-per-micron to the method's owner, once it is typed whole.

        Bound to the box losing focus, which Enter also does, not to every
        keystroke: a partly typed '0.5' passes through '0'. The method takes
        the value whether or not a preview image is loaded, since a folder
        count needs none. A refused value is reported by the boundary, and
        the redraw puts the box back to the method's value.
        """
        typed = self.ids['text_cell_count_pixels_per_um_id'].text
        gui_logger.text_input('CELL_COUNT_PIXELS_PER_UM', typed)

        def _apply():
            self._settings = post_processing.with_pixels_per_um(self._settings, typed)

        run_reported(_apply, self._show_pixels_per_um, 'CELL_COUNT_PIXELS_PER_UM')

    def _show_pixels_per_um(self) -> None:
        box = self.ids['text_cell_count_pixels_per_um_id']
        held = str(self._settings['context']['pixels_per_um'])
        if box.text != held:
            # The box is put back to what the method holds: the record pair
            # says what was typed and what the app kept instead. Assigning
            # .text does not dispatch the focus event the commit is bound to.
            gui_logger.text_input('CELL_COUNT_PIXELS_PER_UM_APPLIED', held)
            box.text = held
        if self._preview_image is not None:
            self.update_filter_max(image=self._preview_image)


# ============================================================================
# PostProcessingAccordion -- Post-Processing Panel Container
# ============================================================================


class PostProcessingAccordion(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = self.__class__.__name__
        self.post = post_processing.PostProcessing()
        self.raw_images_folder = './capture/'
        self.combine_colors = False  # True if raw images are in separate red/green/blue channels and need to be first combined
        self.ext = 'tiff'
        self.stitching_method = 'position'
        self.stitched_save_name = 'last_composite_img.tiff'
        self.positions_file = './capture/2x2.tsv'
        self.pos2pix = 2630  # relevant if stitching method is position. The scale conversion for pos info into pixels

        from modules.common_utils import DEFAULT_STAGE_TRAVEL_UM

        self.tiling_min = {
            'x': int(DEFAULT_STAGE_TRAVEL_UM['x']),
            'y': int(DEFAULT_STAGE_TRAVEL_UM['y']),
        }

        self.tiling_max = {'x': 0, 'y': 0}

        self.tiling_count = {'x': 1, 'y': 1}

        self.accordion_item_states = {
            'cell_count_accordion_id': None,
            'stitch_accordion_id': None,
            'composite_gen_accordion_id': None,
            'zprojection_accordion_id': None,
            'create_avi_accordion_id': None,
        }
        self.init_cell_count()
        self._graphing_popup = None

    @staticmethod
    def accordion_item_state(accordion_item):
        if accordion_item.collapse:
            return 'closed'
        return 'open'

    def hide_stitch(self):
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
            'cell_count_accordion_id': self.accordion_item_state(
                self.ids['cell_count_accordion_id']
            ),
            'stitch_accordion_id': self.accordion_item_state(self.ids['stitch_accordion_id']),
            'composite_gen_accordion_id': self.accordion_item_state(
                self.ids['composite_gen_accordion_id']
            ),
            'zprojection_accordion_id': self.accordion_item_state(
                self.ids['zprojection_accordion_id']
            ),
            'create_avi_accordion_id': self.accordion_item_state(
                self.ids['create_avi_accordion_id']
            ),
        }

    def accordion_collapse(self):

        new_accordion_item_states = self.get_accordion_item_states()

        changed_items = []
        for accordion_item_id, prev_accordion_item_state in self.accordion_item_states.items():
            if new_accordion_item_states[accordion_item_id] == prev_accordion_item_state:
                # No change
                continue

            # Update state and add state change to list
            self.accordion_item_states[accordion_item_id] = self.accordion_item_state(
                self.ids[accordion_item_id]
            )
            changed_items.append(accordion_item_id)

    def init_cell_count(self):
        self._cell_count_popup = None

    def convert_to_avi(self):
        logger.debug('[LVP Main  ] PostProcessingAccordian.convert_to_avi() not yet implemented')

    def open_cell_count(self):
        gui_logger.button('OPEN_CELL_COUNT')
        ctx = _app_ctx.ctx
        if self._cell_count_popup is None:
            ctx.cell_count_content.set_post_processing_module(self.post)
            self._cell_count_popup = Popup(
                title='Post Processing - Object Analysis',
                content=ctx.cell_count_content,
                size_hint=(0.85, 0.85),
                auto_dismiss=True,
            )

        self._cell_count_popup.open()

    def open_graphing(self):
        gui_logger.button('OPEN_GRAPHING')
        ctx = _app_ctx.ctx
        if self._graphing_popup is None:
            ctx.graphing_controls.set_post_processing_module(self.post)
            self._graphing_popup = Popup(
                title='Post Processing - Object Plotting',
                content=ctx.graphing_controls,
                size_hint=(0.85, 0.85),
                auto_dismiss=True,
            )

        self._graphing_popup.open()


def open_last_save_folder():
    ctx = _app_ctx.ctx

    OS_FOLDER_MAP = {'win32': 'explorer', 'darwin': 'open', 'linux': 'xdg-open'}

    if sys.platform not in OS_FOLDER_MAP:
        logger.info(
            f'[LVP Main  ] PostProcessing.open_folder() not yet implemented for {sys.platform} platform'
        )
        return

    command = OS_FOLDER_MAP[sys.platform]
    if ctx.last_save_folder is None:
        subprocess.Popen([command, str(pathlib.Path(ctx.settings['live_folder']).resolve())])
    else:
        subprocess.Popen([command, str(ctx.last_save_folder)])


# ============================================================================
# CellCountDisplay
# ============================================================================


class CellCountDisplay(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
