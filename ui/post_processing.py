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
from modules.post_processing_api import BuildResult
from modules.stitcher import Stitcher
import modules.zprojector as zprojector
import modules.graph_analysis as graph_analysis
import modules.post_processing as post_processing
import modules.image_utils as image_utils
import ui.image_utils_kivy as image_utils_kivy
import modules.app_context as _app_ctx
from ui.ui_helpers import run_reported, run_unasked, submit_reported

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
        popup.text = result.message
        degraded = isinstance(result, BuildResult) and bool(result.degraded_outputs)
        Clock.schedule_once(lambda dt: popup.dismiss(), 5 if degraded else 2)

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

    def set_source(self, path) -> None:
        """Enhance the image or folder at ``path``; the API says which it is."""
        target = pathlib.Path(path)
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
                run_unasked(
                    lambda: scope_display.hold_derived_image(display_image, significant_bits),
                    'ENHANCE_PREVIEW',
                )

        Clock.schedule_once(_show, 0)

    def _export_done(self, result) -> None:
        self.busy = False
        if result is None:
            self.status_text = ''
            return
        self.last_output_folder = str(result.output_folder)
        self.status_text = result.message


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


# The axis spinners' text when no axis is chosen, as the kv rule sets it.
_NO_X_AXIS = 'X-Axis'
_NO_Y_AXIS = 'Y-Axis'


class GraphingControls(BoxLayout):
    x_axis_label = 'X-Axis'
    y_axis_label = 'Y-Axis'
    graph_title = ''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('LVP Main: GraphingControls.__init__()')
        self.fig = None
        self._post = post_processing.PostProcessing()
        self.graphing_area = self.ids.graphing_area
        self.graph_widget = None
        self.graph_df = None
        self._x_axes = []
        self._y_axes = []
        self.selected_x_axis = None
        self.selected_y_axis = None
        self._trendline_kind = graph_analysis.NO_TRENDLINE
        self._trendline = None
        self._redraw_graph()

    # Each spinner handler returns when the spinner shows what is already
    # chosen: that is the redraw writing the stored choice back, not a person
    # choosing, so it is neither logged nor fitted again.

    def set_x_axis(self):
        axis = self.ids['graphing_x_axis_spinner'].text
        if axis == (self.selected_x_axis or _NO_X_AXIS):
            return
        gui_logger.select('GRAPHING_X_AXIS', axis)
        self.selected_x_axis = axis
        self.ids.x_axis_label_input.text = axis
        run_reported(self._fit_trendline, self._redraw_graph, 'GRAPHING_X_AXIS')

    def set_y_axis(self):
        axis = self.ids['graphing_y_axis_spinner'].text
        if axis == (self.selected_y_axis or _NO_Y_AXIS):
            return
        gui_logger.select('GRAPHING_Y_AXIS', axis)
        self.selected_y_axis = axis
        self.ids.y_axis_label_input.text = axis
        run_reported(self._fit_trendline, self._redraw_graph, 'GRAPHING_Y_AXIS')

    def update_trendline(self):
        kind = self.ids.trendline_spinner.text
        if kind == self._trendline_kind:
            return
        gui_logger.select('TRENDLINE', kind)
        self._trendline_kind = kind
        run_reported(self._fit_trendline, self._redraw_graph, 'TRENDLINE')

    def _fit_trendline(self) -> None:
        """Fit the chosen trendline to the chosen axes.

        A refused fit leaves no trendline chosen, so the spinner reads None
        and a later axis change does not ask for the refused fit again.
        """
        self._trendline = None
        if self._trendline_kind == graph_analysis.NO_TRENDLINE:
            return
        if self.selected_x_axis is None or self.selected_y_axis is None:
            return
        kind, self._trendline_kind = self._trendline_kind, graph_analysis.NO_TRENDLINE
        self._trendline = graph_analysis.fit_trendline(
            kind, self.graph_df[self.selected_x_axis], self.graph_df[self.selected_y_axis]
        )
        self._trendline_kind = kind

    def _redraw_graph(self) -> None:
        """Draw what is chosen: the data, its two axes and the fitted trendline."""
        self.ids.graphing_x_axis_spinner.values = self._x_axes
        self.ids.graphing_y_axis_spinner.values = self._y_axes
        self.ids.graphing_x_axis_spinner.text = self.selected_x_axis or _NO_X_AXIS
        self.ids.graphing_y_axis_spinner.text = self.selected_y_axis or _NO_Y_AXIS
        self.initialize_graph()
        kinds = graph_analysis.TRENDLINE_KINDS
        if self.selected_x_axis is not None and self.selected_y_axis is not None:
            x = self.graph_df[self.selected_x_axis]
            y = self.graph_df[self.selected_y_axis]
            kinds = graph_analysis.trendline_kinds(x, y)
            self.ax.scatter(x, y)
            if pd.api.types.is_datetime64_any_dtype(x):
                self.ax.xaxis.set_major_formatter(
                    ConciseDateFormatter(self.ax.xaxis.get_major_locator())
                )
            if self._trendline is not None:
                self.ax.plot(self._trendline.x, self._trendline.y, 'r--')
        self.ids.trendline_spinner.values = (graph_analysis.NO_TRENDLINE, *kinds)
        self.ids.trendline_spinner.text = self._trendline_kind
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

    def update_graph_title(self):
        self.ax.set_title(self.ids.graph_title_input.text)
        self.graph_title = self.ids.graph_title_input.text
        self.update_graph()

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
        run_reported(lambda: self._load_source(file), self._redraw_graph, 'LOAD_GRAPHING_DATA')

    def _load_source(self, file) -> None:
        """Take *file* as the graph's data; the axis choices and trendline start over.

        A second file's columns need not be the first's, so a choice made
        against the first is not carried onto the second.
        """
        self.graph_df = post_processing.read_cell_count_results(file)
        self._x_axes, self._y_axes = post_processing.results_axes(self.graph_df)
        self.selected_x_axis = None
        self.selected_y_axis = None
        self._trendline_kind = graph_analysis.NO_TRENDLINE
        self._trendline = None

    def set_post_processing_module(self, postprocessingmodule):
        self._post = postprocessingmodule


# ============================================================================
# CellCountControls -- Cell Counting and Analysis
# ============================================================================


# The size filters' bounds are in microns whatever the preview's scale: an
# image with no scale is counted in pixels, and a bound on it is refused.
_AREA_UNIT = '\u03bcm\u00b2'
_PERIMETER_UNIT = '\u03bcm'


def _scale_text(pixels_per_um) -> str:
    """The scale box's text: the method's override, or empty for the image's own."""
    return '' if pixels_per_um is None else str(pixels_per_um)


def _range_text(low, high, unit: str) -> str:
    """A size filter's bounds as a person reads them; None is no bound."""
    if low is None and high is None:
        text = 'any'
    elif low is None:
        text = f'\u2264 {int(high)}'
    elif high is None:
        text = f'\u2265 {int(low)}'
    else:
        text = f'{int(low)}-{int(high)}'
    return f'{text} {unit}'.strip()


class CellCountControls(BoxLayout):
    ENABLE_PREVIEW_AUTO_REFRESH = False

    done = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('LVP Main: CellCountControls.__init__()')
        self._preview_source_image = None
        self._preview_source_significant_bits = 16
        # The scale the preview file states (um per pixel), or None.
        self._preview_pixel_size_um = None
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
        """Show the method on the panel. Writes nothing back to it."""
        self.ids.text_cell_count_pixels_per_um_id.text = _scale_text(
            settings['context']['pixels_per_um']
        )
        self.ids.cell_count_fluorescent_mode_id.active = settings['context']['fluorescent_mode']
        self.ids.slider_cell_count_threshold_id.value = settings['segmentation']['parameters'][
            'threshold'
        ]
        self._show_size_filters()
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
        self._regenerate_image_preview()

    def _preview_scale(self) -> float | None:
        """Pixels per micron the preview counts at, or None for pixels."""
        return post_processing.cell_count_scale(self._settings, self._preview_pixel_size_um)

    def _size_bounds_from_slider(self, slider_id, to_physical):
        """The bounds a range slider shows; an end at its stop is no bound."""
        slider = self.ids[slider_id]
        low, high = to_physical((slider.value[0], slider.value[1]))
        return (
            None if slider.value[0] <= slider.min else low,
            None if slider.value[1] >= slider.max else high,
        )

    def _show_size_filters(self) -> None:
        """Put the area and perimeter bounds on their sliders and labels.

        No bound is the slider's stop at that end.
        """
        area_unit, perimeter_unit = _AREA_UNIT, _PERIMETER_UNIT
        for slider_id, label_id, size, unit, to_values in (
            (
                'slider_cell_count_area_id',
                'label_cell_count_area_id',
                'area',
                area_unit,
                self._area_range_slider_physical_to_values,
            ),
            (
                'slider_cell_count_perimeter_id',
                'label_cell_count_perimeter_id',
                'perimeter',
                perimeter_unit,
                self._perimeter_range_slider_physical_to_values,
            ),
        ):
            slider = self.ids[slider_id]
            bounds = self._settings['filters'][size]
            low = slider.min if bounds['min'] is None else to_values((bounds['min'], 0))[0]
            high = slider.max if bounds['max'] is None else to_values((0, bounds['max']))[1]
            slider.value = (low, high)
            self.ids[label_id].text = _range_text(bounds['min'], bounds['max'], unit)

    def set_preview_source_file(self, file) -> None:
        # One read returns pixels AND their payload depth, so the preview always
        # scales by the source's true depth and cannot read the two out of sync.
        def _load():
            image, significant_bits = post_processing.read_cell_count_image(file)
            self._preview_source_significant_bits = significant_bits
            self._preview_pixel_size_um = image_utils.read_pixel_size_um(file)
            self.set_preview_source(image=image)

        run_reported(_load, None, 'LOAD_CELL_COUNT_INPUT_IMAGE')

    def calculate_area_filter_max(self, image):
        """The largest area the preview can hold, in microns at its scale.

        With no scale it is the pixel count: the slider's reach only, since a
        bound on an image with no scale is refused.
        """
        pixels_per_um = self._preview_scale()
        max_area_pixels = image.shape[0] * image.shape[1]
        if pixels_per_um is None:
            return max_area_pixels
        return max_area_pixels / (pixels_per_um**2)

    def calculate_perimeter_filter_max(self, image):
        """The largest perimeter the preview can hold, in microns at its scale.

        With no scale it is in pixels: the slider's reach only.
        """
        pixels_per_um = self._preview_scale()
        # Assume max perimeter will never need to be larger than 2x frame size border
        # The 2x is to provide margin for handling various curvatures
        max_perimeter_pixels = 2 * ((2 * image.shape[0]) + (2 * image.shape[1]))
        if pixels_per_um is None:
            return max_perimeter_pixels
        return max_perimeter_pixels / pixels_per_um

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
        self._show_size_filters()

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
            pixels_per_um=self._preview_scale(),
            name='The preview image',
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
        low, high = self._size_bounds_from_slider(
            'slider_cell_count_area_id', self._area_range_slider_values_to_physical
        )
        gui_logger.slider('CELL_COUNT_AREA_RANGE', _range_text(low, high, ''))
        self._settings['filters']['area'] = {'min': low, 'max': high}
        self.ids['label_cell_count_area_id'].text = _range_text(low, high, _AREA_UNIT)

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def show_area_range_dragged(self) -> None:
        """The area label follows a drag; the bound is taken on release."""
        low, high = self._size_bounds_from_slider(
            'slider_cell_count_area_id', self._area_range_slider_values_to_physical
        )
        self.ids['label_cell_count_area_id'].text = _range_text(low, high, _AREA_UNIT)

    def slider_adjustment_perimeter(self):
        low, high = self._size_bounds_from_slider(
            'slider_cell_count_perimeter_id', self._perimeter_range_slider_values_to_physical
        )
        gui_logger.slider('CELL_COUNT_PERIMETER_RANGE', _range_text(low, high, ''))
        self._settings['filters']['perimeter'] = {'min': low, 'max': high}
        self.ids['label_cell_count_perimeter_id'].text = _range_text(low, high, _PERIMETER_UNIT)

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def show_perimeter_range_dragged(self) -> None:
        """The perimeter label follows a drag; the bound is taken on release."""
        low, high = self._size_bounds_from_slider(
            'slider_cell_count_perimeter_id', self._perimeter_range_slider_values_to_physical
        )
        self.ids['label_cell_count_perimeter_id'].text = _range_text(low, high, _PERIMETER_UNIT)

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
        held = _scale_text(self._settings['context']['pixels_per_um'])
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
