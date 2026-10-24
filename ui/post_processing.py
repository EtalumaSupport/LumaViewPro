# Copyright Etaluma, Inc.
import dataclasses
import logging
import pathlib
import subprocess
import sys
from typing import ClassVar

import numpy as np

from kivy.clock import Clock
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup

from ui.progress_popup import show_popup
from modules import gui_logger
from modules.cell_count_method import CellCountMethod, FilterBounds
from modules.post_processing_api import BuildResult, CellCountSizeScale, ZProject
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
    gets the result, or None, once the build has ended, for a panel that
    holds a state for its length.
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
        popup.show_outcome(result.message)
        # A result that names a fallback or files left out stays up long
        # enough to be read; the build says which, the panel only holds it.
        noted = isinstance(result, BuildResult) and bool(
            result.degraded_outputs or result.excluded_inputs
        )
        Clock.schedule_once(lambda dt: popup.dismiss(), 5 if noted else 2)

    submit_reported(_build, _show, label, lane=_app_ctx.ctx.session.post_processing.lane)


class QuickEnhanceControls(BoxLayout):
    """GUI shell for the non-destructive Quick Enhance derived-file path."""

    done = BooleanProperty(False)
    busy = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _app_ctx.register_early('quick_enhance_controls', self)

    def set_source(self, path) -> None:
        """Enhance the image or folder at ``path``; the API says which it is."""
        target = pathlib.Path(path)
        self.busy = True
        gui_logger.display('QUICK_ENHANCE', 'busy')
        self.export(target)

    @show_popup
    def export(self, popup, target: pathlib.Path) -> None:
        popup.title = 'Enhance'
        popup.text = ''
        popup.progress = 0
        popup.auto_dismiss = False

        def _build(progress):
            return _app_ctx.ctx.session.post_processing.enhance(
                target, on_progress=progress, on_derived_image=self._queue_derived_image
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
        # Its progress and outcome words are the build popup's, recorded there.
        self.busy = False
        gui_logger.display('QUICK_ENHANCE', 'idle')


class StitchControls(BoxLayout):
    done = BooleanProperty(False)
    stitching_mode = StringProperty('Quality')
    # The words each button shows for a mode of the session's stitch_modes().
    _MODE_VALUES: ClassVar[dict[str, str]] = {
        'Quality': 'quality',
        'Fast Preview': 'fast_preview',
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
        stitching_mode = self._MODE_VALUES[mode_label]
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
        methods = _app_ctx.ctx.session.post_processing.projection_methods()
        self.ids['zprojection_method_spinner'].values = methods
        # Opens on the API's default projection, as a REST or script caller's does.
        self.ids['zprojection_method_spinner'].text = ZProject.method

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
                frame_rate_fps=fps,
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
# The trendline spinner's choice of none.
_NO_TRENDLINE = 'None'


class GraphingControls(BoxLayout):
    x_axis_label = 'X-Axis'
    y_axis_label = 'Y-Axis'
    graph_title = ''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('LVP Main: GraphingControls.__init__()')
        self.graphing_area = self.ids.graphing_area
        self.graph_widget = None
        # The form the session draws and saves from; the panel holds no data.
        self._results = None
        self._x_axes = []
        self._y_axes = []
        self._trendline_kinds = ()
        self.selected_x_axis = None
        self.selected_y_axis = None
        self._trendline_kind = _NO_TRENDLINE
        self._figure = None

    def show(self) -> None:
        """Draw the graph as the form stands: the empty frame until a file and axes are chosen."""
        run_reported(self._draw, self._redraw_graph, 'OPEN_GRAPHING')

    # Each spinner handler returns when the spinner shows what is already
    # chosen: that is the redraw writing the stored choice back, so nothing is
    # drawn again. A person's pick is recorded by the spinner itself.

    def set_x_axis(self):
        axis = self.ids['graphing_x_axis_spinner'].text
        if axis == (self.selected_x_axis or _NO_X_AXIS):
            return
        self.selected_x_axis = axis
        self.ids.x_axis_label_input.text = axis
        run_reported(self._take_axes, self._redraw_graph, 'GRAPHING_X_AXIS')

    def set_y_axis(self):
        axis = self.ids['graphing_y_axis_spinner'].text
        if axis == (self.selected_y_axis or _NO_Y_AXIS):
            return
        self.selected_y_axis = axis
        self.ids.y_axis_label_input.text = axis
        run_reported(self._take_axes, self._redraw_graph, 'GRAPHING_Y_AXIS')

    def update_trendline(self):
        kind = self.ids.trendline_spinner.text
        if kind == self._trendline_kind:
            return
        self._trendline_kind = kind
        run_reported(self._draw, self._redraw_graph, 'TRENDLINE')

    def _take_axes(self) -> None:
        """Ask the session which trendlines the chosen axes take, then draw them."""
        self._trendline_kinds = _app_ctx.ctx.session.post_processing.graph_options(
            self._results, x=self.selected_x_axis, y=self.selected_y_axis
        ).trendlines
        self._draw()

    def _draw(self) -> None:
        """Have the session draw what is chosen; the redraw shows it.

        A refused trendline leaves none chosen, so the spinner reads None, the
        redraw draws the graph without it, and a later axis change does not
        ask for the refused fit again.
        """
        self._figure = None
        if self._trendline_kind == _NO_TRENDLINE:
            return
        if self.selected_x_axis is None or self.selected_y_axis is None:
            return
        kind, self._trendline_kind = self._trendline_kind, _NO_TRENDLINE
        self._figure = self._drawn(kind)
        self._trendline_kind = kind

    def _drawn(self, kind: str):
        """The session's figure of the form, with the *kind* of trendline."""
        return _app_ctx.ctx.session.post_processing.draw_graph(
            self._results,
            x=self.selected_x_axis,
            y=self.selected_y_axis,
            trendline=None if kind == _NO_TRENDLINE else kind,
            x_label=self.x_axis_label,
            y_label=self.y_axis_label,
            title=self.graph_title,
        )

    def _redraw_graph(self) -> None:
        """Show what is chosen: the spinners' choices and the session's figure of them."""
        self.ids.graphing_x_axis_spinner.values = self._x_axes
        self.ids.graphing_y_axis_spinner.values = self._y_axes
        self.ids.graphing_x_axis_spinner.text = self.selected_x_axis or _NO_X_AXIS
        self.ids.graphing_y_axis_spinner.text = self.selected_y_axis or _NO_Y_AXIS
        self.ids.trendline_spinner.values = (_NO_TRENDLINE, *self._trendline_kinds)
        self.ids.trendline_spinner.text = self._trendline_kind
        if self._figure is None:
            self._figure = self._drawn(self._trendline_kind)
        if self.graph_widget is None:
            from ui.figure_canvas import FigureCanvasKivyAgg

            self.graph_widget = FigureCanvasKivyAgg(self._figure)
            self.graphing_area.add_widget(self.graph_widget)
        else:
            self.graph_widget.show(self._figure)

    def update_x_axis_label(self):
        self.x_axis_label = self.ids.x_axis_label_input.text
        run_reported(self._draw, self._redraw_graph, 'GRAPH_X_AXIS_LABEL')

    def update_y_axis_label(self):
        self.y_axis_label = self.ids.y_axis_label_input.text
        run_reported(self._draw, self._redraw_graph, 'GRAPH_Y_AXIS_LABEL')

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
        self.graph_title = self.ids.graph_title_input.text
        run_reported(self._draw, self._redraw_graph, 'GRAPH_TITLE')

    def save_graph(self, filepath):
        """Have the session save the graph the form describes to *filepath*."""
        post = _app_ctx.ctx.session.post_processing
        results, x, y = self._results, self.selected_x_axis, self.selected_y_axis
        trendline = None if self._trendline_kind == _NO_TRENDLINE else self._trendline_kind
        labels = {
            'x_label': self.x_axis_label,
            'y_label': self.y_axis_label,
            'title': self.graph_title,
        }
        submit_reported(
            lambda: post.graph(results, filepath, x=x, y=y, trendline=trendline, **labels),
            None,
            'SAVE_GRAPH',
            lane=post.lane,
        )

    def set_graphing_source(self, file):
        run_reported(lambda: self._load_source(file), self._redraw_graph, 'LOAD_GRAPHING_DATA')

    def _load_source(self, file) -> None:
        """Take *file* as the graph's data; the axis choices and trendline start over.

        A second file's columns need not be the first's, so a choice made
        against the first is not carried onto the second.
        """
        options = _app_ctx.ctx.session.post_processing.graph_options(file)
        self._results = file
        self._x_axes, self._y_axes = list(options.x_axes), list(options.y_axes)
        self._trendline_kinds = options.trendlines
        self.selected_x_axis = None
        self.selected_y_axis = None
        self._trendline_kind = _NO_TRENDLINE
        self._figure = None


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

    def __init__(self, method: CellCountMethod, size_scale: CellCountSizeScale, **kwargs):
        super().__init__(**kwargs)
        logger.info('LVP Main: CellCountControls.__init__()')
        self._preview_source_image = None
        self._preview_source_significant_bits = 16
        # The scale the preview file states (um per pixel), or None.
        self._preview_pixel_size_um = None
        self._preview_image = None
        # The method the panel shows and counts by, replaced whole on each change.
        self._method = method
        # What the size sliders stand for, the session's answer for the
        # preview image and the method; asked again when either's scale changes.
        self._size_scale = size_scale
        self._set_ui_to_method(self._method)

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

        method = self._method
        _run_build(
            lambda progress: _app_ctx.ctx.session.post_processing.count_cells(
                path, method=method, on_progress=progress
            ),
            popup,
            'APPLY_CELL_COUNT_TO_FOLDER',
        )

    def _take_size_scale(self) -> None:
        """Ask what the size sliders stand for at the preview image and the method, and size them."""
        shape = None if self._preview_source_image is None else self._preview_source_image.shape
        self._size_scale = _app_ctx.ctx.session.post_processing.cell_count_size_scale(
            self._method, shape=shape, pixel_size_um=self._preview_pixel_size_um
        )
        for slider_id, stops in (
            ('slider_cell_count_area_id', self._size_scale.area),
            ('slider_cell_count_perimeter_id', self._size_scale.perimeter),
        ):
            self.ids[slider_id].max = int(stops.positions_of((0, stops.reach))[1])

    def _set_ui_to_method(self, method: CellCountMethod):
        """Show the method on the panel. Writes nothing back to it."""
        filters = method.filters
        self.ids.text_cell_count_pixels_per_um_id.text = _scale_text(method.pixels_per_um)
        self.ids.cell_count_fluorescent_mode_id.active = method.fluorescent_mode
        self.ids.slider_cell_count_threshold_id.value = method.threshold_pct
        self._show_size_filters()
        for slider_id, bounds in (
            ('slider_cell_count_sphericity_id', filters.sphericity),
            ('slider_cell_count_min_intensity_id', filters.intensity_min_pct),
            ('slider_cell_count_mean_intensity_id', filters.intensity_mean_pct),
            ('slider_cell_count_max_intensity_id', filters.intensity_max_pct),
        ):
            self.ids[slider_id].value = (bounds.min, bounds.max)
        self._regenerate_image_preview()

    def _size_bounds_from_slider(self, slider_id, size):
        """The ``size`` bounds a range slider shows; an end at its stop is no bound."""
        slider = self.ids[slider_id]
        low, high = getattr(self._size_scale, size).size_at((slider.value[0], slider.value[1]))
        return (
            None if slider.value[0] <= slider.min else low,
            None if slider.value[1] >= slider.max else high,
        )

    def _show_size_filters(self) -> None:
        """Put the area and perimeter bounds on their sliders and labels.

        No bound is the slider's stop at that end.
        """
        filters = self._method.filters
        for slider_id, label_id, size, bounds, unit in (
            (
                'slider_cell_count_area_id',
                'label_cell_count_area_id',
                'area',
                filters.area_um2,
                _AREA_UNIT,
            ),
            (
                'slider_cell_count_perimeter_id',
                'label_cell_count_perimeter_id',
                'perimeter',
                filters.perimeter_um,
                _PERIMETER_UNIT,
            ),
        ):
            slider = self.ids[slider_id]
            stops = getattr(self._size_scale, size)
            low = slider.min if bounds.min is None else stops.positions_of((bounds.min, 0))[0]
            high = slider.max if bounds.max is None else stops.positions_of((0, bounds.max))[1]
            slider.value = (low, high)
            self.ids[label_id].text = _range_text(bounds.min, bounds.max, unit)

    def set_preview_source_file(self, file) -> None:
        # One read returns pixels AND their payload depth, so the preview always
        # scales by the source's true depth and cannot read the two out of sync.
        def _load():
            image, significant_bits = post_processing.read_cell_count_image(file)
            self._preview_source_significant_bits = significant_bits
            self._preview_pixel_size_um = image_utils.read_pixel_size_um(file)
            self.set_preview_source(image=image)

        run_reported(_load, None, 'LOAD_CELL_COUNT_INPUT_IMAGE')

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
        self._take_size_scale()
        self._show_size_filters()
        self._regenerate_image_preview()

    def save_method_as(self, file):
        logger.info(f'[LVP Main  ] CellCountContent.save_method_as({file})')
        method = self._method
        run_reported(
            lambda: _app_ctx.ctx.session.post_processing.save_cell_count_method(method, file),
            None,
            'SAVE_CELL_COUNT_METHOD',
        )

    def load_method_from_file(self, file):
        logger.info(f'[LVP Main  ] CellCountContent.load_method_from_file({file})')

        def _load():
            self._method = _app_ctx.ctx.session.post_processing.load_cell_count_method(file)

        def _show():
            # The loaded method may carry its own scale.
            self._take_size_scale()
            self._set_ui_to_method(self._method)

        run_reported(_load, _show, 'LOAD_CELL_COUNT_METHOD')

    def _regenerate_image_preview(self):
        if self._preview_source_image is None:
            return

        image = _app_ctx.ctx.session.post_processing.preview_cell_count(
            self._preview_source_image,
            self._method,
            significant_bits=self._preview_source_significant_bits,
            pixel_size_um=self._preview_pixel_size_um,
        ).image

        self._preview_image = image

        img_widget = _app_ctx.ctx.cell_count_content.ids['cell_count_image_id']
        img_widget.texture = image_utils_kivy.image_to_texture(
            image=image, existing=img_widget.texture
        )

    def _take_filter(self, name: str, low, high) -> None:
        """Hold a method whose *name* filter is bounded by *low* and *high*."""
        filters = dataclasses.replace(
            self._method.filters, **{name: FilterBounds(min=low, max=high)}
        )
        self._method = dataclasses.replace(self._method, filters=filters)

    def slider_adjustment_threshold(self):
        value = self.ids['slider_cell_count_threshold_id'].value
        gui_logger.slider('CELL_COUNT_THRESHOLD', value)
        self._method = dataclasses.replace(self._method, threshold_pct=value)

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_area(self):
        low, high = self._size_bounds_from_slider('slider_cell_count_area_id', 'area')
        gui_logger.slider('CELL_COUNT_AREA_RANGE', _range_text(low, high, ''))
        self._take_filter('area_um2', low, high)
        self.ids['label_cell_count_area_id'].text = _range_text(low, high, _AREA_UNIT)

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def show_area_range_dragged(self) -> None:
        """The area label follows a drag; the bound is taken on release."""
        low, high = self._size_bounds_from_slider('slider_cell_count_area_id', 'area')
        self.ids['label_cell_count_area_id'].text = _range_text(low, high, _AREA_UNIT)

    def slider_adjustment_perimeter(self):
        low, high = self._size_bounds_from_slider('slider_cell_count_perimeter_id', 'perimeter')
        gui_logger.slider('CELL_COUNT_PERIMETER_RANGE', _range_text(low, high, ''))
        self._take_filter('perimeter_um', low, high)
        self.ids['label_cell_count_perimeter_id'].text = _range_text(low, high, _PERIMETER_UNIT)

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def show_perimeter_range_dragged(self) -> None:
        """The perimeter label follows a drag; the bound is taken on release."""
        low, high = self._size_bounds_from_slider('slider_cell_count_perimeter_id', 'perimeter')
        self.ids['label_cell_count_perimeter_id'].text = _range_text(low, high, _PERIMETER_UNIT)

    def slider_adjustment_sphericity(self):
        lo = self.ids['slider_cell_count_sphericity_id'].value[0]
        hi = self.ids['slider_cell_count_sphericity_id'].value[1]
        gui_logger.slider('CELL_COUNT_SPHERICITY_RANGE', f'{lo}-{hi}')
        self._take_filter('sphericity', lo, hi)

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_min_intensity(self):
        lo = self.ids['slider_cell_count_min_intensity_id'].value[0]
        hi = self.ids['slider_cell_count_min_intensity_id'].value[1]
        gui_logger.slider('CELL_COUNT_MIN_INTENSITY_RANGE', f'{lo}-{hi}')
        self._take_filter('intensity_min_pct', lo, hi)

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_mean_intensity(self):
        lo = self.ids['slider_cell_count_mean_intensity_id'].value[0]
        hi = self.ids['slider_cell_count_mean_intensity_id'].value[1]
        gui_logger.slider('CELL_COUNT_MEAN_INTENSITY_RANGE', f'{lo}-{hi}')
        self._take_filter('intensity_mean_pct', lo, hi)

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def slider_adjustment_max_intensity(self):
        lo = self.ids['slider_cell_count_max_intensity_id'].value[0]
        hi = self.ids['slider_cell_count_max_intensity_id'].value[1]
        gui_logger.slider('CELL_COUNT_MAX_INTENSITY_RANGE', f'{lo}-{hi}')
        self._take_filter('intensity_max_pct', lo, hi)

        if self.ENABLE_PREVIEW_AUTO_REFRESH:
            self._regenerate_image_preview()

    def flourescent_mode_toggle(self):
        gui_logger.toggle(
            'CELL_COUNT_FLUORESCENT_MODE',
            bool(self.ids['cell_count_fluorescent_mode_id'].active),
        )
        self._method = dataclasses.replace(
            self._method, fluorescent_mode=bool(self.ids['cell_count_fluorescent_mode_id'].active)
        )

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
            self._method = post_processing.with_pixels_per_um(self._method, typed)

        run_reported(_apply, self._show_pixels_per_um, 'CELL_COUNT_PIXELS_PER_UM')

    def _show_pixels_per_um(self) -> None:
        box = self.ids['text_cell_count_pixels_per_um_id']
        held = _scale_text(self._method.pixels_per_um)
        if box.text != held:
            # The box is put back to what the method holds: the record pair
            # says what was typed and what the app kept instead. Assigning
            # .text does not dispatch the focus event the commit is bound to.
            gui_logger.text_input('CELL_COUNT_PIXELS_PER_UM_APPLIED', held)
            box.text = held
        if self._preview_image is not None:
            self._take_size_scale()
            self._show_size_filters()


# ============================================================================
# PostProcessingAccordion -- Post-Processing Panel Container
# ============================================================================


class PostProcessingAccordion(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = self.__class__.__name__
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

    def open_cell_count(self):
        gui_logger.button('OPEN_CELL_COUNT')
        ctx = _app_ctx.ctx
        if self._cell_count_popup is None:
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
            self._graphing_popup = Popup(
                title='Post Processing - Object Plotting',
                content=ctx.graphing_controls,
                size_hint=(0.85, 0.85),
                auto_dismiss=True,
            )
            ctx.graphing_controls.show()

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
