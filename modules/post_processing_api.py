# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""PostProcessingAPI -- the session's builds over a folder of captured images.

Stitch, z-projection, composite, video, Quick Enhance and cell count: each
member takes a folder and its build's own settings, reports progress through
an optional callback, and returns what the build made or raises the typed
outcome (``LiveFolderPathRefusedError`` when nothing is at the folder,
``'not_found'``, or a file is, ``'not_a_folder'``;
``PostProcessingRefusedError`` when the folder is there and cannot yield
it; ``PostProcessingFailedError`` when the build did not produce
everything).
Every caller -- the GUI, a script, REST -- reaches the same member.

The builds run on their own lane, never on the file lane a protocol run
writes its images through. A build of a large folder takes minutes; on the
run's lane it would hold up the run's writes and, past the stall threshold,
make the run judge its writer stuck. On this lane a run cannot see it.

A member reads nothing from the scope but what the installation and the
scope's capabilities say about the folder (its tiling config, whether the
scope has a turret), so where a build runs can change without changing its
callers.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from matplotlib.figure import Figure

from lvp_logger import logger
from modules import graph_analysis, live_work, path_utils
from modules.activity_claim import ActivityHolder, the_holder_named
from modules.cell_count_method import CellCountMethod
from modules.finite_number import is_finite_number
from modules.exceptions import (
    ExcludedInput,
    PostProcessingFailedError,
    PostProcessingOperation,
    PostProcessingRefusedError,
)
from modules.live_work import WorkItem
from modules.sequential_io_executor import IOTask
from modules.api_surface import FilePath, api, api_fields

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from modules.api_surface import ProgressCallback
    from modules.layer_record import ChannelColors
    from modules.sequential_io_executor import SequentialIOExecutor


@api_fields('path', 'algorithm', 'fallback_from', 'fallback_reason')
@dataclasses.dataclass(frozen=True)
class DegradedOutput:
    """A file a build made by its fallback algorithm, and why.

    Attributes:
        path: The file.
        algorithm: The algorithm that made it.
        fallback_from: The algorithm it fell back from.
        fallback_reason: Why the first could not make it.
    """

    path: pathlib.Path
    algorithm: str
    fallback_from: str
    fallback_reason: str


@api_fields(
    'message', 'new_count', 'output_root', 'artifact_paths', 'degraded_outputs', 'excluded_inputs'
)
@dataclasses.dataclass(frozen=True)
class BuildResult:
    """What a stitch, z-projection, composite or video build made.

    Attributes:
        message: The outcome in words, with what was skipped and why.
        new_count: How many files it made.
        output_root: The folder it read and wrote under.
        artifact_paths: Each file it made, where it landed.
        degraded_outputs: The files a fallback algorithm made; empty when
            none.
        excluded_inputs: The files of each kind in the folder the build
            did not take (a stitch's per-tile composites), not counting
            its own earlier outputs; empty when none. The message names
            them, with the route to their output where there is one.
    """

    message: str
    new_count: int
    output_root: pathlib.Path
    artifact_paths: tuple[pathlib.Path, ...]
    degraded_outputs: tuple[DegradedOutput, ...]
    excluded_inputs: tuple[ExcludedInput, ...]


@api_fields('source_path', 'output_path', 'recipe_path')
@dataclasses.dataclass(frozen=True)
class EnhancedFile:
    """One file Quick Enhance derived: its source, the derived image, and its recipe."""

    source_path: pathlib.Path
    output_path: pathlib.Path
    recipe_path: pathlib.Path


@api_fields('message', 'output_folder', 'created')
@dataclasses.dataclass(frozen=True)
class EnhanceResult:
    """What a Quick Enhance made.

    Attributes:
        message: The outcome in words.
        output_folder: The folder the derived files are in.
        created: Each derived file, with its source and recipe.
    """

    message: str
    output_folder: pathlib.Path
    created: tuple[EnhancedFile, ...]


@api_fields('message', 'results_path', 'counted')
@dataclasses.dataclass(frozen=True)
class CellCountResult:
    """What a cell count wrote.

    Attributes:
        message: The outcome in words.
        results_path: The results table it wrote.
        counted: How many images it counted.
    """

    message: str
    results_path: pathlib.Path
    counted: int


@api_fields('num_cells', 'total_object_area', 'area_unit', 'total_object_intensity')
@dataclasses.dataclass(frozen=True)
class CellCounts:
    """What a cell count measured in one image: the values of its row in ``results.csv``.

    Attributes:
        num_cells: The regions the method kept.
        total_object_area: Their area, in ``area_unit``.
        area_unit: ``'um2'``, or ``'px2'`` when the image was counted in
            pixels.
        total_object_intensity: Their area times their mean intensity.
    """

    num_cells: int
    total_object_area: float
    area_unit: str
    total_object_intensity: float


@api_fields('image', 'counts')
@dataclasses.dataclass(frozen=True)
class CellCountPreview:
    """One image counted by a method, as a form shows it.

    Attributes:
        image: The image with the kept regions' outlines drawn on it.
        counts: What the count measured.
    """

    image: np.ndarray
    counts: CellCounts


@api_fields('positions', 'sizes', 'reach')
@dataclasses.dataclass(frozen=True)
class SizeSliderStops:
    """What a size slider's positions stand for: between two stops, a size in proportion.

    Attributes:
        positions: The slider positions of the stops, ascending.
        sizes: The size each stop stands for, in microns (square microns
            for an area), or in pixels when the image has no scale.
        reach: The largest size the image can hold; the slider goes no
            further than its position.
    """

    positions: tuple[float, ...]
    sizes: tuple[float, ...]
    reach: float

    @api(in_process=True, reads=True)
    def size_at(self, positions: tuple[float, float]) -> tuple[float, float]:
        """The sizes a slider's two positions stand for.

        Raises:
            Nothing.
        """
        import numpy as np

        low, high = np.interp(positions, self.positions, self.sizes)
        return float(low), float(high)

    @api(in_process=True, reads=True)
    def positions_of(self, sizes: tuple[float, float]) -> tuple[float, float]:
        """The slider positions that show two sizes.

        Raises:
            Nothing.
        """
        import numpy as np

        low, high = np.interp(sizes, self.sizes, self.positions)
        return float(low), float(high)


@api_fields('pixels_per_um', 'area', 'perimeter')
@dataclasses.dataclass(frozen=True)
class CellCountSizeScale:
    """The scale an image is counted at, and what its size sliders stand for.

    Attributes:
        pixels_per_um: The pixels per micron the count measures at, or None
            when it counts in pixels.
        area: The area slider's stops.
        perimeter: The perimeter slider's stops.
    """

    pixels_per_um: float | None
    area: SizeSliderStops
    perimeter: SizeSliderStops


@api_fields('x_axes', 'y_axes', 'trendlines')
@dataclasses.dataclass(frozen=True)
class GraphOptions:
    """What a graph of a results table can be drawn with.

    Attributes:
        x_axes: The columns that can be the X axis: numbers and times.
        y_axes: The columns that can be the Y axis: numbers.
        trendlines: The trendline kinds the axes given take: every kind
            until an axis rules some out (a time X takes no Power or
            Logarithmic).
    """

    x_axes: tuple[str, ...]
    y_axes: tuple[str, ...]
    trendlines: tuple[str, ...]


@api_fields('message', 'path')
@dataclasses.dataclass(frozen=True)
class GraphResult:
    """What a graph save wrote.

    Attributes:
        message: The outcome in words.
        path: The PNG it wrote.
    """

    message: str
    path: pathlib.Path


@api_fields('method')
@dataclasses.dataclass(frozen=True)
class ZProject:
    """A z-projection, as ``build`` takes it: each Z-stack to one image.

    Attributes:
        method: One of ``PostProcessingAPI.projection_methods()``.
    """

    method: str = 'Max'


@api_fields('mode')
@dataclasses.dataclass(frozen=True)
class Stitch:
    """A stitch, as ``build`` takes it: each tiled scan to one mosaic.

    Attributes:
        mode: One of ``PostProcessingAPI.stitch_modes()``: ``'quality'`` or
            ``'fast_preview'``.
    """

    mode: str = 'quality'


@api_fields('frames_per_sec', 'timestamp_overlay')
@dataclasses.dataclass(frozen=True)
class Video:
    """A video, as ``build`` takes it: each position's time series to one video.

    Attributes:
        frames_per_sec: The playback rate, a number of at least 1 or its
            text; None plays a recording at its own measured rate.
        timestamp_overlay: Whether each frame shows its time.
    """

    frames_per_sec: float | str | None = None
    timestamp_overlay: bool = False


@api_fields('operation', 'result')
@dataclasses.dataclass(frozen=True)
class BuildStep:
    """One build ``build`` ran, and what it made.

    Attributes:
        operation: The build.
        result: What it made.
    """

    operation: PostProcessingOperation
    result: BuildResult


@api_fields('operation', 'message')
@dataclasses.dataclass(frozen=True)
class SkippedBuild:
    """A build ``build`` was asked for that had nothing to do in the folder.

    Attributes:
        operation: The build.
        message: Why, in the build's own words.
    """

    operation: PostProcessingOperation
    message: str


@api_fields('message', 'built', 'skipped')
@dataclasses.dataclass(frozen=True)
class BuildsResult:
    """What ``build`` made.

    Attributes:
        message: Each build's outcome in words, in the order run.
        built: Each build run and what it made, in the order run.
        skipped: Each build that had nothing to do, and why.
    """

    message: str
    built: tuple[BuildStep, ...]
    skipped: tuple[SkippedBuild, ...]


def _raw_rows(images: pd.DataFrame) -> pd.Series:
    """The raw captures among a folder's images: a raw capture's derivation is empty."""
    return images['Derivation'].astype(str) == ''


def _rows_among(files: frozenset[pathlib.Path], images: pd.DataFrame) -> pd.Series:
    """The images among *files*: what the build before carried forward."""
    return images['Filepath'].isin(files)


def _build_result(answer: dict) -> BuildResult:
    """The published record of a builder's answer."""
    root = pathlib.Path(answer['output_root'])
    return BuildResult(
        message=answer['message'],
        new_count=answer['new_count'],
        output_root=root,
        artifact_paths=tuple(pathlib.Path(path) for path in answer['artifact_paths']),
        degraded_outputs=tuple(
            DegradedOutput(
                path=root / degraded['filepath'],
                algorithm=degraded['algorithm'],
                fallback_from=degraded['fallback_from'],
                fallback_reason=degraded['fallback_reason'],
            )
            for degraded in answer.get('degraded_outputs', ())
        ),
        excluded_inputs=tuple(answer['excluded_inputs']),
    )


def _cell_count_result(answer: dict) -> CellCountResult:
    """The published record of a cell count's answer."""
    return CellCountResult(
        message=answer['message'],
        results_path=pathlib.Path(answer['results_path']),
        counted=answer['counted'],
    )


def _answering[R](record: Callable[[dict], R], build: Callable[..., dict]) -> Callable[..., R]:
    """``build``, answering ``record`` of what it returns.

    Wrapped so the lane still reads the slow-task budget the build declares.
    """

    @functools.wraps(build)
    def answer(*args, **kwargs) -> R:
        return record(build(*args, **kwargs))

    return answer


def _refuse_unusable_image_scale(pixel_size_um: float | None) -> None:
    """Refuse an image scale a count could not measure at: not a positive number of microns."""
    if pixel_size_um is not None and (not is_finite_number(pixel_size_um) or pixel_size_um <= 0):
        raise PostProcessingRefusedError(
            operation=PostProcessingOperation.CELL_COUNT,
            reason='invalid_setting',
            message=(
                'pixel_size_um must be a positive number of microns per pixel, or None for an '
                f'image that states no scale, not {pixel_size_um!r}.'
            ),
        )


def _existing_file_or_folder(path: pathlib.Path) -> pathlib.Path:
    """*path*, if a file or a folder is there: what an enhance reads."""
    return path if path.is_dir() else path_utils.existing_file(path)


def _results_table(results: FilePath):
    """The cell-count results file at *results*, read as the table a graph draws."""
    from modules.post_processing import read_cell_count_results

    return read_cell_count_results(path_utils.existing_file(results))


def _plottable_axes(table, x: str | None, y: str | None) -> tuple[list[str], list[str]]:
    """The columns of *table* each axis can take, once *x* and *y* are among them."""
    from modules.post_processing import results_axes

    x_axes, y_axes = results_axes(table)
    for axis, column, plottable in (('X', x, x_axes), ('Y', y, y_axes)):
        if column is not None and column not in plottable:
            raise PostProcessingRefusedError(
                operation=PostProcessingOperation.GRAPHING,
                reason='axis_not_plottable',
                message=f'{column} cannot be the {axis} axis: the {axis} axis can be '
                f'{", ".join(plottable) or "no column of this file"}.',
            )
    return x_axes, y_axes


def _figure(table, x, y, trendline, x_label, y_label, title) -> Figure:
    """The graph of *y* against *x* from *table*: the axes checked, the trendline fitted."""
    if table is None or x is None or y is None:
        return graph_analysis.draw_graph(
            table,
            None,
            None,
            None,
            x_label=x_label or '',
            y_label=y_label or '',
            title=title,
        )
    _plottable_axes(table, x, y)
    fitted = (
        None if trendline is None else graph_analysis.fit_trendline(trendline, table[x], table[y])
    )
    return graph_analysis.draw_graph(
        table,
        x,
        y,
        fitted,
        x_label=x if x_label is None else x_label,
        y_label=y if y_label is None else y_label,
        title=title,
    )


def _save_graph(results, path, *, x, y, trendline, x_label, y_label, title, on_progress):
    """Draw the graph of the *results* file and write it to *path*: ``graph``'s work on the lane."""
    figure = _figure(_results_table(results), x, y, trendline, x_label, y_label, title)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format='png')
    except OSError as e:
        raise PostProcessingFailedError(
            operation=PostProcessingOperation.GRAPHING,
            missing=f'The graph could not be written to {path} ({e.strerror or e}).',
        ) from e
    return GraphResult(message=f'Saved the graph of {y} against {x} to {path}.', path=path)


class PostProcessingAPI:
    """The post-processing builds, each run on the post-processing lane."""

    def __init__(
        self,
        *,
        lane: SequentialIOExecutor,
        tiling_configs_path: Callable[[], pathlib.Path],
        has_turret: Callable[[], bool],
        settings_snapshot: Callable[[], dict],
        channel_colors: Callable[[], ChannelColors],
        scope_holder: Callable[[], ActivityHolder | None],
        files_draining: Callable[[], bool],
    ):
        """
        Args:
            lane: The post-processing lane every build runs on.
            tiling_configs_path: The installation's tiling config, from its
                one owner.
            has_turret: Whether the scope has a turret, which decides how a
                folder's images are grouped by objective.
            settings_snapshot: The live user configuration, read when a
                build needs a user setting (the composite's output format
                and blend thresholds).
            channel_colors: Every layer's colour, from the Session's one
                read of it, read as each build is asked for.
            scope_holder: What holds the scope now -- a run, a recording, a
                diagnostic, a home -- or None; asked as each build starts.
            files_draining: Whether a finished run's files are still being
                written; asked as each build starts.
        """
        self.lane = lane
        self._tiling_configs_path = tiling_configs_path
        self._has_turret = has_turret
        self._settings_snapshot = settings_snapshot
        self._channel_colors = channel_colors
        self._scope_holder = scope_holder
        self._files_draining = files_draining
        # Each build submitted through _run, by its lane task, with its
        # progress as its own callback last said; kept from the submit until
        # the call returns, so the lane's running task is named by it.
        self._builds: dict[IOTask, WorkItem] = {}
        self._builds_lock = threading.Lock()

    def work(self) -> tuple[WorkItem, ...]:
        """The build the lane is running and how many wait behind it.

        A task the lane runs that no member submitted is a protocol's
        processors, which the run hands to the lane itself; it reports no
        progress.
        """
        items = []
        running = self.lane.running_task
        if running is not None:
            with self._builds_lock:
                build = self._builds.get(running)
            items.append(
                build
                if build is not None
                else WorkItem(live_work.POST_PROCESSING, "a protocol's post-processing")
            )
        queued = self.lane.queue_size()
        if queued:
            items.append(
                WorkItem(
                    live_work.POST_PROCESSING_QUEUED,
                    'post-processing waiting to start',
                    left=queued,
                )
            )
        return tuple(items)

    @api(reads=True)
    def stitch_modes(self) -> tuple[str, ...]:
        """The modes ``stitch(mode=)`` and ``Stitch(mode)`` take.

        Raises:
            Nothing.
        """
        from modules.stitcher import Stitcher

        return (Stitcher.QUALITY_MODE, Stitcher.FAST_PREVIEW_MODE)

    @api(reads=True)
    def projection_methods(self) -> tuple[str, ...]:
        """The methods ``zproject(method=)`` and ``ZProject(method)`` take.

        Raises:
            Nothing.
        """
        from modules.zprojector import ZProjector

        return tuple(ZProjector.methods())

    @api
    def stitch(
        self,
        folder: FilePath,
        *,
        mode: str = Stitch.mode,
        on_progress: ProgressCallback | None = None,
    ) -> BuildResult:
        """Stitch each tiled scan in *folder* into one mosaic per group.

        Args:
            mode: ``'quality'`` or ``'fast_preview'``.

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                *folder*; ``'not_a_folder'``, a file is.
            PostProcessingRefusedError: reason ``invalid_setting``, *mode*
                is not a stitching mode, before the build is queued; or a
                folder it cannot build from (``ProtocolPostProcessor``): reason
                ``protocol_data_unreadable``, ``no_images``,
                ``unsupported_source_format``, ``output_outside_folder``,
                ``collision``, ``excluded_inputs`` or ``no_data``.
            PostProcessingFailedError: some groups could not be built; the
                files that were are named.
        """
        return self._run(
            _answering(_build_result, self._stitch_build(Stitch(mode=mode))),
            'stitch',
            pathlib.Path(folder),
            operation=PostProcessingOperation.STITCH,
            on_progress=on_progress,
        )

    @api
    def zproject(
        self,
        folder: FilePath,
        *,
        method: str = ZProject.method,
        on_progress: ProgressCallback | None = None,
    ) -> BuildResult:
        """Project each Z-stack in *folder* to one image by *method*
        (one of ``projection_methods()``).

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                *folder*; ``'not_a_folder'``, a file is.
            PostProcessingRefusedError: reason ``invalid_setting``, *method*
                is not a projection method, before the build is queued; or a
                folder it cannot build from (``ProtocolPostProcessor``): reason
                ``protocol_data_unreadable``, ``no_images``,
                ``unsupported_source_format``, ``output_outside_folder``,
                ``collision``, ``excluded_inputs`` or ``no_data``.
            PostProcessingFailedError: some groups could not be built; the
                files that were are named.
        """
        return self._run(
            _answering(_build_result, self._zproject_build(ZProject(method=method))),
            'zproject',
            pathlib.Path(folder),
            operation=PostProcessingOperation.Z_PROJECTION,
            on_progress=on_progress,
        )

    @api
    def composite(
        self,
        folder: FilePath,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> BuildResult:
        """Merge each multi-channel position in *folder* into one composite.

        The output format and each channel's blend threshold are the user's
        configuration -- the same settings a run's own merge uses -- read
        when the build is asked for.

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                *folder*; ``'not_a_folder'``, a file is.
            PostProcessingRefusedError: a
                folder it cannot build from (``ProtocolPostProcessor``): reason
                ``protocol_data_unreadable``, ``no_images``,
                ``unsupported_source_format``, ``output_outside_folder``,
                ``collision``, ``excluded_inputs`` or ``no_data``.
            PostProcessingFailedError: some groups could not be built; the
                files that were are named.
        """
        return self._run(
            _answering(_build_result, self._composite_build()),
            'composite',
            pathlib.Path(folder),
            operation=PostProcessingOperation.COMPOSITE,
            on_progress=on_progress,
        )

    @api
    def video(
        self,
        folder: FilePath,
        *,
        frames_per_sec: float | str | None = Video.frames_per_sec,
        timestamp_overlay: bool = Video.timestamp_overlay,
        on_progress: ProgressCallback | None = None,
    ) -> BuildResult:
        """Build video(s) from *folder*: a protocol scan's time series, or a
        manual frames recording.

        Args:
            frames_per_sec: The playback rate, a number of at least 1 or its
                text; None plays a recording at its own measured rate.

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                *folder*; ``'not_a_folder'``, a file is.
            PostProcessingRefusedError: reason ``invalid_setting``,
                *frames_per_sec* is not a playback rate, before the build is
                queued; or a folder it cannot build from
                (``ProtocolPostProcessor``): reason
                ``protocol_data_unreadable``, ``no_images``,
                ``unsupported_source_format``, ``output_outside_folder``,
                ``collision``, ``excluded_inputs`` or ``no_data``.
            PostProcessingFailedError: some groups could not be built; the
                files that were are named.
        """
        build = self._video_build(
            Video(frames_per_sec=frames_per_sec, timestamp_overlay=timestamp_overlay)
        )
        return self._run(
            _answering(_build_result, build),
            'video',
            pathlib.Path(folder),
            operation=PostProcessingOperation.VIDEO,
            on_progress=on_progress,
        )

    @api
    def build(
        self,
        folder: FilePath,
        *,
        zproject: ZProject | None = None,
        stitch: Stitch | None = None,
        composite: bool = False,
        video: Video | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> BuildsResult:
        """Run the chosen builds over *folder* in one call, in the order that
        keeps the data honest: z-project, stitch, composite, video.

        A projection is made of raw slices, never of mosaics; a stitched
        composite is made from the stitched channels, never by stitching
        composites. Each build takes only what the build before it in this
        call carried forward -- what that build made, and the images it did
        not consume (an ordinary Add step's image beside a tiled stack) --
        so the call makes the chain's product and its inputs, not every
        combination the folder's images allow: z-project, stitch and
        composite of a 2x2 tiled, 3-slice, 2-channel scan make 8
        projections, 2 stitched projections and 1 composite, where the three
        builds run one by one make 36 files. A build is chosen by passing its
        settings record (*composite*, which has no settings of its own -- its
        format and thresholds are the user's configuration -- by True); one
        left None is not run. Every chosen build's
        settings are checked before any build starts, and the call runs as
        one task on the post-processing lane, so no other build touches the
        folder between its steps. Each build alone (``stitch``,
        ``zproject``, ``composite``, ``video``) is unchanged.

        A build with nothing to do in the folder -- video on a single-scan
        run -- is skipped and named in the answer, and the next build takes
        what it was given. Any other refusal or failure stops the call and is
        raised as that build's own outcome, unchanged; the files the earlier
        builds made are in the folder and its record, and the same call made
        again rebuilds nothing already made and resumes at the build that
        stopped, once its cause is gone. One that does not go -- a raw slice
        missing -- stops the call there every time; build that folder with
        the members one by one.

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                *folder*; ``'not_a_folder'``, a file is.
            PostProcessingRefusedError: reason ``invalid_setting``, no build
                is chosen, or a chosen build's setting is one its own member
                refuses, before anything is queued; or the refusal of the
                build that stopped the call (its ``operation`` names it).
            PostProcessingFailedError: the build that stopped the call did not
                produce everything; its ``operation`` names it.
        """
        # Each chosen build made now, so every one's settings are checked
        # before any is queued.
        made = (
            (PostProcessingOperation.Z_PROJECTION, zproject and self._zproject_build(zproject)),
            (PostProcessingOperation.STITCH, stitch and self._stitch_build(stitch)),
            (PostProcessingOperation.COMPOSITE, composite and self._composite_build()),
            (PostProcessingOperation.VIDEO, video and self._video_build(video)),
        )
        chosen = [(operation, run) for operation, run in made if run]
        if not chosen:
            raise PostProcessingRefusedError(
                operation=PostProcessingOperation.POST_PROCESSING,
                reason='invalid_setting',
                message='No build was chosen: pass the settings of at least one build.',
            )
        return self._run(
            functools.partial(self._chain, folder=pathlib.Path(folder), chosen=chosen),
            'build',
            pathlib.Path(folder),
            operation=PostProcessingOperation.POST_PROCESSING,
            on_progress=on_progress,
        )

    def _chain(
        self,
        path: pathlib.Path,
        *,
        folder: pathlib.Path,
        chosen: list[tuple[PostProcessingOperation, Callable[..., dict]]],
        on_progress: ProgressCallback,
    ) -> BuildsResult:
        """The chosen builds, in order, on the lane task ``build`` submitted."""
        built: list[BuildStep] = []
        skipped: list[SkippedBuild] = []
        # The first build takes the raw captures: a derived image already in
        # the folder is no input of this chain's.
        takes: Callable = _raw_rows
        for index, (operation, run) in enumerate(chosen):
            stage = f'Building {index + 1} of {len(chosen)}: {operation}'
            self._name_running_build(f'{stage}, of {folder}')

            def stage_progress(percent: float, detail: str | None = None, index=index, stage=stage):
                said = stage if detail is None else f'{stage}\n{detail}'
                on_progress((index + percent / 100) / len(chosen) * 100, said)

            try:
                answer = run(path, select=takes, on_progress=stage_progress)
            except PostProcessingRefusedError as refused:
                if refused.reason != 'no_data':
                    raise
                skipped.append(SkippedBuild(operation=operation, message=str(refused)))
                continue
            built.append(BuildStep(operation=operation, result=_build_result(answer)))
            if index + 1 < len(chosen):
                # Only a build another follows hands on what it carries: a
                # video is last, and one of a manual recording reads no
                # protocol folder to carry anything from.
                takes = functools.partial(_rows_among, answer['carried_forward'])
        said = [f'{step.operation}: {step.result.message}' for step in built]
        said += [f'{step.operation}: skipped -- {step.message}' for step in skipped]
        return BuildsResult(message=' '.join(said), built=tuple(built), skipped=tuple(skipped))

    def _name_running_build(self, name: str) -> None:
        """Rename the build the lane is running, for ``work`` to say."""
        with self._builds_lock:
            running = self.lane.running_task
            if running in self._builds:
                self._builds[running] = dataclasses.replace(self._builds[running], name=name)

    def _stitch_build(self, settings: Stitch) -> Callable[..., dict]:
        """The stitch of *settings*, its mode checked before anything is queued."""
        from modules.stitcher import Stitcher

        modes = self.stitch_modes()
        if settings.mode not in modes:
            raise PostProcessingRefusedError(
                operation=PostProcessingOperation.STITCH,
                reason='invalid_setting',
                message=f'{settings.mode!r} is not a stitching mode; use one of {", ".join(modes)}.',
            )
        return functools.partial(
            Stitcher(has_turret=self._has_turret()).load_folder,
            tiling_configs_file_loc=self._tiling_configs_path(),
            stitching_mode=settings.mode,
            channel_colors=self._channel_colors(),
        )

    def _zproject_build(self, settings: ZProject) -> Callable[..., dict]:
        """The z-projection of *settings*, its method checked before anything is queued."""
        from modules.zprojector import ZProjector

        methods = self.projection_methods()
        if settings.method not in methods:
            raise PostProcessingRefusedError(
                operation=PostProcessingOperation.Z_PROJECTION,
                reason='invalid_setting',
                message=(
                    f'{settings.method!r} is not a projection method; use one of '
                    f'{", ".join(methods)}.'
                ),
            )
        return functools.partial(
            ZProjector(has_turret=self._has_turret()).load_folder,
            tiling_configs_file_loc=self._tiling_configs_path(),
            method=settings.method,
            channel_colors=self._channel_colors(),
        )

    def _composite_build(self) -> Callable[..., dict]:
        """The composite, its format and thresholds read from the live settings now.

        Nothing is checked here: the stored values are kept in range where
        they are written and loaded.
        """
        import modules.config_helpers as config_helpers
        from modules.composite_generation import CompositeGeneration

        live = self._settings_snapshot()
        return functools.partial(
            CompositeGeneration(has_turret=self._has_turret()).load_folder,
            tiling_configs_file_loc=self._tiling_configs_path(),
            output_format=live['image_output_format']['sequenced'],
            brightness_thresholds_percent=config_helpers.get_composite_blend_thresholds(live),
            channel_colors=self._channel_colors(),
        )

    def _video_build(self, settings: Video) -> Callable[..., dict]:
        """The video of *settings*, its rate checked before anything is queued."""
        from modules.video_builder import VideoBuilder

        return functools.partial(
            VideoBuilder(has_turret=self._has_turret()).build_from_folder,
            tiling_configs_file_loc=self._tiling_configs_path(),
            frames_per_sec=self._playback_rate(settings.frames_per_sec),
            enable_timestamp_overlay=settings.timestamp_overlay,
            channel_colors=self._channel_colors(),
        )

    @api
    def enhance(
        self,
        target: FilePath,
        *,
        on_progress: ProgressCallback | None = None,
        on_derived_image: Callable[[np.ndarray, int], None] | None = None,
    ) -> EnhanceResult:
        """Quick Enhance *target* -- one image, or every image in a folder --
        each to its own derived file beside its source.

        Args:
            on_derived_image: Called with each enhanced image and its
                significant bits as it is saved, for a caller that shows it.

        Returns:
            Each derived file, the folder they are in, and the outcome in
            words.

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                *target*.
            PostProcessingRefusedError: reason ``unreadable``, *target* is
                a file that is not an image Quick Enhance can read; reason
                ``no_data``, a folder that holds none.
            PostProcessingFailedError: a file could not be enhanced.
        """
        return self._run(
            self._enhance,
            'enhance',
            pathlib.Path(target),
            operation=PostProcessingOperation.QUICK_ENHANCE,
            reads=_existing_file_or_folder,
            on_progress=on_progress,
            on_derived_image=on_derived_image,
            channel_colors=self._channel_colors(),
        )

    @api(reads=True)
    def default_cell_count_method(self) -> CellCountMethod:
        """The cell-count method LumaViewPro's panel starts from, for
        ``count_cells(method=)``: ``CellCountMethod()``.

        Raises:
            Nothing.
        """
        return CellCountMethod()

    @api
    def load_cell_count_method(self, path: FilePath) -> CellCountMethod:
        """The cell-count method saved in the file at *path*.

        Read on the caller's thread, not the post-processing lane, so a
        form taking a method does not wait behind a build. A file saved
        before the count read each image's scale carries the old fixed
        default of 1.0 pixels per micron, which the load drops, saying so
        once.

        Raises:
            PostProcessingRefusedError: reason ``method_unreadable``, the
                file cannot be read or is not a method file; reason
                ``method_invalid``, it holds a method the count cannot use.
                The message names the file.
        """
        from modules.post_processing import load_cell_count_method

        return load_cell_count_method(path)

    @api
    def save_cell_count_method(self, method: CellCountMethod, path: FilePath) -> None:
        """Write *method* to *path* as a cell-count method file, which
        ``load_cell_count_method`` reads back.

        Written on the caller's thread, as the load is read. A missing
        folder is made where *path* sends it.

        Raises:
            LiveFolderPathRefusedError: ``'not_a_file'``, a folder is at
                *path*; ``'wrong_extension'``, its name does not end in
                ``.json`` (in any case), which is not added; ``'not_a_folder'``,
                a file is where its folder would be. Nothing is written.
            PostProcessingRefusedError: reason ``method_invalid``, a
                *method* sent as its fields holds what ``CellCountMethod``
                refuses. Nothing is written.
        """
        from modules.post_processing import save_cell_count_method

        save_cell_count_method(method, path)

    @api
    def cell_count_size_scale(
        self,
        method: CellCountMethod,
        *,
        shape: tuple[int, ...] | None = None,
        pixel_size_um: float | None = None,
    ) -> CellCountSizeScale:
        """The scale an image of *shape* is counted at by *method*, and what its size sliders stand for.

        The scale is the one ``preview_cell_count`` and ``count_cells``
        measure at: the method's *pixels_per_um*, else *pixel_size_um*, the
        image's own (microns per pixel; None when it states none), else
        pixels. Each slider's top stop is the largest size the image holds
        at that scale, and its low end is stretched, so the small sizes most
        methods bound get most of the travel. With *shape* None (no image
        yet) a position stands for its own size.

        Raises:
            PostProcessingRefusedError: reason ``invalid_setting``, *shape*
                is not an image's rows and columns, each at least one, or
                *pixel_size_um* is not a positive number of microns.
        """
        from modules.post_processing import cell_count_scale, size_slider_stops

        if shape is not None and (
            len(shape) not in (2, 3) or not all(type(side) is int and side >= 1 for side in shape)
        ):
            raise PostProcessingRefusedError(
                operation=PostProcessingOperation.CELL_COUNT,
                reason='invalid_setting',
                message=(
                    "shape must be an image's rows and columns (and channels), each at "
                    f'least 1, or None for no image, not {shape!r}.'
                ),
            )
        _refuse_unusable_image_scale(pixel_size_um)
        pixels_per_um = cell_count_scale(method, pixel_size_um)

        def stops(size: str) -> SizeSliderStops:
            if shape is None:
                return SizeSliderStops(positions=(0.0, 100.0), sizes=(0.0, 100.0), reach=100.0)
            positions, sizes, reach = size_slider_stops(size, shape, pixels_per_um)
            return SizeSliderStops(positions=positions, sizes=sizes, reach=reach)

        return CellCountSizeScale(
            pixels_per_um=pixels_per_um, area=stops('area'), perimeter=stops('perimeter')
        )

    @api(in_process=True)
    def preview_cell_count(
        self,
        image: np.ndarray,
        method: CellCountMethod,
        *,
        significant_bits: int,
        pixel_size_um: float | None,
    ) -> CellCountPreview:
        """Count the cells in *image* by *method*, as ``count_cells`` counts
        each image of a folder, and answer the outlines and the counts.

        Counted on the caller's thread, not the post-processing lane, so a
        form previewing a method while a build or a run goes on is not held
        behind it. *image* is the pixels the caller holds, so a form that
        adjusts the method reads no file. It is measured at the scale
        ``count_cells`` would measure its file at: the method's
        *pixels_per_um*, else *pixel_size_um*, the image's own (microns
        per pixel), else in pixels.

        Args:
            significant_bits: The image's payload depth, by which a 16-bit
                image is scaled to the 8 bits the count reads.
            pixel_size_um: The scale the image states, or None.

        Raises:
            PostProcessingRefusedError: reason ``no_scale``, the image is
                counted in pixels and *method* bounds an area or a
                perimeter, which are in microns; ``invalid_setting``,
                *pixel_size_um* is not a positive number of microns.
        """
        from modules.post_processing import PostProcessing, cell_count_scale

        _refuse_unusable_image_scale(pixel_size_um)
        outlined, stats = PostProcessing().preview_cell_count(
            image,
            method,
            significant_bits,
            pixels_per_um=cell_count_scale(method, pixel_size_um),
        )
        summary = stats['summary']
        return CellCountPreview(
            image=outlined,
            counts=CellCounts(
                num_cells=summary['num_regions'],
                total_object_area=summary['total_object_area'],
                area_unit=summary['area_unit'],
                total_object_intensity=summary['total_object_intensity'],
            ),
        )

    @api
    def count_cells(
        self,
        folder: FilePath,
        *,
        method: CellCountMethod,
        on_progress: ProgressCallback | None = None,
    ) -> CellCountResult:
        """Count the cells in every image in *folder* by *method*, writing
        ``results.csv`` into the folder.

        Returns:
            The results table, how many images were counted, and the
            outcome in words.

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                *folder*; ``'not_a_folder'``, a file is.
            PostProcessingRefusedError: reason ``method_invalid``, a
                *method* sent as its fields holds what ``CellCountMethod``
                refuses; or the folder has nothing to count (reason
                ``no_data``).
            PostProcessingFailedError: the results could not be written, or
                some images could not be counted; the file holds every image
                that was, and the error names the rest.
        """
        from modules.post_processing import PostProcessing

        return self._run(
            _answering(_cell_count_result, PostProcessing().apply_cell_count_to_folder),
            'count_cells',
            folder,
            method,
            operation=PostProcessingOperation.CELL_COUNT,
            on_progress=on_progress,
        )

    @api
    def graph_options(
        self, results: FilePath, *, x: str | None = None, y: str | None = None
    ) -> GraphOptions:
        """The columns of the cell-count *results* file a graph can take, and
        the trendlines the axes given so far take.

        Read on the caller's thread, not the post-processing lane, so a form
        filling its choices does not wait behind a build.

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                *results*; ``'not_a_file'``, a folder is.
            PostProcessingRefusedError: ``'results_unreadable'``, the file
                cannot be graphed; ``'axis_not_plottable'``, *x* or *y* is
                not a column that axis can take.
        """
        table = _results_table(results)
        x_axes, y_axes = _plottable_axes(table, x, y)
        trendlines = graph_analysis.trendline_kinds(
            None if x is None else table[x], None if y is None else table[y]
        )
        return GraphOptions(x_axes=tuple(x_axes), y_axes=tuple(y_axes), trendlines=trendlines)

    @api(in_process=True)
    def draw_graph(
        self,
        results: FilePath | None,
        *,
        x: str | None,
        y: str | None,
        trendline: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        title: str = '',
    ) -> Figure:
        """The figure ``graph`` saves for the same arguments, for a GUI in this process to show.

        With no *results*, or *x* or *y* None, it is the empty frame a
        graph shows before its axes are chosen. Each label defaults to its
        column's name.

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                *results*; ``'not_a_file'``, a folder is.
            PostProcessingRefusedError: ``'results_unreadable'``, the file
                cannot be graphed; ``'axis_not_plottable'``, *x* or *y* is
                not a column that axis can take; ``'fit_impossible'``, the
                values cannot give the *trendline*.
        """
        table = None if results is None else _results_table(results)
        return _figure(table, x, y, trendline, x_label, y_label, title)

    @api
    def graph(
        self,
        results: FilePath | None,
        path: FilePath,
        *,
        x: str | None,
        y: str | None,
        trendline: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        title: str = '',
    ) -> GraphResult:
        """Save the graph of column *y* against column *x* of the cell-count
        *results* file to *path* as a PNG.

        *trendline* is one of the kinds ``graph_options`` names, or None for
        none. Each label defaults to its column's name. A missing folder is
        made where *path* sends it.

        Returns:
            The PNG written and the outcome in words.

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                *results*; ``'not_a_file'``, a folder is there or at *path*;
                ``'wrong_extension'``, *path* does not end in ``.png`` (in
                any case), which is not added; ``'not_a_folder'``, a file is
                where its folder would be. Nothing is written.
            PostProcessingRefusedError: ``'graph_incomplete'``, no *results*
                file or no *x* or *y* is given; ``'results_unreadable'``,
                the file cannot be graphed; ``'axis_not_plottable'``, *x* or
                *y* is not a column that axis can take; ``'fit_impossible'``,
                the values cannot give the *trendline*. Nothing is written.
            PostProcessingFailedError: the folder or the file could not be
                written; the message names the file and the system's reason.
        """
        if results is None or x is None or y is None:
            raise PostProcessingRefusedError(
                operation=PostProcessingOperation.GRAPHING,
                reason='graph_incomplete',
                message='There is nothing to graph yet: choose a results file and an X and a Y '
                'axis first.',
            )
        path = path_utils.file_to_write(path, extension='.png')
        return self._run(
            _save_graph,
            'graph',
            results,
            path,
            operation=PostProcessingOperation.GRAPHING,
            x=x,
            y=y,
            trendline=trendline,
            x_label=x_label,
            y_label=y_label,
            title=title,
            reads=path_utils.existing_file,
        )

    def _refuse_while_busy(self, operation: PostProcessingOperation) -> None:
        """Refuse a build that would start while the scope's own work needs the host.

        Raises:
            PostProcessingRefusedError: ``'scope_held'``, naming what holds
                the scope; ``'files_writing'``, a finished run's files are
                still being written.
        """
        holder = self._scope_holder()
        if holder is not None:
            raise PostProcessingRefusedError(
                operation=operation,
                reason='scope_held',
                message=(
                    f'{the_holder_named(holder)} is using the microscope, and a build now '
                    'would slow it down. Try again when it ends.'
                ),
            )
        if self._files_draining():
            raise PostProcessingRefusedError(
                operation=operation,
                reason='files_writing',
                message=(
                    "A protocol's files are still being written, and a build now would slow "
                    'the writing down. Try again when they are written.'
                ),
            )

    def _run(
        self,
        action,
        member: str,
        folder,
        *args,
        operation: PostProcessingOperation,
        reads=path_utils.existing_folder,
        **kwargs,
    ):
        """Run *action* on the post-processing lane and return its answer.

        The build says on the lane when it starts and how it ends, so a
        folder's derived files can be traced to the build that wrote them
        and its overlap with a run read back from the log. A build that
        raises is named by its outcome's type only: the reporter where the
        outcome's flight stops logs and shows it once.
        *folder* is the build's first argument, the folder (or, for an
        enhance, the file) it reads. Its progress reaches ``work`` through
        the ``on_progress`` it is handed, the caller's own still called.
        *reads* refuses a *folder* that is not there to read; it is asked
        on the lane, where the build reads it, so the answer cannot go
        stale while an earlier build runs. So is the scope's state: a build
        does not start while something holds the scope or a run's files are
        still being written, since it would share the cores, the
        interpreter lock and the disk with that work. A build already
        running when a run starts runs on.
        """
        caller_progress = kwargs.get('on_progress')

        def on_progress(percent: float, detail: str | None = None) -> None:
            with self._builds_lock:
                if task in self._builds:
                    self._builds[task] = dataclasses.replace(self._builds[task], percent=percent)
            if caller_progress is not None:
                caller_progress(percent, detail)

        @functools.wraps(action)
        def build(path, *a, **kw):
            self._refuse_while_busy(operation)
            logger.info(f'[PostProc  ] {member} started on {folder}')
            started = time.monotonic()
            try:
                result = action(reads(path), *a, **kw)
            except BaseException as outcome:
                logger.info(
                    f'[PostProc  ] {member} ended after {time.monotonic() - started:.1f} s: '
                    f'{type(outcome).__name__}'
                )
                raise
            logger.info(
                f'[PostProc  ] {member} ended after {time.monotonic() - started:.1f} s: '
                f'{result.message}'
            )
            return result

        task = IOTask(
            action=build, args=(folder, *args), kwargs={**kwargs, 'on_progress': on_progress}
        )
        with self._builds_lock:
            self._builds[task] = WorkItem(live_work.POST_PROCESSING, f'{member} of {folder}')
        try:
            return self.lane.call(task, f'post_processing.{member}', None)
        finally:
            with self._builds_lock:
                del self._builds[task]

    @staticmethod
    def _playback_rate(frames_per_sec: float | str | None) -> float | None:
        if frames_per_sec is None:
            return None
        try:
            rate = float(frames_per_sec)
        except (TypeError, ValueError):
            rate = None
        if rate is None or not rate >= 1:
            raise PostProcessingRefusedError(
                operation=PostProcessingOperation.VIDEO,
                reason='invalid_setting',
                message=(
                    f'{frames_per_sec!r} is not a playback rate: give at least 1 frame '
                    "per second, or nothing for the recording's own rate."
                ),
            )
        return rate

    @staticmethod
    def _enhance(
        target: pathlib.Path,
        on_progress: ProgressCallback | None,
        on_derived_image: Callable[[np.ndarray, int], None] | None,
        channel_colors: ChannelColors,
    ) -> EnhanceResult:
        import cv2

        from modules import image_utils
        from modules.quick_enhance import (
            QUICK_ENHANCE_OPERATION,
            QuickEnhancer,
            QuickEnhanceSettings,
        )

        enhancer = QuickEnhancer()
        settings = QuickEnhanceSettings()
        if target.is_dir():

            def _progress(done: int, total: int, path: pathlib.Path) -> None:
                if on_progress is not None:
                    on_progress(
                        100 if total == 0 else 100 * done / total, f'Image {done} of {total}'
                    )

            result = enhancer.export_folder(
                target,
                settings,
                progress_callback=_progress,
                display_callback=on_derived_image,
                channel_colors=channel_colors,
            )
            created = result['created']
        else:
            # By the test a folder's images are chosen by, so one file and a
            # folder of them are refused for the same thing.
            if not image_utils.is_image(target):
                raise PostProcessingRefusedError(
                    operation=QUICK_ENHANCE_OPERATION,
                    reason='unreadable',
                    message=f'{target.name} is not an image Quick Enhance can read.',
                )
            try:
                created = [
                    enhancer.export_file(
                        target,
                        settings,
                        display_callback=on_derived_image,
                        channel_colors=channel_colors,
                    )
                ]
            except (OSError, ValueError, cv2.error, MemoryError) as e:
                raise PostProcessingFailedError(
                    operation=QUICK_ENHANCE_OPERATION,
                    missing=f'{target.name} could not be enhanced.',
                    errors=[str(e)],
                ) from e
            if on_progress is not None:
                on_progress(100, 'Image 1 of 1')
        output_folder = QuickEnhancer.output_folder({'created': created})
        return EnhanceResult(
            # The derived files sit beside their sources, so the words do not
            # repeat a path the person just chose.
            message='Enhance complete.',
            output_folder=output_folder,
            created=tuple(
                EnhancedFile(
                    source_path=pathlib.Path(item['source_path']),
                    output_path=pathlib.Path(item['output_path']),
                    recipe_path=pathlib.Path(item['recipe_path']),
                )
                for item in created
            ),
        )
