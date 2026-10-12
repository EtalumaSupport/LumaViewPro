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

from lvp_logger import logger
from modules import live_work, path_utils
from modules.exceptions import (
    PostProcessingFailedError,
    PostProcessingOperation,
    PostProcessingRefusedError,
)
from modules.live_work import WorkItem
from modules.sequential_io_executor import IOTask
from modules.api_surface import FilePath, api, api_fields

if TYPE_CHECKING:
    import numpy as np

    from modules.api_surface import ProgressCallback
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


@api_fields('message', 'new_count', 'output_root', 'artifact_paths', 'degraded_outputs')
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
    """

    message: str
    new_count: int
    output_root: pathlib.Path
    artifact_paths: tuple[pathlib.Path, ...]
    degraded_outputs: tuple[DegradedOutput, ...]


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


def _existing_file_or_folder(path: pathlib.Path) -> pathlib.Path:
    """*path*, if a file or a folder is there: what an enhance reads."""
    return path if path.is_dir() else path_utils.existing_file(path)


class PostProcessingAPI:
    """The post-processing builds, each run on the post-processing lane."""

    def __init__(
        self,
        *,
        lane: SequentialIOExecutor,
        tiling_configs_path: Callable[[], pathlib.Path],
        has_turret: Callable[[], bool],
        settings_snapshot: Callable[[], dict],
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
        """
        self.lane = lane
        self._tiling_configs_path = tiling_configs_path
        self._has_turret = has_turret
        self._settings_snapshot = settings_snapshot
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

    @api
    def stitch(
        self,
        folder: FilePath,
        *,
        mode: str = 'quality',
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
        from modules.stitcher import Stitcher

        modes = (Stitcher.QUALITY_MODE, Stitcher.FAST_PREVIEW_MODE)
        if mode not in modes:
            raise PostProcessingRefusedError(
                operation=PostProcessingOperation.STITCH,
                reason='invalid_setting',
                message=f'{mode!r} is not a stitching mode; use one of {", ".join(modes)}.',
            )
        stitcher = Stitcher(has_turret=self._has_turret())
        return self._run(
            _answering(_build_result, stitcher.load_folder),
            'stitch',
            pathlib.Path(folder),
            tiling_configs_file_loc=self._tiling_configs_path(),
            on_progress=on_progress,
            stitching_mode=mode,
        )

    @api
    def zproject(
        self,
        folder: FilePath,
        *,
        method: str,
        on_progress: ProgressCallback | None = None,
    ) -> BuildResult:
        """Project each Z-stack in *folder* to one image by *method*
        (one of ``ZProjector.methods()``).

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
        from modules.zprojector import ZProjector

        if method not in ZProjector.methods():
            raise PostProcessingRefusedError(
                operation=PostProcessingOperation.Z_PROJECTION,
                reason='invalid_setting',
                message=(
                    f'{method!r} is not a projection method; use one of '
                    f'{", ".join(ZProjector.methods())}.'
                ),
            )
        zprojector = ZProjector(has_turret=self._has_turret())
        return self._run(
            _answering(_build_result, zprojector.load_folder),
            'zproject',
            pathlib.Path(folder),
            tiling_configs_file_loc=self._tiling_configs_path(),
            on_progress=on_progress,
            method=method,
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
        import modules.config_helpers as config_helpers
        from modules.composite_generation import CompositeGeneration

        settings = self._settings_snapshot()
        composite_gen = CompositeGeneration(has_turret=self._has_turret())
        return self._run(
            _answering(_build_result, composite_gen.load_folder),
            'composite',
            pathlib.Path(folder),
            tiling_configs_file_loc=self._tiling_configs_path(),
            on_progress=on_progress,
            output_format=settings['image_output_format']['sequenced'],
            brightness_thresholds_percent=config_helpers.get_composite_blend_thresholds(settings),
        )

    @api
    def video(
        self,
        folder: FilePath,
        *,
        frames_per_sec: float | str | None = None,
        timestamp_overlay: bool = False,
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
        from modules.video_builder import VideoBuilder

        rate = self._playback_rate(frames_per_sec)
        video_builder = VideoBuilder(has_turret=self._has_turret())
        return self._run(
            _answering(_build_result, video_builder.build_from_folder),
            'video',
            pathlib.Path(folder),
            tiling_configs_file_loc=self._tiling_configs_path(),
            on_progress=on_progress,
            frames_per_sec=rate,
            enable_timestamp_overlay=timestamp_overlay,
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
            reads=_existing_file_or_folder,
            on_progress=on_progress,
            on_derived_image=on_derived_image,
        )

    @api
    def count_cells(
        self,
        folder: FilePath,
        *,
        method: dict,
        on_progress: ProgressCallback | None = None,
    ) -> CellCountResult:
        """Count the cells in every image in *folder* by the cell-count
        *method*, writing ``results.csv`` into the folder.

        Returns:
            The results table, how many images were counted, and the
            outcome in words.

        Raises:
            LiveFolderPathRefusedError: ``'not_found'``, nothing is at
                *folder*; ``'not_a_folder'``, a file is.
            PostProcessingRefusedError: the method cannot be used (reason
                ``method_invalid``), before the count is queued; or the
                folder has nothing to count (reason ``no_data``).
        """
        from modules.post_processing import PostProcessing, check_cell_count_method

        check_cell_count_method(method)
        return self._run(
            _answering(_cell_count_result, PostProcessing().apply_cell_count_to_folder),
            'count_cells',
            folder,
            method,
            on_progress=on_progress,
        )

    def _run(self, action, member: str, folder, *args, reads=path_utils.existing_folder, **kwargs):
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
        stale while an earlier build runs.
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
                    enhancer.export_file(target, settings, display_callback=on_derived_image)
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
