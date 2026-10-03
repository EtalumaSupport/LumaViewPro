# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""PostProcessingAPI -- the session's builds over a folder of captured images.

Stitch, z-projection, composite, video, Quick Enhance and cell count: each
member takes a folder and its build's own settings, reports progress through
an optional callback, and returns what the build made or raises the typed
outcome (``PostProcessingRefusedError`` when the folder cannot yield it,
``PostProcessingFailedError`` when the build did not produce everything).
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

import functools
import pathlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from lvp_logger import logger
from modules.exceptions import PostProcessingFailedError, PostProcessingRefusedError
from modules.sequential_io_executor import IOTask

if TYPE_CHECKING:
    import numpy as np

    from modules.protocol_post_processor import ProgressCallback
    from modules.sequential_io_executor import SequentialIOExecutor


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

    def stitch(
        self,
        folder: str | pathlib.Path,
        *,
        mode: str = 'quality',
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Stitch each tiled scan in *folder* into one mosaic per group.

        Args:
            mode: ``'quality'`` or ``'fast_preview'``.
        """
        from modules.stitcher import Stitcher

        modes = (Stitcher.QUALITY_MODE, Stitcher.FAST_PREVIEW_MODE)
        if mode not in modes:
            raise PostProcessingRefusedError(
                operation='Stitch',
                reason='invalid_setting',
                message=f'{mode!r} is not a stitching mode; use one of {", ".join(modes)}.',
            )
        stitcher = Stitcher(has_turret=self._has_turret())
        return self._run(
            stitcher.load_folder,
            'stitch',
            pathlib.Path(folder),
            tiling_configs_file_loc=self._tiling_configs_path(),
            on_progress=on_progress,
            stitching_mode=mode,
        )

    def zproject(
        self,
        folder: str | pathlib.Path,
        *,
        method: str,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Project each Z-stack in *folder* to one image by *method*
        (one of ``ZProjector.methods()``)."""
        from modules.zprojector import ZProjector

        if method not in ZProjector.methods():
            raise PostProcessingRefusedError(
                operation='Z-Projection',
                reason='invalid_setting',
                message=(
                    f'{method!r} is not a projection method; use one of '
                    f'{", ".join(ZProjector.methods())}.'
                ),
            )
        zprojector = ZProjector(has_turret=self._has_turret())
        return self._run(
            zprojector.load_folder,
            'zproject',
            pathlib.Path(folder),
            tiling_configs_file_loc=self._tiling_configs_path(),
            on_progress=on_progress,
            method=method,
        )

    def composite(
        self,
        folder: str | pathlib.Path,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Merge each multi-channel position in *folder* into one composite.

        The output format and each channel's blend threshold are the user's
        configuration -- the same settings a run's own merge uses -- read
        when the build is asked for.
        """
        import modules.config_helpers as config_helpers
        from modules.composite_generation import CompositeGeneration

        settings = self._settings_snapshot()
        composite_gen = CompositeGeneration(has_turret=self._has_turret())
        return self._run(
            composite_gen.load_folder,
            'composite',
            pathlib.Path(folder),
            tiling_configs_file_loc=self._tiling_configs_path(),
            on_progress=on_progress,
            output_format=settings['image_output_format']['sequenced'],
            brightness_thresholds_percent=config_helpers.get_composite_blend_thresholds(settings),
        )

    def video(
        self,
        folder: str | pathlib.Path,
        *,
        frames_per_sec: float | str | None = None,
        timestamp_overlay: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Build video(s) from *folder*: a protocol scan's time series, or a
        manual frames recording.

        Args:
            frames_per_sec: The playback rate, a number of at least 1 or its
                text; None plays a recording at its own measured rate.
        """
        from modules.video_builder import VideoBuilder

        rate = self._playback_rate(frames_per_sec)
        video_builder = VideoBuilder(has_turret=self._has_turret())
        return self._run(
            video_builder.build_from_folder,
            'video',
            pathlib.Path(folder),
            tiling_configs_file_loc=self._tiling_configs_path(),
            on_progress=on_progress,
            frames_per_sec=rate,
            enable_timestamp_overlay=timestamp_overlay,
        )

    def enhance(
        self,
        target: str | pathlib.Path,
        *,
        on_progress: ProgressCallback | None = None,
        on_derived_image: Callable[[np.ndarray, int], None] | None = None,
    ) -> dict:
        """Quick Enhance *target* -- one image, or every image in a folder --
        each to its own derived file beside its source.

        Args:
            on_derived_image: Called with each enhanced image and its
                significant bits as it is saved, for a caller that shows it.

        Returns:
            ``created``, one entry per derived file; ``output_folder``; and
            ``message``, the outcome in words.
        """
        return self._run(
            self._enhance, 'enhance', pathlib.Path(target), on_progress, on_derived_image
        )

    def count_cells(
        self,
        folder: str | pathlib.Path,
        *,
        method: dict,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Count the cells in every image in *folder* by the cell-count
        *method*, writing ``results.csv`` into the folder.

        Returns:
            ``results_path``, ``counted`` and ``message``.

        Raises:
            PostProcessingRefusedError: the method cannot be used (reason
                ``method_invalid``), before the count is queued; or the
                folder has nothing to count.
        """
        from modules.post_processing import PostProcessing, check_cell_count_method

        check_cell_count_method(method)
        return self._run(
            PostProcessing().apply_cell_count_to_folder,
            'count_cells',
            folder,
            method,
            on_progress=on_progress,
        )

    def _run(self, action, member: str, folder, *args, **kwargs):
        """Run *action* on the post-processing lane and return its answer.

        The build says on the lane when it starts and how it ends, so a
        folder's derived files can be traced to the build that wrote them
        and its overlap with a run read back from the log. A build that
        raises is named by its outcome's type only: the reporter where the
        outcome's flight stops logs and shows it once.
        *folder* is the build's first argument, the folder (or, for an
        enhance, the file) it reads.
        """

        @functools.wraps(action)
        def build(*a, **kw):
            logger.info(f'[PostProc  ] {member} started on {folder}')
            started = time.monotonic()
            try:
                result = action(*a, **kw)
            except BaseException as outcome:
                logger.info(
                    f'[PostProc  ] {member} ended after {time.monotonic() - started:.1f} s: '
                    f'{type(outcome).__name__}'
                )
                raise
            logger.info(
                f'[PostProc  ] {member} ended after {time.monotonic() - started:.1f} s: '
                f'{result["message"]}'
            )
            return result

        return self.lane.call(
            IOTask(action=build, args=(folder, *args), kwargs=kwargs),
            f'post_processing.{member}',
            None,
        )

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
                operation='Video',
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
    ) -> dict:
        import cv2

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
        return {
            'created': created,
            'output_folder': output_folder,
            # The derived files sit beside their sources, so the words do not
            # repeat a path the person just chose.
            'message': 'Enhance complete.',
        }
