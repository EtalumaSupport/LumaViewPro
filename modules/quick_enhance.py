"""Deterministic, local image enhancements for visual inspection.

Quick Enhance deliberately creates derived files only.  It is not a
quantitative-processing path and must never be used by image acquisition or
the live display worker.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Callable

import cv2
import numpy as np

from lvp_logger import logger
from modules import common_utils, image_utils


PIPELINE_VERSION = '1'
QUANTITATIVE_USE_WARNING = (
    'Quick Enhance is for visual inspection and derived exports. '
    'Use raw images for quantitative analysis.'
)
SUPPORTED_SUFFIXES = frozenset({'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'})
MAX_IMAGE_PIXELS = 100_000_000


@dataclass(frozen=True)
class QuickEnhanceSettings:
    """Conservative settings for a deterministic visual enhancement."""

    preset: str = 'Auto (Recommended)'
    low_percentile: float = 1.0
    high_percentile: float = 99.0
    gamma: float = 1.0
    background_enabled: bool = False
    background_sigma: float = 24.0
    denoise_enabled: bool = False
    denoise_kernel_size: int = 3

    @classmethod
    def for_preset(cls, preset: str) -> 'QuickEnhanceSettings':
        presets = {
            'Auto (Recommended)': cls(preset='Auto (Recommended)'),
            'Brightfield / Phase': cls(preset='Brightfield / Phase', gamma=0.9),
            'Contrast Only': cls(preset='Contrast Only'),
        }
        try:
            return presets[preset]
        except KeyError as exc:
            raise ValueError(f'Unknown Quick Enhance preset: {preset}') from exc


class QuickEnhancer:
    """Apply and export a safe enhancement recipe without GUI dependencies."""

    def validate(
        self,
        settings: QuickEnhanceSettings,
        dtype: np.dtype,
        significant_bits: int,
    ) -> list[str]:
        errors = []
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            errors.append('Quick Enhance supports uint8 and uint16 images only.')
        container_bits = dtype.itemsize * 8
        if not isinstance(significant_bits, (int, np.integer)) or not 1 <= significant_bits <= container_bits:
            errors.append(f'Significant bits must be between 1 and {container_bits}.')
        if not 0 <= settings.low_percentile < settings.high_percentile <= 100:
            errors.append('Low percentile must be below high percentile, within 0 to 100.')
        if not 0.5 <= settings.gamma <= 2.0:
            errors.append('Gamma must be between 0.5 and 2.0.')
        if not 1.0 <= settings.background_sigma <= 100.0:
            errors.append('Background radius must be between 1 and 100 pixels.')
        if settings.denoise_kernel_size not in (3, 5, 7):
            errors.append('Denoise kernel must be 3, 5, or 7 pixels.')
        return errors

    def settings_for_source(
        self,
        source_path: str | pathlib.Path,
        settings: QuickEnhanceSettings,
    ) -> tuple[QuickEnhanceSettings, str]:
        """Resolve Auto per file using acquisition metadata, never pixel guesses."""
        if settings.preset != 'Auto (Recommended)':
            return settings, f'{settings.preset} selected.'
        source_path = pathlib.Path(source_path)
        channel = self._source_channel(source_path)
        if channel in common_utils.get_transmitted_layers():
            return (
                QuickEnhanceSettings.for_preset('Brightfield / Phase'),
                f'Detected transmitted channel: {channel}.',
            )
        if channel == 'Composite':
            return QuickEnhanceSettings.for_preset('Auto (Recommended)'), 'Detected Composite TIFF.'
        if channel in common_utils.get_image_layers():
            return QuickEnhanceSettings.for_preset('Auto (Recommended)'), f'Detected fluorescence channel: {channel}.'
        return QuickEnhanceSettings.for_preset('Auto (Recommended)'), 'Image type unknown; using neutral auto levels.'

    @staticmethod
    def _source_channel(source_path: pathlib.Path) -> str | None:
        metadata = image_utils.read_postproc_input_metadata(source_path) or {}
        channel = metadata.get('channel')
        if isinstance(channel, str) and channel:
            return channel
        # Some ImageJ-style monochrome TIFFs do not round-trip the channel
        # field through read_postproc_input_metadata. LVP names include the
        # layer as a token, so use that explicit label—not pixel appearance.
        tokens = {token.upper() for token in re.split(r'[^A-Za-z0-9]+', source_path.stem)}
        channel_by_token = {
            'BF': 'BF',
            'PC': 'PC',
            'DF': 'DF',
            'BLUE': 'Blue',
            'GREEN': 'Green',
            'RED': 'Red',
            'LUMI': 'Lumi',
            'COMPOSITE': 'Composite',
        }
        for token, label in channel_by_token.items():
            if token in tokens:
                return label
        return None

    @staticmethod
    def colorize_mono_for_visual_output(image: np.ndarray, channel: str | None) -> np.ndarray:
        """Return RGB false color for a mono fluorescence derived image.

        Quick Enhance is a visual-export path. It retains transmitted-light
        and unknown mono images as mono, while recognized fluorescence layers
        receive their established display color only in the derived output.
        """
        if image.ndim == 2 and channel in common_utils.get_image_layers():
            return image_utils.mono_to_rgb_falsecolor(image, channel)
        return image

    def apply(
        self,
        image: np.ndarray,
        settings: QuickEnhanceSettings,
        significant_bits: int,
    ) -> np.ndarray:
        """Return an enhanced copy at the source dtype and payload range."""
        if image.size == 0:
            raise ValueError('Quick Enhance cannot process an empty image.')
        is_mono = image.ndim == 2
        is_color = image.ndim == 3 and image.shape[2] in (3, 4)
        if not (is_mono or is_color):
            raise ValueError('Quick Enhance supports monochrome, RGB, and RGBA images only.')
        if image.size > MAX_IMAGE_PIXELS:
            raise MemoryError(
                'Image is too large for Quick Enhance. Export a smaller region or use raw data.'
            )
        errors = self.validate(settings, image.dtype, significant_bits)
        if errors:
            raise ValueError(f'Invalid Quick Enhance settings: {" ".join(errors)}')

        source_max = float((1 << int(significant_bits)) - 1)
        # Work in float so background subtraction cannot wrap unsigned values.
        # RGB/RGBA receives one shared levels/gamma curve derived from mean
        # luminance. That keeps channel relationships predictable instead of
        # independently stretching the component channels into a new composite.
        work = np.clip(image.astype(np.float32, copy=False), 0.0, source_max) / source_max
        channels = work if is_mono else work[..., :3].copy()
        if not np.isfinite(work).all():
            channels = np.nan_to_num(channels, nan=0.0, posinf=1.0, neginf=0.0)

        if settings.background_enabled and min(image.shape) >= 3:
            background = cv2.GaussianBlur(
                channels,
                (0, 0),
                sigmaX=float(settings.background_sigma),
                sigmaY=float(settings.background_sigma),
                borderType=cv2.BORDER_REPLICATE,
            )
            # Keep a conservative low background pedestal; this avoids an
            # all-black result while removing large-scale uneven illumination.
            pedestal = np.percentile(background, 5.0, axis=(0, 1), keepdims=True)
            channels = np.clip(channels - background + pedestal, 0.0, 1.0)

        if settings.denoise_enabled and min(image.shape) >= settings.denoise_kernel_size:
            channels = cv2.medianBlur(channels, int(settings.denoise_kernel_size))

        levels_source = channels if is_mono else channels.mean(axis=2)
        finite = levels_source[np.isfinite(levels_source)]
        if finite.size == 0:
            return image.copy()
        black, white = np.percentile(
            finite, (float(settings.low_percentile), float(settings.high_percentile))
        )
        if white <= black:
            # A uniform image has no meaningful levels to stretch.  Returning
            # a copy avoids inventing contrast (or changing it via gamma).
            return image.copy()
        channels = np.clip((channels - black) / (white - black), 0.0, 1.0)
        if settings.gamma != 1.0:
            channels = np.power(channels, float(settings.gamma), dtype=np.float32)

        restored = image.copy()
        enhanced_channels = np.rint(np.clip(channels, 0.0, 1.0) * source_max).astype(
            image.dtype, copy=False
        )
        if is_mono:
            return enhanced_channels
        restored[..., :3] = enhanced_channels
        return restored

    def preview(
        self,
        image: np.ndarray,
        settings: QuickEnhanceSettings,
        significant_bits: int,
        max_dimension: int = 1200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 8-bit before/after display arrays; intended for a worker task."""
        preview_source = self._downsample(image, max_dimension)
        preview_after = self.apply(preview_source, settings, significant_bits)
        return (
            image_utils.convert_to_8bit(preview_source, significant_bits),
            image_utils.convert_to_8bit(preview_after, significant_bits),
        )

    @staticmethod
    def _downsample(image: np.ndarray, max_dimension: int) -> np.ndarray:
        height, width = image.shape[:2]
        current = max(height, width)
        if current <= max_dimension:
            return image.copy()
        scale = max_dimension / current
        return cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def recipe_dict(
        self,
        *,
        source_path: pathlib.Path,
        output_path: pathlib.Path,
        settings: QuickEnhanceSettings,
        input_dtype: np.dtype,
        output_dtype: np.dtype,
        significant_bits: int,
    ) -> dict:
        return {
            'source_filename': source_path.name,
            'derived_filename': output_path.name,
            'pipeline': 'Quick Enhance',
            'pipeline_version': PIPELINE_VERSION,
            'timestamp': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
            'input_dtype': np.dtype(input_dtype).name,
            'output_dtype': np.dtype(output_dtype).name,
            'significant_bits': int(significant_bits),
            'operations': {
                'auto_levels': {
                    'enabled': True,
                    'low_percentile': settings.low_percentile,
                    'high_percentile': settings.high_percentile,
                },
                'gamma': {'enabled': settings.gamma != 1.0, 'value': settings.gamma},
                'background_correction': {
                    'enabled': settings.background_enabled,
                    'sigma': settings.background_sigma,
                },
                'denoise': {
                    'enabled': settings.denoise_enabled,
                    'kernel_size': settings.denoise_kernel_size,
                },
            },
            'settings': asdict(settings),
            'quantitative_use_warning': QUANTITATIVE_USE_WARNING,
        }

    def export_file(self, source_path: str | pathlib.Path, settings: QuickEnhanceSettings) -> dict:
        source_path = pathlib.Path(source_path)
        settings, _ = self.settings_for_source(source_path, settings)
        image, significant_bits = image_utils.load_pixels(source_path)
        output = self.apply(image, settings, significant_bits)
        output_path = self._next_output_path(source_path)
        recipe_path = output_path.with_suffix('.recipe.json')

        temp_output = output_path.with_name(f'.{output_path.stem}.{uuid.uuid4().hex}.tmp.tif')
        temp_recipe = recipe_path.with_name(f'.{recipe_path.name}.{uuid.uuid4().hex}.tmp')
        try:
            source_metadata = image_utils.read_postproc_input_metadata(source_path) or {}
            channel = str(source_metadata.get('channel') or self._source_channel(source_path) or 'BF')
            metadata = image_utils.build_postproc_output_metadata(
                input_path=source_path,
                channel=channel,
                significant_bits=significant_bits,
            )
            output_for_write = self.colorize_mono_for_visual_output(output, channel)
            if source_path.suffix.lower() not in ('.tif', '.tiff') and output.ndim == 3:
                # cv2 loads PNG/JPEG in BGR(A), whereas TIFF metadata and the
                # project's TIFF writer are RGB-native.
                output_for_write = cv2.cvtColor(
                    output,
                    cv2.COLOR_BGRA2RGBA if output.shape[2] == 4 else cv2.COLOR_BGR2RGB,
                )
            image_utils.write_tiff(
                data=output_for_write,
                file_loc=temp_output,
                metadata=metadata,
                ome=False,
                color=channel,
                significant_bits=significant_bits,
                save_encoding='right_aligned',
            )
            recipe = self.recipe_dict(
                source_path=source_path,
                output_path=output_path,
                settings=settings,
                input_dtype=image.dtype,
                output_dtype=output.dtype,
                significant_bits=significant_bits,
            )
            temp_recipe.write_text(json.dumps(recipe, indent=2), encoding='utf-8')
            os.replace(temp_recipe, recipe_path)
            os.replace(temp_output, output_path)
        except (OSError, ValueError, cv2.error, MemoryError):
            for temporary in (temp_output, temp_recipe):
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.warning('[QuickEnhance] Could not remove temporary file %s', temporary)
            raise

        return {'source_path': source_path, 'output_path': output_path, 'recipe_path': recipe_path}

    def export_folder(
        self,
        folder: str | pathlib.Path,
        settings: QuickEnhanceSettings,
        progress_callback: Callable[[int, int, pathlib.Path], None] | None = None,
    ) -> dict:
        folder = pathlib.Path(folder)
        files = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            and '_enhanced' not in path.stem.lower()
        )
        created = []
        skipped = []
        total = len(files)
        for completed, source_path in enumerate(files, start=1):
            try:
                created.append(self.export_file(source_path, settings))
            except (OSError, ValueError, cv2.error, MemoryError) as exc:
                logger.warning('[QuickEnhance] Skipping %s: %s', source_path, exc, exc_info=True)
                skipped.append({'source_path': source_path, 'error': str(exc)})
            if progress_callback is not None:
                progress_callback(completed, total, source_path)
        return {
            'status': True,
            'created_count': len(created),
            'created': created,
            'skipped': skipped,
            'total': total,
        }

    @staticmethod
    def output_folder(result: dict) -> pathlib.Path | None:
        """Return the common derived-output folder from a completed export."""
        created = result.get('created') or []
        if not created:
            return None
        output_path = created[0].get('output_path')
        return pathlib.Path(output_path).parent if output_path is not None else None

    @staticmethod
    def _next_output_path(source_path: pathlib.Path) -> pathlib.Path:
        base = source_path.with_name(f'{source_path.stem}_enhanced.tif')
        if not base.exists():
            return base
        for index in range(2, 10_002):
            candidate = source_path.with_name(f'{source_path.stem}_enhanced_{index}.tif')
            if not candidate.exists():
                return candidate
        raise FileExistsError(
            f'Could not choose a derived filename for {source_path.name}: too many existing exports.'
        )
