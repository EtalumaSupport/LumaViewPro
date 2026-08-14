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
from collections.abc import Callable

import cv2
import numpy as np

from lvp_logger import logger
from modules import common_utils, image_utils


PIPELINE_VERSION = '3'
QUANTITATIVE_USE_WARNING = (
    'Quick Enhance is for visual inspection and derived exports. '
    'Use raw images for quantitative analysis. '
    'For AI-assisted, validated quantitative enhancement workflows, use LumaQuant Pro.'
)
# Quick Enhance reads more than the project's own capture format, so this is a
# superset of the TIFF suffixes rather than an independent list -- the two must
# not be able to disagree about what counts as a TIFF.
SUPPORTED_SUFFIXES = image_utils.TIFF_SUFFIXES | frozenset({'.png', '.jpg', '.jpeg', '.bmp'})
MAX_IMAGE_PIXELS = 100_000_000


@dataclass(frozen=True)
class QuickEnhanceSettings:
    """Fixed, non-AI recipe for deterministic visual enhancement."""

    low_percentile: float = 1.0
    high_percentile: float = 99.0
    gamma: float = 0.9
    illumination_correction_enabled: bool = True
    denoise_enabled: bool = False
    denoise_kernel_size: int = 3


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
        if (
            not isinstance(significant_bits, (int, np.integer))
            or not 1 <= significant_bits <= container_bits
        ):
            errors.append(f'Significant bits must be between 1 and {container_bits}.')
        if not 0 <= settings.low_percentile < settings.high_percentile <= 100:
            errors.append('Low percentile must be below high percentile, within 0 to 100.')
        if not 0.5 <= settings.gamma <= 2.0:
            errors.append('Gamma must be between 0.5 and 2.0.')
        if settings.denoise_kernel_size not in (3, 5, 7):
            errors.append('Denoise kernel must be 3, 5, or 7 pixels.')
        return errors

    @staticmethod
    def _legacy_false_color_channel(source_pixels: np.ndarray) -> tuple[str, int] | None:
        """Return the channel encoded by a legacy one-plane RGB TIFF, if any."""
        if source_pixels.ndim != 3 or source_pixels.shape[2] != 3:
            return None
        nonzero_channels = [index for index in range(3) if source_pixels[..., index].any()]
        if len(nonzero_channels) != 1:
            return None
        index = nonzero_channels[0]
        return ('Red', 'Green', 'Blue')[index], index

    @classmethod
    def _source_channel(
        cls,
        source_path: pathlib.Path,
        *,
        source_pixels: np.ndarray | None = None,
        source_metadata: dict | None = None,
    ) -> str | None:
        metadata = source_metadata or image_utils.read_postproc_input_metadata(source_path) or {}
        channel = metadata.get('channel')
        if isinstance(channel, str) and channel:
            return channel
        if source_pixels is not None:
            legacy_channel = cls._legacy_false_color_channel(source_pixels)
            if legacy_channel is not None:
                return legacy_channel[0]
        # Some ImageJ-style monochrome TIFFs do not round-trip the channel
        # field through read_postproc_input_metadata. A channel may follow a
        # numeric acquisition prefix (for example ``0green_s``), so letters
        # rather than digits form the conservative filename boundary.
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
            if re.search(rf'(?<![A-Za-z]){token}(?![A-Za-z])', source_path.stem, re.IGNORECASE):
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

    @staticmethod
    def _apply_guarded_unsharp(channels: np.ndarray, is_mono: bool) -> np.ndarray:
        """Sharpen signal edges without amplifying the dark background."""
        if min(channels.shape[:2]) < 3:
            return channels
        intensity = channels if is_mono else channels.mean(axis=2)
        finite = intensity[np.isfinite(intensity)]
        if finite.size == 0:
            return channels
        background = float(np.percentile(finite, 65.0))
        signal_top = float(np.percentile(finite, 95.0))
        if signal_top <= background:
            signal_top = float(finite.max())
        if signal_top <= background:
            return channels

        blurred = cv2.GaussianBlur(intensity.astype(np.float32, copy=False), (0, 0), 1.0)
        signal_gate = np.clip(
            (intensity - background) / max(signal_top - background, 1.0 / 65535.0),
            0.0,
            1.0,
        )
        sharpened_intensity = np.clip(
            intensity + 0.65 * (intensity - blurred) * signal_gate,
            0.0,
            1.0,
        )
        if is_mono:
            return sharpened_intensity
        gain = sharpened_intensity / np.maximum(intensity, 1.0 / 65535.0)
        return np.clip(channels * gain[..., np.newaxis], 0.0, 1.0)

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
        # Work in float so illumination correction cannot wrap unsigned values.
        # RGB/RGBA uses one corrected luminance plane, then scales its source
        # RGB values together. This preserves colour relationships rather than
        # independently remapping channels into a new composite.
        work = np.clip(image.astype(np.float32, copy=False), 0.0, source_max) / source_max
        source_channels = work if is_mono else work[..., :3]
        if not np.isfinite(source_channels).all():
            source_channels = np.nan_to_num(source_channels, nan=0.0, posinf=1.0, neginf=0.0)
        channels = source_channels.copy()
        intensity = channels if is_mono else channels.mean(axis=2)

        if settings.illumination_correction_enabled and min(image.shape) >= 3:
            illumination = self._global_illumination_plane(intensity)
            # The global plane corrects field-scale illumination drift without
            # following individual objects, so it cannot add dark edge halos.
            # The reference restores the scene's overall brightness.
            reference = float(np.median(illumination))
            intensity = np.clip(
                intensity / np.maximum(illumination, 1.0 / 65535.0) * reference,
                0.0,
                1.0,
            )

        if is_mono:
            channels = intensity
        else:
            original_intensity = source_channels.mean(axis=2)
            gain = intensity / np.maximum(original_intensity, 1.0 / 65535.0)
            channels = np.clip(source_channels * gain[..., np.newaxis], 0.0, 1.0)

        if settings.denoise_enabled and min(image.shape) >= settings.denoise_kernel_size:
            channels = cv2.medianBlur(channels, int(settings.denoise_kernel_size))

        levels_source = channels if is_mono else channels.mean(axis=2)
        finite = levels_source[np.isfinite(levels_source)]
        if finite.size == 0:
            return image.copy()
        black, white = np.percentile(
            finite, (float(settings.low_percentile), float(settings.high_percentile))
        )
        if white > black:
            channels = np.clip((channels - black) / (white - black), 0.0, 1.0)
            if settings.gamma != 1.0:
                channels = np.power(channels, float(settings.gamma), dtype=np.float32)
        channels = self._apply_guarded_unsharp(channels, is_mono)

        restored = image.copy()
        enhanced_channels = np.rint(np.clip(channels, 0.0, 1.0) * source_max).astype(
            image.dtype, copy=False
        )
        if is_mono:
            return enhanced_channels
        restored[..., :3] = enhanced_channels
        return restored

    @staticmethod
    def _global_illumination_plane(intensity: np.ndarray) -> np.ndarray:
        """Fit a robust field-scale illumination plane, never a local object map."""
        height, width = intensity.shape
        stride = max(1, int(np.ceil(np.sqrt(intensity.size / 100_000))))
        sample = intensity[::stride, ::stride]
        values = sample.reshape(-1)
        finite = np.isfinite(values)
        if finite.sum() < 3:
            return np.full_like(intensity, float(np.nanmedian(intensity)))
        cutoff = np.percentile(values[finite], 90.0)
        fit_mask = finite & (values <= cutoff)
        if fit_mask.sum() < 3:
            fit_mask = finite
        sample_y = np.arange(0, height, stride, dtype=np.float32) / max(height - 1, 1)
        sample_x = np.arange(0, width, stride, dtype=np.float32) / max(width - 1, 1)
        x = np.broadcast_to(sample_x, sample.shape).reshape(-1)
        y = np.broadcast_to(sample_y[:, np.newaxis], sample.shape).reshape(-1)
        design = np.column_stack((x[fit_mask], y[fit_mask], np.ones(fit_mask.sum())))
        coefficients, *_ = np.linalg.lstsq(design, values[fit_mask], rcond=None)
        full_y = np.arange(height, dtype=np.float32) / max(height - 1, 1)
        full_x = np.arange(width, dtype=np.float32) / max(width - 1, 1)
        plane = (
            coefficients[0] * full_x[np.newaxis, :]
            + coefficients[1] * full_y[:, np.newaxis]
            + coefficients[2]
        )
        plane = plane.astype(np.float32, copy=False)
        return np.maximum(plane, 1.0 / 65535.0, out=plane)

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
                'illumination_correction': {
                    'enabled': settings.illumination_correction_enabled,
                    'method': 'global_plane_normalization',
                    'fit_excludes_upper_percentile': 10.0,
                },
                'denoise': {
                    'enabled': settings.denoise_enabled,
                    'kernel_size': settings.denoise_kernel_size,
                },
                'sharpen': {
                    'enabled': True,
                    'method': 'signal_gated_unsharp_mask',
                    'sigma_px': 1.0,
                    'gain': 0.65,
                },
            },
            'settings': asdict(settings),
            'quantitative_use_warning': QUANTITATIVE_USE_WARNING,
        }

    def export_file(
        self,
        source_path: str | pathlib.Path,
        settings: QuickEnhanceSettings,
        display_callback: Callable[[np.ndarray, int], None] | None = None,
    ) -> dict:
        source_path = pathlib.Path(source_path)
        source_pixels, significant_bits = image_utils.load_pixels(
            source_path, collapse_legacy_false_color=False
        )
        legacy_channel = self._legacy_false_color_channel(source_pixels)
        image = (
            source_pixels[..., legacy_channel[1]].copy()
            if legacy_channel is not None
            else source_pixels
        )
        output = self.apply(image, settings, significant_bits)
        output_path = self._next_output_path(source_path)
        recipe_path = output_path.with_suffix('.recipe.json')

        temp_output = output_path.with_name(f'.{output_path.stem}.{uuid.uuid4().hex}.tmp.tif')
        temp_recipe = recipe_path.with_name(f'.{recipe_path.name}.{uuid.uuid4().hex}.tmp')
        try:
            source_metadata = image_utils.read_postproc_input_metadata(source_path) or {}
            channel = str(
                self._source_channel(
                    source_path,
                    source_pixels=source_pixels,
                    source_metadata=source_metadata,
                )
                or 'BF'
            )
            metadata = image_utils.build_postproc_output_metadata(
                input_path=source_path,
                channel=channel,
                significant_bits=significant_bits,
            )
            output_for_write = self.colorize_mono_for_visual_output(output, channel)
            if output_for_write.ndim == 3 and output_for_write.shape[2] == 4:
                # The TIFF writer supports RGB samples, not RGBA. Alpha is a
                # display-only component for these composite inputs, so retain
                # the enhanced RGB data and preserve the Composite metadata.
                output_for_write = output_for_write[..., :3]
            if not image_utils.is_tiff(source_path) and output_for_write.ndim == 3:
                # cv2 loads PNG/JPEG in BGR(A), whereas TIFF metadata and the
                # project's TIFF writer are RGB-native. Convert the alpha-free
                # display payload so a Composite PNG cannot reintroduce RGBA.
                output_for_write = cv2.cvtColor(
                    output_for_write,
                    cv2.COLOR_BGR2RGB,
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

        if display_callback is not None:
            try:
                display_callback(output_for_write, significant_bits)
            except Exception:
                logger.warning(
                    '[QuickEnhance] Could not queue derived image for display', exc_info=True
                )

        return {'source_path': source_path, 'output_path': output_path, 'recipe_path': recipe_path}

    def export_folder(
        self,
        folder: str | pathlib.Path,
        settings: QuickEnhanceSettings,
        progress_callback: Callable[[int, int, pathlib.Path], None] | None = None,
        display_callback: Callable[[np.ndarray, int], None] | None = None,
    ) -> dict:
        folder = pathlib.Path(folder)
        files = sorted(
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and '_enhanced' not in path.stem.lower()
        )
        created = []
        skipped = []
        total = len(files)
        for completed, source_path in enumerate(files, start=1):
            try:
                created.append(
                    self.export_file(
                        source_path,
                        settings,
                        display_callback=display_callback,
                    )
                )
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
