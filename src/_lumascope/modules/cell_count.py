# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import math
from collections.abc import Mapping

import cv2
import numpy as np
import skimage

import _lumascope.modules.image_utils as image_utils
from _lumascope.modules.channel_record import GRAY, Color
from _lumascope.modules.settings_transformer import SettingsTransformer


class CellCount:
    def __init__(self):
        self._settings_transformer = SettingsTransformer(to_direction='digital')

    def _get_region_info(self, labeled_mask, intensity_image, pixels_per_um):

        def _sphericity(area, perimeter):
            if perimeter == 0:
                return -1
            return (4 * math.pi * area) / (perimeter**2)

        regions = skimage.measure.regionprops(
            labeled_mask, intensity_image=intensity_image, extra_properties=(_sphericity,)
        )

        region_prop_params = [
            'area',
            'perimeter',
            'intensity_min',
            'intensity_mean',
            'intensity_max',
        ]

        region_info = {}
        for region_idx, region in enumerate(regions):
            region_param_info = {}
            for param in region_prop_params:
                # Without a scale the count measures in pixels and says so.
                if param == 'area' and pixels_per_um is None:
                    val = round(region[param], 2)
                    units = 'px^2'
                elif param == 'area':
                    val = round(region[param] / (pixels_per_um**2), 2)
                    units = 'um^2'
                elif param in ('perimeter', 'equivalent_diameter') and pixels_per_um is None:
                    val = round(region[param], 2)
                    units = 'px'
                elif param in ('perimeter', 'equivalent_diameter'):
                    val = round(region[param] / (pixels_per_um), 2)
                    units = 'um'
                elif param in ['intensity_min', 'intensity_mean', 'intensity_max']:
                    val = np.round(region[param] / (255 / 100), 2)
                    units = '%'
                else:
                    val = region[param]
                    units = None

                region_param_info[param.lower()] = {'val': val, 'units': units}

            region_param_info['sphericity'] = {
                'val': _sphericity(
                    area=region_param_info['area']['val'],
                    perimeter=region_param_info['perimeter']['val'],
                ),
                'units': None,
            }

            # The pixel count, whatever the scale: the summary's area and
            # intensity are totalled in pixels.
            region_param_info['area_px2'] = {'val': float(region['area']), 'units': 'px^2'}

            region_info[region_idx] = region_param_info

        return {'summary': {'num_regions': len(region_info)}, 'regions': region_info}

    def _filter_regions(self, region_info, contours, settings):

        filtered_regions = {}
        filtered_contours = []

        def _within_bounds(subject, subject_key, criteria, criteria_key):
            if (criteria[criteria_key]['min'] is not None) and (
                subject[subject_key]['val'] < criteria[criteria_key]['min']
            ):
                return False

            return not (
                (criteria[criteria_key]['max'] is not None)
                and (subject[subject_key]['val'] > criteria[criteria_key]['max'])
            )

        total_object_area_px2 = 0.0
        total_object_intensity = 0

        filter_stats = {
            'area': 0,
            'perimeter': 0,
            'sphericity': 0,
            'intensity_min': 0,
            'intensity_mean': 0,
            'intensity_max': 0,
        }

        for (region_idx, region), contour in zip(
            region_info['regions'].items(), contours, strict=False
        ):
            if not _within_bounds(region, 'area', settings, 'area'):
                filter_stats['area'] += 1
                continue

            if not _within_bounds(region, 'perimeter', settings, 'perimeter'):
                filter_stats['perimeter'] += 1
                continue

            if not _within_bounds(region, 'sphericity', settings, 'sphericity'):
                filter_stats['sphericity'] += 1
                continue

            if not _within_bounds(region, 'intensity_min', settings['intensity'], 'min'):
                filter_stats['intensity_min'] += 1
                continue

            if not _within_bounds(region, 'intensity_mean', settings['intensity'], 'mean'):
                filter_stats['intensity_mean'] += 1
                continue

            if not _within_bounds(region, 'intensity_max', settings['intensity'], 'max'):
                filter_stats['intensity_max'] += 1
                continue

            total_object_area_px2 += region['area_px2']['val']
            total_object_intensity += region['area_px2']['val'] * region['intensity_mean']['val']
            filtered_regions[region_idx] = region
            filtered_contours.append(contour)

        filtered_region_info = {
            'summary': {
                'num_regions': len(filtered_regions),
                'total_object_area_px2': total_object_area_px2,
                'total_object_intensity': round(total_object_intensity, 1),
            },
            'filter_reasons': filter_stats,
            'summary_unfiltered': {'num_regions': len(contours)},
            'regions': filtered_regions,
        }

        return filtered_region_info, filtered_contours

    def process_image(
        self,
        image: np.ndarray,
        settings: Mapping,
        significant_bits: int,
        pixels_per_um: float | None,
        include_images: list[str] | str | None = None,
        *,
        baked: bool,
        color: Color | None,
    ) -> tuple[dict, dict]:
        """Count the objects in *image* and answer the images asked for.

        Args:
            baked: The image is one channel's, so a color image's planes are
                that channel's mono times its color and it is read by its
                brightest plane; else as ``rgb_image_to_gray`` reads any
                color image, by its one populated plane or its luminance.
            color: The color of the channel the image was taken in, which a
                mono image is shown in, or None when it states none (shown
                gray). The outlines are drawn in the color that stands out
                against the image's (``_outline_color``).
        """
        if include_images is None:
            include_images = ['filtered_contours']

        if image.dtype != np.uint8:
            if image.dtype == np.uint16:
                image = image_utils.convert_to_8bit(image, significant_bits=significant_bits)
            else:
                raise NotImplementedError(f'Unable to process image of type {image.dtype}')

        if include_images == 'all':
            include_images = [
                'gray',
                'denoised',
                'threshold',
                'unfiltered_contours',
                'filtered_contours',
            ]

        return_images = {}

        digital_settings = self._settings_transformer.transform(
            settings=settings, pixels_per_um=pixels_per_um
        )

        gray_image = image_utils.rgb_image_to_gray(image=image, baked=baked)

        if digital_settings['context']['fluorescent_mode'] is False:
            # Invert the image
            gray_image = cv2.bitwise_not(gray_image)

        if 'gray' in include_images:
            return_images['gray'] = gray_image

        # Denoise
        denoised_img = cv2.fastNlMeansDenoising(gray_image)

        if 'denoised' in include_images:
            return_images['denoised'] = denoised_img

        # Threshold
        _th, threshold_img = cv2.threshold(
            denoised_img,
            digital_settings['segmentation']['parameters']['threshold'],
            255,
            cv2.THRESH_BINARY,
        )

        if 'threshold' in include_images:
            return_images['threshold'] = threshold_img

        # Countours
        contours, _hierarchy = cv2.findContours(
            threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if 'unfiltered_contours' in include_images:
            return_images['unfiltered_contours'] = _outlined(image, contours, color)

        # Labeling
        label_map = np.zeros_like(gray_image, dtype='uint16')
        for i, contour in enumerate(contours):
            label_map = cv2.drawContours(label_map, [contour], -1, i + 1, -1)

        region_info = self._get_region_info(
            labeled_mask=label_map,
            intensity_image=gray_image,
            pixels_per_um=pixels_per_um,
        )

        filtered_region_info, filtered_contours = self._filter_regions(
            region_info=region_info, contours=contours, settings=settings['filters']
        )
        summary = filtered_region_info['summary']
        summary['total_object_area_um2'] = (
            None
            if pixels_per_um is None
            else round(summary['total_object_area_px2'] / (pixels_per_um**2), 2)
        )

        if 'filtered_contours' in include_images:
            return_images['filtered_contours'] = _outlined(image, filtered_contours, color)

        return return_images, filtered_region_info


# The outline on an image with no color of its own (gray), and on an image
# whose color's complement is nothing (white).
_ORANGE = (255, 100, 0)


def _outline_color(color: Color) -> tuple[int, int, int]:
    """The color outlines stand out in against an image of *color*: its
    complement, brightened so its brightest component is full, so a pale
    color's outline is no darker than a pure one's; orange against white."""
    complement = [1.0 - weight for weight in color]
    brightest = max(complement)
    if brightest == 0:
        return _ORANGE
    return tuple(round(255 * weight / brightest) for weight in complement)


def _outlined(image: np.ndarray, contours, color: Color | None) -> np.ndarray:
    """*image* with *contours* drawn on it, in red-green-blue.

    A mono image is shown in *color*, gray when None; a color image in its
    own colors, its outline set against *color*, else against its mean
    color.
    """
    if image.ndim == 2:
        shown = image_utils.mono_to_rgb_falsecolor(image, GRAY if color is None else color)
        against = GRAY if color is None else color
    else:
        shown = np.ascontiguousarray(image[..., :3])
        against = color if color is not None else _mean_color(shown)
    cv2.drawContours(shown, contours, -1, _outline_color(against), 3)
    return shown


def _mean_color(image: np.ndarray) -> Color:
    """A color image's mean color at full brightness; gray for a black one."""
    means = image.reshape(-1, 3).mean(axis=0)
    brightest = float(means.max())
    if brightest == 0:
        return GRAY
    return tuple(float(mean) / brightest for mean in means)
