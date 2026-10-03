#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import contextlib
import copy
import csv
import json
import math
import numbers
import os
import time
import uuid
from collections.abc import Mapping

import numpy as np

import modules.image_utils as image_utils

from lvp_logger import logger
from modules.cell_count import CellCount
from modules.common_utils import CustomJSONizer
from modules.exceptions import PostProcessingFailedError, PostProcessingRefusedError
from modules.protocol_post_processor import ProgressCallback

# The operation's name as a person reads it, in its refusals and failures.
CELL_COUNT_OPERATION = 'Cell Count'

# What marks a JSON file as a saved cell-count method.
_CELL_COUNT_METHOD_METADATA = {'type': 'cell_count_method', 'version': '1'}

# Each filter's bounds, by their path under 'filters'. A bound of None is open.
_CELL_COUNT_FILTERS = (
    ('area',),
    ('perimeter',),
    ('sphericity',),
    ('intensity', 'min'),
    ('intensity', 'mean'),
    ('intensity', 'max'),
)


def default_cell_count_method() -> dict:
    """The cell-count method a person starts from, before any is loaded or adjusted.

    A new dict on every call: the caller owns and edits its copy. Areas are in
    square microns, perimeters in microns, intensities and the threshold in
    percent of full scale.
    """
    return {
        'context': {
            'pixels_per_um': 1.0,
            'fluorescent_mode': True,
        },
        'segmentation': {
            'algorithm': 'initial',
            'parameters': {
                'threshold': 20,
            },
        },
        'filters': {
            'area': {'min': 0, 'max': 100},
            'perimeter': {'min': 0, 'max': 100},
            'sphericity': {'min': 0.0, 'max': 1.0},
            'intensity': {
                'min': {'min': 0, 'max': 100},
                'mean': {'min': 0, 'max': 100},
                'max': {'min': 0, 'max': 100},
            },
        },
    }


def _refuse_method(source: str, problem: str) -> None:
    raise PostProcessingRefusedError(
        operation=CELL_COUNT_OPERATION,
        reason='method_invalid',
        message=f'{source} cannot be used: {problem}.',
    )


def _method_field(method: Mapping, path: tuple[str, ...], source: str):
    value = method
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            _refuse_method(source, f'it has no {".".join(path)}')
        value = value[key]
    return value


def _is_number(value) -> bool:
    # bool is an int to Python; true/false in a method file is not a number.
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def check_cell_count_method(method: object, *, source: str = 'The cell-count method') -> None:
    """Refuse a cell-count method the count cannot use, naming the field.

    The count reads every field checked here. A pixels-per-micron that is
    not a positive number does not fail the count: it scales every region
    out of the area and perimeter filters and counts nothing, or writes NaN
    areas, so it is refused before any image is read. The file's
    ``metadata`` is not part of the method and is not required.

    Args:
        source: What the refusal calls the method, e.g. naming its file.

    Raises:
        PostProcessingRefusedError: reason ``method_invalid``.
    """
    if not isinstance(method, Mapping):
        _refuse_method(source, f'it is a {type(method).__name__}, not a set of named settings')

    scale = _method_field(method, ('context', 'pixels_per_um'), source)
    if not _is_number(scale) or scale <= 0:
        _refuse_method(
            source,
            f'context.pixels_per_um must be a positive number of camera pixels '
            f'per micron, not {scale!r}',
        )

    fluorescent = _method_field(method, ('context', 'fluorescent_mode'), source)
    if not isinstance(fluorescent, bool):
        _refuse_method(
            source, f'context.fluorescent_mode must be true or false, not {fluorescent!r}'
        )

    _method_field(method, ('segmentation', 'algorithm'), source)
    threshold = _method_field(method, ('segmentation', 'parameters', 'threshold'), source)
    if not _is_number(threshold):
        _refuse_method(
            source, f'segmentation.parameters.threshold must be a number, not {threshold!r}'
        )

    for path in _CELL_COUNT_FILTERS:
        name = '.'.join(('filters', *path))
        low = _method_field(method, ('filters', *path, 'min'), source)
        high = _method_field(method, ('filters', *path, 'max'), source)
        for bound, value in (('min', low), ('max', high)):
            if value is not None and not _is_number(value):
                _refuse_method(source, f'{name}.{bound} must be a number or null, not {value!r}')
        if low is not None and high is not None and low > high:
            _refuse_method(source, f'{name}.min ({low}) is above its max ({high})')


def read_cell_count_image(path: str | os.PathLike) -> tuple[np.ndarray, int]:
    """Read an image to count: its pixels and their payload depth.

    One read returns pixels AND their payload depth together, so a
    right-aligned 12-bit TIFF scales to 8-bit by its true depth and the two
    can never be read out of sync.

    Raises:
        PostProcessingRefusedError: reason ``unreadable``; the file is missing
            or is not an image, and the message names it.
    """
    try:
        return image_utils.load_pixels(path)
    except (FileNotFoundError, ValueError) as e:
        raise PostProcessingRefusedError(
            operation=CELL_COUNT_OPERATION,
            reason='unreadable',
            message=f'The image cannot be counted: {e}.',
        ) from e


def _number_typed(text: str) -> int | float | str:
    """The number *text* spells, or *text* itself when it spells none."""
    for parse in (int, float):
        with contextlib.suppress(ValueError):
            return parse(text)
    return text


def with_pixels_per_um(method: Mapping, pixels_per_um: float | str) -> dict:
    """A copy of *method* that counts at *pixels_per_um*, refused if the count cannot use it.

    A string is read as the number it spells, as a person types it into a
    box -- a whole number as an integer, so the box can show back exactly what
    was typed; one that spells no number is refused naming it.

    Raises:
        PostProcessingRefusedError: reason ``method_invalid``; *method* is
            unchanged.
    """
    if isinstance(pixels_per_um, str):
        pixels_per_um = _number_typed(pixels_per_um)
    changed = copy.deepcopy(dict(method))
    changed['context'] = {**changed.get('context', {}), 'pixels_per_um': pixels_per_um}
    check_cell_count_method(changed)
    return changed


def load_cell_count_method(path: str | os.PathLike) -> dict:
    """Read a saved cell-count method file and return the method it holds.

    The file is the one ``save_cell_count_method`` writes: the method, plus a
    ``metadata`` entry naming its type and version that marks it as a method
    file. The method is checked as the count checks it, so a method that
    loads is one the count can use.

    Raises:
        PostProcessingRefusedError: the file cannot be read, is not a
            method file, or holds a method the count cannot use; the message
            names the file.
    """

    def refuse(reason: str, problem: str) -> PostProcessingRefusedError:
        return PostProcessingRefusedError(
            operation=CELL_COUNT_OPERATION,
            reason=reason,
            message=f'The cell-count method file {path} cannot be loaded: {problem}.',
        )

    try:
        with open(path) as f:
            saved = json.load(f)
    except OSError as e:
        raise refuse('method_unreadable', f'it could not be read ({e.strerror})') from e
    except ValueError as e:
        raise refuse('method_unreadable', f'it is not JSON ({e})') from e

    metadata = saved.get('metadata') if isinstance(saved, Mapping) else None
    if not isinstance(metadata, Mapping) or not {'type', 'version'} <= metadata.keys():
        raise refuse('method_unreadable', 'it has no metadata naming its type and version')
    check_cell_count_method(saved, source=f'The cell-count method in {path}')
    return saved


def save_cell_count_method(method: Mapping, path: str | os.PathLike) -> None:
    """Write *method* to *path* as a cell-count method file.

    Raises:
        PostProcessingRefusedError: the method cannot be used, and nothing
            is written.
    """
    check_cell_count_method(method)
    saved = {**method, 'metadata': dict(_CELL_COUNT_METHOD_METADATA)}
    with open(path, 'w') as f:
        json.dump(saved, f, indent=4, cls=CustomJSONizer)


class PostProcessing:
    # A superset of the TIFF suffixes, not an independent list, so this cannot
    # drift out of agreement with what the rest of the project calls a TIFF.
    SUPPORTED_IMAGE_TYPES = tuple(
        sorted(image_utils.TIFF_SUFFIXES | {'.jpg', '.jpeg', '.png', '.bmp'})
    )

    def __init__(self):
        self._cell_count = CellCount()

    def convert_to_avi(self, filepath):
        pass

    def stitch(self, filepath):
        pass

    def preview_cell_count(
        self, image: np.ndarray, settings: Mapping, significant_bits: int
    ) -> tuple[np.ndarray, dict]:
        """Count the cells in one image by the cell-count method *settings*.

        Raises:
            PostProcessingRefusedError: the method cannot be used
                (``check_cell_count_method``).
        """
        check_cell_count_method(settings)
        preview_images, cell_stats = self._cell_count.process_image(
            image=image, settings=settings, significant_bits=significant_bits
        )

        return preview_images['filtered_contours'], cell_stats

    def apply_cell_count_to_folder(
        self,
        path: str | os.PathLike,
        settings: dict,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Count the cells in every image in *path* and write ``results.csv`` there.

        The results file replaces an existing one only once the new one is
        complete on disk: a folder with nothing to count, or a write that
        fails, leaves the previous results as they were.

        Returns:
            ``results_path``; ``counted``, how many images were analysed; and
            ``message``, the outcome in words.

        Raises:
            PostProcessingRefusedError: No image in the folder could be
                analysed -- none there, or none readable.
            PostProcessingFailedError: The results could not be written, or
                some images could not be read; the file holds every image
                that was, and the error names the rest.
        """
        fields = ['file', 'time', 'num_cells', 'total_object_area (um2)', 'total_object_intensity']
        filenames = [f for f in os.listdir(path) if f.endswith(self.SUPPORTED_IMAGE_TYPES)]
        results = []
        unreadable = []

        for done, filename in enumerate(filenames, start=1):
            file_path = os.path.join(path, filename)
            try:
                image, significant_bits = read_cell_count_image(file_path)
            except PostProcessingRefusedError as e:
                logger.warning(f'[LVP Main  ] Skipping unreadable image {filename}: {e}')
                unreadable.append(str(e))
                continue

            _, region_info = self.preview_cell_count(
                image=image, settings=settings, significant_bits=significant_bits
            )

            time_created_raw = os.path.getctime(file_path)
            time_created = time.ctime(time_created_raw)

            results.append(
                {
                    'filename': os.path.basename(filename),
                    'time': time_created,
                    'num_cells': region_info['summary']['num_regions'],
                    'total_object_area (um2)': region_info['summary']['total_object_area'],
                    'total_object_intensity': region_info['summary']['total_object_intensity'],
                }
            )
            if on_progress is not None:
                on_progress(100 * done / len(filenames), f'{done}/{len(filenames)}: {filename}')

        if not results:
            message = (
                f'None of the {len(filenames)} image(s) in the folder could be read.'
                if filenames
                else 'No images were found in the selected folder.'
            )
            raise PostProcessingRefusedError(
                operation=CELL_COUNT_OPERATION, reason='no_data', message=message
            )

        results_file_path = os.path.join(path, 'results.csv')
        temp_path = os.path.join(path, f'.results.{uuid.uuid4().hex}.tmp.csv')
        try:
            with open(temp_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                for record in results:
                    writer.writerow(record.values())
            os.replace(temp_path, results_file_path)
        except OSError as e:
            with contextlib.suppress(OSError):
                os.remove(temp_path)
            raise PostProcessingFailedError(
                operation=CELL_COUNT_OPERATION,
                missing=f'The results could not be written to {results_file_path} ({e}).',
            ) from e

        if unreadable:
            raise PostProcessingFailedError(
                operation=CELL_COUNT_OPERATION,
                missing=f'{len(unreadable)} of {len(filenames)} image(s) could not be read.',
                produced_paths=(results_file_path,),
                output_root=str(path),
                errors=unreadable,
            )
        return {
            'results_path': results_file_path,
            'counted': len(results),
            'message': f'Counted cells in {len(results)} image(s). Results: {results_file_path}',
        }
