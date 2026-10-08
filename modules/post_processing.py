#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import contextlib
import copy
import csv
import json
import math
import numbers
import os
import re
import time
import uuid
from collections.abc import Mapping
from typing import NoReturn

import numpy as np
import pandas as pd

import modules.image_utils as image_utils

from lvp_logger import logger
from modules.cell_count import CellCount
from modules.common_utils import CustomJSONizer, read_table
from modules.exceptions import (
    CellCountScaleDroppedNotice,
    PostProcessingFailedError,
    PostProcessingRefusedError,
)
from modules.notification_center import notifications
from modules.protocol_post_processor import ProgressCallback

# The operation's name as a person reads it, in its refusals and failures.
CELL_COUNT_OPERATION = 'Cell Count'

# The operation's name for reading a results file back to graph it.
GRAPHING_OPERATION = 'Graphing'

# The layout time.ctime writes each image's time in, which a results file is
# read back by. '%d' also reads ctime's space-padded day.
RESULTS_TIME_FORMAT = '%a %b %d %H:%M:%S %Y'

# How a number is written in a results file: a column whose every cell is
# written this way is a column of numbers, of whole numbers when every cell
# is the first form.
_WHOLE_NUMBER = re.compile(r'[+-]?\d+')
_NUMBER = re.compile(r'[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?')

# What marks a JSON file as a saved cell-count method.
# Version 2: context.pixels_per_um is an optional override. Every version-1 file
# carries the old fixed default of 1.0, which the load drops.
_CELL_COUNT_METHOD_METADATA = {'type': 'cell_count_method', 'version': '2'}

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
    percent of full scale. It sets no scale, so each image is counted at the
    scale it states (``cell_count_scale``), and no size filter, so an image
    that states none is counted in pixels.
    """
    return {
        'context': {
            'pixels_per_um': None,
            'fluorescent_mode': True,
        },
        'segmentation': {
            'algorithm': 'initial',
            'parameters': {
                'threshold': 20,
            },
        },
        'filters': {
            'area': {'min': None, 'max': None},
            'perimeter': {'min': None, 'max': None},
            'sphericity': {'min': 0.0, 'max': 1.0},
            'intensity': {
                'min': {'min': 0, 'max': 100},
                'mean': {'min': 0, 'max': 100},
                'max': {'min': 0, 'max': 100},
            },
        },
    }


def _refuse_method(source: str, problem: str) -> NoReturn:
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
    areas, so it is refused before any image is read. None is no override:
    each image is counted at its own scale. The file's ``metadata`` is not
    part of the method and is not required.

    Args:
        source: What the refusal calls the method, e.g. naming its file.

    Raises:
        PostProcessingRefusedError: reason ``method_invalid``.
    """
    scale = _method_field(method, ('context', 'pixels_per_um'), source)
    if scale is not None and (not _is_number(scale) or scale <= 0):
        _refuse_method(
            source,
            f'context.pixels_per_um must be a positive number of camera pixels '
            f"per micron, or null to use each image's own scale, not {scale!r}",
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


def cell_count_scale(method: Mapping, image_pixel_size_um: float | None) -> float | None:
    """The pixels per micron a count measures an image at, or None for pixels.

    The method's ``context.pixels_per_um`` when a person set one; else the
    image's own scale (``image_utils.read_pixel_size_um``, microns per pixel);
    else None, and the image is counted in pixels.
    """
    override = method['context']['pixels_per_um']
    if override is not None:
        return override
    if image_pixel_size_um is not None:
        return 1.0 / image_pixel_size_um
    return None


def _refuse_size_filters_without_a_scale(method: Mapping, name: str) -> None:
    """Refuse a count in pixels whose method bounds an area or a perimeter.

    The bounds are in microns; applying them to pixels would count a
    different set of objects and report it as the filtered one.
    """
    for size in ('area', 'perimeter'):
        bounds = method['filters'][size]
        if bounds['min'] is not None or bounds['max'] is not None:
            raise PostProcessingRefusedError(
                operation=CELL_COUNT_OPERATION,
                reason='no_scale',
                message=(
                    f'{name} states no scale, so it is counted in pixels, and the '
                    f"method's {size} filter is in microns. Clear the {size} filter, "
                    'or type a pixels-per-micron scale.'
                ),
            )


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
        # An emptied box is no override: count at each image's own scale.
        pixels_per_um = _number_typed(pixels_per_um) if pixels_per_um.strip() else None
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
    context = saved.get('context')
    if (
        metadata['version'] == '1'
        and isinstance(context, Mapping)
        and context.get('pixels_per_um') == 1.0
    ):
        saved['context'] = {**context, 'pixels_per_um': None}
        notifications.report_outcome(
            CellCountScaleDroppedNotice(path), solicited=True, category=CELL_COUNT_OPERATION
        )
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


def read_cell_count_results(path: str | os.PathLike) -> pd.DataFrame:
    """Read a cell-count results file into a table to graph.

    Every cell is read as the text written. A column whose every cell is a
    number is a column of numbers (whole numbers when every cell is one);
    any other column is text, so a cell such as ``NA`` is never read as a
    missing number. A ``time`` column of text is parsed by the format the
    count writes it in, so it reads back as a datetime column. Which columns
    can be plotted is ``results_axes``'s answer.

    Raises:
        PostProcessingRefusedError: reason ``results_unreadable``; the file
            cannot be read, is not a CSV LumaViewPro could have written (not
            UTF-8, a NUL, a quote that does not close, a row of the wrong
            length), has a time the count did not write,
            or has no column of numbers to plot. The message names the file.
    """

    def refuse(problem: str) -> PostProcessingRefusedError:
        return PostProcessingRefusedError(
            operation=GRAPHING_OPERATION,
            reason='results_unreadable',
            message=f'The results file {path} cannot be graphed: {problem}.',
        )

    try:
        with open(path, encoding='utf-8-sig', newline='') as f:
            columns, rows = read_table(f.read(), sep=',')
    except OSError as e:
        raise refuse(f'it could not be read ({e.strerror})') from e
    except ValueError as e:
        raise refuse(f'it is not a CSV file ({e})') from e
    table = pd.DataFrame(
        {
            column: _results_column([row[position] for row in rows])
            for position, column in enumerate(columns)
        }
    )
    if 'time' in table and not pd.api.types.is_numeric_dtype(table['time']):
        try:
            table['time'] = pd.to_datetime(table['time'], format=RESULTS_TIME_FORMAT)
        except (TypeError, ValueError) as e:
            raise refuse(f'its time column is not in the form the count writes ({e})') from e
    if not results_axes(table)[1]:
        raise refuse('it has no column of numbers to plot')
    return table


def _results_column(cells: list[str]) -> pd.Series:
    """One results column: numbers when every cell is written as one, else its text."""
    if cells and all(_WHOLE_NUMBER.fullmatch(cell) for cell in cells):
        return pd.Series([int(cell) for cell in cells], dtype='int64')
    if cells and all(_NUMBER.fullmatch(cell) for cell in cells):
        return pd.Series([float(cell) for cell in cells], dtype='float64')
    return pd.Series(cells, dtype=object)


def results_axes(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    """The columns of *table* a graph can take as its X axis and as its Y axis.

    Decided by each column's type, never its name: a number column is either
    axis; a datetime column is an X axis only; text, such as the file name, is
    neither.
    """
    numeric = [c for c in table if pd.api.types.is_numeric_dtype(table[c])]
    dates = [c for c in table if pd.api.types.is_datetime64_any_dtype(table[c])]
    return [c for c in table if c in numeric or c in dates], numeric


class PostProcessing:
    def __init__(self):
        self._cell_count = CellCount()

    def convert_to_avi(self, filepath):
        pass

    def stitch(self, filepath):
        pass

    def preview_cell_count(
        self,
        image: np.ndarray,
        settings: Mapping,
        significant_bits: int,
        *,
        pixels_per_um: float | None,
        name: str = 'The image',
    ) -> tuple[np.ndarray, dict]:
        """Count the cells in one image by the cell-count method *settings*.

        *pixels_per_um* is the scale to measure at (``cell_count_scale``);
        None counts in pixels. *name* is what a refusal calls the image.

        Raises:
            PostProcessingRefusedError: the method cannot be used
                (``check_cell_count_method``), or, with no scale, it bounds an
                area or a perimeter (reason ``no_scale``).
        """
        check_cell_count_method(settings)
        if pixels_per_um is None:
            _refuse_size_filters_without_a_scale(settings, name)
        preview_images, cell_stats = self._cell_count.process_image(
            image=image,
            settings=settings,
            significant_bits=significant_bits,
            pixels_per_um=pixels_per_um,
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

        Each image is measured at ``cell_count_scale``'s answer for it: the
        method's override, else the scale the image states, else pixels. The
        results say each row's area unit (``um2`` or ``px2``), so a folder of
        mixed images is one file.

        Returns:
            ``results_path``; ``counted``, how many images were analysed; and
            ``message``, the outcome in words.

        Raises:
            PostProcessingRefusedError: No image in the folder could be
                analysed -- none there, or none readable.
            PostProcessingFailedError: The results could not be written, or
                some images could not be counted (unreadable, or a size
                filter on an image with no scale); the file holds every image
                that was, and the error names the rest.
        """
        fields = [
            'file',
            'time',
            'num_cells',
            'total_object_area',
            'area_unit',
            'total_object_intensity',
        ]
        filenames = [f for f in os.listdir(path) if image_utils.is_image(f)]
        results = []
        not_counted = []

        for done, filename in enumerate(filenames, start=1):
            file_path = os.path.join(path, filename)
            try:
                image, significant_bits = read_cell_count_image(file_path)
                _, region_info = self.preview_cell_count(
                    image=image,
                    settings=settings,
                    significant_bits=significant_bits,
                    pixels_per_um=cell_count_scale(
                        settings, image_utils.read_pixel_size_um(file_path)
                    ),
                    name=filename,
                )
            except PostProcessingRefusedError as e:
                logger.warning(f'[LVP Main  ] Not counting {filename}: {e}')
                not_counted.append(str(e))
                continue

            time_created_raw = os.path.getctime(file_path)
            time_created = time.ctime(time_created_raw)

            results.append(
                {
                    'filename': os.path.basename(filename),
                    'time': time_created,
                    'num_cells': region_info['summary']['num_regions'],
                    'total_object_area': region_info['summary']['total_object_area'],
                    'area_unit': region_info['summary']['area_unit'],
                    'total_object_intensity': region_info['summary']['total_object_intensity'],
                }
            )
            if on_progress is not None:
                on_progress(100 * done / len(filenames), f'{done}/{len(filenames)}: {filename}')

        if not results:
            message = (
                f'None of the {len(filenames)} image(s) in the folder could be counted.'
                if filenames
                else 'No images were found in the selected folder.'
            )
            raise PostProcessingRefusedError(
                operation=CELL_COUNT_OPERATION, reason='no_data', message=message
            )

        results_file_path = os.path.join(path, 'results.csv')
        temp_path = os.path.join(path, f'.results.{uuid.uuid4().hex}.tmp.csv')
        try:
            with open(temp_path, 'w', newline='', encoding='utf-8') as f:
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

        if not_counted:
            raise PostProcessingFailedError(
                operation=CELL_COUNT_OPERATION,
                missing=f'{len(not_counted)} of {len(filenames)} image(s) could not be counted.',
                produced_paths=(results_file_path,),
                output_root=str(path),
                errors=not_counted,
            )
        return {
            'results_path': results_file_path,
            'counted': len(results),
            'message': f'Counted cells in {len(results)} image(s). Results: {results_file_path}',
        }
