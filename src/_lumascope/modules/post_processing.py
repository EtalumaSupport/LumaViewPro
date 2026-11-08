#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import contextlib
import dataclasses
import csv
import datetime
import json
import os
import pathlib
import re
import uuid
from collections.abc import Mapping

import numpy as np
import pandas as pd

import _lumascope.modules.channel_record as channel_record
import _lumascope.modules.image_utils as image_utils
import _lumascope.modules.path_utils as path_utils
import _lumascope.modules.quick_enhance as quick_enhance
from _lumascope.modules.protocol_post_processing_helper import (
    ProtocolPostProcessingHelper,
    RunRecord,
)
from _lumascope.modules.tiling_config import TilingGrids

from _lumascope.lvp_logger import logger
from _lumascope.modules.cell_count import CellCount
from _lumascope.modules.cell_count_method import CellCountMethod, from_saved_form, saved_form
from _lumascope.modules.common_utils import read_table
from _lumascope.modules.python_values import python_values
from _lumascope.modules.exceptions import (
    CellCountScaleDroppedNotice,
    LeftOutReason,
    LiveFolderPathRefusedError,
    PostProcessingFailedError,
    PostProcessingOperation,
    PostProcessingRefusedError,
)
from _lumascope.modules.notification_center import notifications
from _lumascope.modules.api_surface import ProgressCallback

# The operation's name as a person reads it, in its refusals and failures.
CELL_COUNT_OPERATION = PostProcessingOperation.CELL_COUNT

# The operation's name for reading a results file back to graph it.
GRAPHING_OPERATION = PostProcessingOperation.GRAPHING

# The layout results files written before each image's time was its own wrote
# it in (time.ctime's), which the reader still reads. '%d' also reads ctime's
# space-padded day.
RESULTS_TIME_FORMAT = '%a %b %d %H:%M:%S %Y'

# How a number is written in a results file: a column whose every cell is
# written this way is a column of numbers, of whole numbers when every cell
# is the first form.
_WHOLE_NUMBER = re.compile(r'[+-]?\d+')
_NUMBER = re.compile(r'[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?')

# The columns of results.csv, in order: which image, where and when it was
# taken, then what was measured. The header and every row are built from it.
RESULTS_COLUMNS = (
    'file',
    'well',
    'channel',
    'scan',
    'tile',
    'z',
    'time',
    'num_cells',
    'total_object_area_px2',
    'total_object_area_um2',
    'total_object_intensity',
)

# What marks a JSON file as a saved cell-count method.
# Version 2: context.pixels_per_um is an optional override. Every version-1 file
# carries the old fixed default of 1.0, which the load drops.
_CELL_COUNT_METHOD_METADATA = {'type': 'cell_count_method', 'version': '2'}


def cell_count_scale(method: CellCountMethod, image_pixel_size_um: float | None) -> float | None:
    """The pixels per micron a count measures an image at, or None for pixels.

    The method's ``pixels_per_um`` when a person set one; else the image's
    own scale (``image_utils.read_pixel_size_um``, microns per pixel); else
    None, and the image is counted in pixels.
    """
    override = method.pixels_per_um
    if override is not None:
        return override
    if image_pixel_size_um is not None:
        return 1.0 / image_pixel_size_um
    return None


def area_filter_max(shape: tuple[int, ...], pixels_per_um: float | None) -> float:
    """The largest area an image of ``shape`` can hold, in square microns at ``pixels_per_um``.

    With no scale it is the pixel count: a slider's reach only, since a
    bound on an image with no scale is refused.
    """
    max_area_pixels = shape[0] * shape[1]
    if pixels_per_um is None:
        return float(max_area_pixels)
    return max_area_pixels / (pixels_per_um**2)


def perimeter_filter_max(shape: tuple[int, ...], pixels_per_um: float | None) -> float:
    """The largest perimeter an image of ``shape`` can hold, in microns at ``pixels_per_um``.

    With no scale it is in pixels: a slider's reach only.
    """
    # Assume max perimeter will never need to be larger than 2x frame size border
    # The 2x is to provide margin for handling various curvatures
    max_perimeter_pixels = 2 * ((2 * shape[0]) + (2 * shape[1]))
    if pixels_per_um is None:
        return float(max_perimeter_pixels)
    return max_perimeter_pixels / pixels_per_um


def size_slider_stops(
    size: str, shape: tuple[int, ...], pixels_per_um: float | None
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """The slider positions, the sizes they stand for, and the reach, for the ``size`` filter.

    The scale is stretched at its low end, so the small sizes most methods
    bound get most of the slider's travel; the top stop is the image's
    largest size, kept above the stop below it. The reach is that largest
    size, which the top stop may be raised past.
    """
    if size == 'area':
        reach = area_filter_max(shape, pixels_per_um)
        return (0.0, 30.0, 60.0, 100.0), (0.0, 1000.0, 10000.0, max(reach, 10001.0)), reach
    reach = perimeter_filter_max(shape, pixels_per_um)
    return (0.0, 50.0, 100.0), (0.0, 100.0, max(reach, 101.0)), reach


def _refuse_size_filters_without_a_scale(method: CellCountMethod, name: str) -> None:
    """Refuse a count in pixels whose method bounds an area or a perimeter.

    The bounds are in microns; applying them to pixels would count a
    different set of objects and report it as the filtered one.
    """
    filters = method.filters
    for size, bounds in (('area', filters.area_um2), ('perimeter', filters.perimeter_um)):
        if bounds.min is not None or bounds.max is not None:
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

    The file's white is not returned: a count is measured against the depth,
    so a summed file is counted on the scale its values reach, not clipped
    at one frame's white.

    Raises:
        LiveFolderPathRefusedError: ``'path_not_found'``, nothing is at ``path``;
            ``'not_a_file'``, a folder is.
        PostProcessingRefusedError: reason ``unreadable``; the file is not
            an image, and the message names it.
    """
    path = path_utils.existing_file(path)
    try:
        image, significant_bits, _ = image_utils.load_pixels(path)
        return image, significant_bits
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


def with_pixels_per_um(method: CellCountMethod, pixels_per_um: float | str) -> CellCountMethod:
    """*method*, counting at *pixels_per_um*, refused if the count cannot use it.

    A string is read as the number it spells, as a person types it into a
    box -- a whole number as an integer, so the box can show back exactly what
    was typed; one that spells no number is refused naming it.

    Raises:
        PostProcessingRefusedError: reason ``method_invalid``.
    """
    if isinstance(pixels_per_um, str):
        # An emptied box is no override: count at each image's own scale.
        pixels_per_um = _number_typed(pixels_per_um) if pixels_per_um.strip() else None
    return dataclasses.replace(method, pixels_per_um=pixels_per_um)


def load_cell_count_method(path: str | os.PathLike) -> CellCountMethod:
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
        with open(path, encoding='utf-8') as f:
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
    return from_saved_form(saved, source=f'The cell-count method in {path}')


def save_cell_count_method(method: CellCountMethod, path: str | os.PathLike) -> None:
    """Write *method* to *path* as a cell-count method file.

    A missing folder is made where *path* sends it.

    Raises:
        LiveFolderPathRefusedError: ``'not_a_file'``, a folder is at
            *path*; ``'wrong_extension'``, its name does not end in
            ``.json`` (in any case), which is not added; ``'not_a_folder'``,
            a file is where its folder would be. Nothing is written.
    """
    path = path_utils.file_to_write(path, extension='.json')
    saved = python_values({**saved_form(method), 'metadata': dict(_CELL_COUNT_METHOD_METADATA)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(saved, f, indent=4)


def read_cell_count_results(path: str | os.PathLike) -> pd.DataFrame:
    """Read a cell-count results file into a table to graph.

    Every cell is read as the text written. A column whose every cell is a
    number is a column of numbers (whole numbers when every cell is one);
    any other column is text, so a cell such as ``NA`` is never read as a
    missing number. A ``time`` column of text is read cell by cell in the
    forms the count writes (``_results_times``), so it reads back as a
    datetime column when its times can be compared, and stays text when
    they cannot. Which columns can be plotted is ``results_axes``'s answer.

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
            times = _results_times(table['time'])
        except ValueError as e:
            raise refuse(f'its time column is not in the form the count writes ({e})') from e
        if times is not None:
            table['time'] = times
    if not results_axes(table)[1]:
        raise refuse('it has no column of numbers to plot')
    return table


def _results_time(cell: str) -> datetime.datetime | None:
    """One time cell: as the count writes it (ISO 8601, with its UTC offset
    when the image stated one), or as files written before wrote it
    (``RESULTS_TIME_FORMAT``); a blank cell states no time.

    Raises:
        ValueError: the cell is a time in neither form.
    """
    if not cell:
        return None
    try:
        return datetime.datetime.fromisoformat(cell)
    except ValueError:
        return datetime.datetime.strptime(cell, RESULTS_TIME_FORMAT)


def _results_times(cells: pd.Series) -> pd.Series | None:
    """A results file's time column as datetimes, or None when its times
    cannot be compared.

    Times compare only when every one states its UTC offset or none does
    (``image_utils.comparable_times``), so a column mixing the two, or
    stating no time at all, is no time axis. Times on one offset keep it, so
    a graph shows them as they were taken; times on more than one, a run
    across a daylight-saving change, are read in UTC, which keeps every
    instant and the spacing between them.

    Raises:
        ValueError: a cell is a time in neither form the count writes.
    """
    times = [_results_time(cell) for cell in cells]
    stated = [t for t in times if t is not None]
    if not stated or image_utils.comparable_times(stated) is None:
        return None
    offsets = {t.utcoffset() for t in stated}
    return pd.to_datetime(pd.Series(times, dtype=object), utc=len(offsets) > 1)


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

    def preview_cell_count(
        self,
        image: np.ndarray,
        method: CellCountMethod,
        significant_bits: int,
        *,
        pixels_per_um: float | None,
        name: str = 'The image',
    ) -> tuple[np.ndarray, dict]:
        """Count the cells in one image by *method*.

        *pixels_per_um* is the scale to measure at (``cell_count_scale``);
        None counts in pixels. *name* is what a refusal calls the image.

        Raises:
            PostProcessingRefusedError: reason ``no_scale``, with no scale
                the method bounds an area or a perimeter.
        """
        if pixels_per_um is None:
            _refuse_size_filters_without_a_scale(method, name)
        preview_images, cell_stats = self._cell_count.process_image(
            image=image,
            settings=saved_form(method),
            significant_bits=significant_bits,
            pixels_per_um=pixels_per_um,
        )

        return preview_images['filtered_contours'], cell_stats

    def apply_cell_count_to_folder(
        self,
        path: str | os.PathLike,
        method: CellCountMethod,
        *,
        tiling_grids: TilingGrids,
        channel: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Count the cells in the images directly in *path* and write
        ``results.csv`` there, with the method counted by beside it as
        ``results_method.json``.

        In a run's folder (or a folder inside one) only the images the run's
        record names are counted: its captures, by the run record, and what
        builds made from them, by the post-processing record. The record is
        read and never written: a count moves no record aside and starts
        none. *tiling_grids* are the installation's, which the run's protocol
        is read against. In a folder with no run record every image is
        counted but a Quick Enhance copy, told by its recipe beside it. With
        a *channel* (a release channel's key, which the caller has checked),
        only the images taken in it are counted (``_counted_as``: the
        channel the file states, else its record's, an old name read as the
        channel it names today); an image of another channel, or of none, is
        left out. Each image left out is named, with why, in ``left_out``
        and the message.
        Rows are in the order the images were taken (``frame_time``: the
        image's own time, else its record row's), then by name; when the
        times do not compare (one missing, or offsets mixed), the images with
        a time come first, by name, then the rest, by name.

        The method file is the one ``save_cell_count_method`` writes, so
        ``load_cell_count_method`` reads it back. A count never replaces an
        earlier one: the pair is written under the first stem free for both
        (``path_utils.free_stem``), ``results.csv`` and
        ``results_method.json``, else ``results_001.csv`` and
        ``results_001_method.json``, and so on, each written whole before it
        takes its name.

        Each image is measured at ``cell_count_scale``'s answer for it: the
        method's override, else the scale the image states, else pixels. The
        results say each row's area unit (``um2`` or ``px2``), so a folder of
        mixed images is one file. Each row's time is ``frame_time``'s, with
        its UTC offset when it states one, and is blank when there is none.
        Each row says where its image was taken (``_where_taken``): the
        well, channel and Z the image states, its channel else the record's,
        and its scan and tile from the record. The columns are
        ``RESULTS_COLUMNS``.

        Returns:
            ``results_path``; ``method_path``; ``counted``, how many images
            were analysed; ``left_out``, each image left out as its name and
            its ``LeftOutReason``; and ``message``, the outcome in words.

        Raises:
            PostProcessingRefusedError: ``protocol_data_unreadable``, the
                run's record, its protocol or its post-processing record
                cannot be read, or the protocol is gone; ``no_data``, no
                image in the folder could be analysed -- none there, none to
                count, or none readable; ``names_exhausted``, every numbered
                name for the results is taken.
            PostProcessingFailedError: The results or the method could not
                be written (with the results alone among its produced paths
                when only the method could not take its place), or some
                images could not be counted (unreadable, or a size filter on
                an image with no scale); the results hold every image that
                was, and the error names the rest.
        """
        listed = sorted(f for f in os.listdir(path) if image_utils.is_image(f))
        try:
            run = ProtocolPostProcessingHelper().read_run(pathlib.Path(path), tiling_grids)
            post_rows = None if run is None else run.post_rows()
        except ValueError as e:
            raise PostProcessingRefusedError(
                operation=CELL_COUNT_OPERATION,
                reason='protocol_data_unreadable',
                message=str(e),
            ) from e
        try:
            # Checked, not reserved: every count runs on the one
            # post-processing lane, so no two in this process choose at once.
            stem = path_utils.free_stem(
                pathlib.Path(path), _RESULTS_STEM, (_RESULTS_ENDING, _METHOD_ENDING)
            )
        except FileExistsError as e:
            raise PostProcessingRefusedError(
                operation=CELL_COUNT_OPERATION,
                reason='names_exhausted',
                message=f'{e}. Move or remove earlier results, then count again.',
            ) from e
        recorded, left_out = _images_to_count(pathlib.Path(path), listed, run, post_rows)
        filenames = list(recorded)
        results = []
        taken_at = {}
        not_counted = []

        for done, filename in enumerate(filenames, start=1):
            file_path = os.path.join(path, filename)
            try:
                image, significant_bits = read_cell_count_image(file_path)
                row = recorded[filename]
                facts = image_utils.read_file_record(file_path)
                if channel is not None and _counted_as(facts, row) != channel:
                    left_out.append((filename, LeftOutReason.OTHER_CHANNEL))
                    continue
                _, region_info = self.preview_cell_count(
                    image=image,
                    method=method,
                    significant_bits=significant_bits,
                    pixels_per_um=cell_count_scale(
                        method, image_utils.read_pixel_size_um(file_path)
                    ),
                    name=filename,
                )
                # The time the image states it was taken, else its record
                # row's, never the file system's: a copied folder's files
                # were all made at the copy.
                taken = image_utils.frame_time(
                    image_utils.stated_frame_time(facts),
                    None if row is None else row['Timestamp'],
                )
            except (PostProcessingRefusedError, LiveFolderPathRefusedError) as e:
                # Not an image after all, a folder named like one, or gone
                # since the folder was listed.
                logger.warning(f'[LVP Main  ] Not counting {filename}: {e}')
                not_counted.append(str(e))
                continue

            taken_at[filename] = taken
            results.append(
                {
                    'file': os.path.basename(filename),
                    **_where_taken(facts, row),
                    'time': taken.isoformat() if taken is not None else '',
                    'num_cells': region_info['summary']['num_regions'],
                    'total_object_area_px2': region_info['summary']['total_object_area_px2'],
                    'total_object_area_um2': region_info['summary']['total_object_area_um2'],
                    'total_object_intensity': region_info['summary']['total_object_intensity'],
                }
            )
            if on_progress is not None:
                on_progress(100 * done / len(filenames), f'{done}/{len(filenames)}: {filename}')

        left_out_words = '; '.join(
            f'{name} ({_LEFT_OUT_WORDS[reason]})' for name, reason in left_out
        )
        left_out_sentence = f' Left out: {left_out_words}.' if left_out else ''
        if not results:
            if not listed:
                message = 'No images were found in the selected folder.'
            elif not_counted:
                message = (
                    f'None of the {len(filenames)} image(s) in the folder could be counted.'
                    f'{left_out_sentence}'
                )
            else:
                message = f'No image in the folder is one to count: {left_out_words}.'
            raise PostProcessingRefusedError(
                operation=CELL_COUNT_OPERATION, reason='no_data', message=message
            )
        results = _in_taken_order(results, taken_at)

        results_file_path = os.path.join(path, f'{stem}{_RESULTS_ENDING}')
        method_file_path = os.path.join(path, f'{stem}{_METHOD_ENDING}')
        results_temp = os.path.join(path, f'.results.{uuid.uuid4().hex}.tmp.csv')
        method_temp = os.path.join(path, f'.results_method.{uuid.uuid4().hex}.tmp.json')

        def discard_temps() -> None:
            for temp in (results_temp, method_temp):
                with contextlib.suppress(OSError):
                    os.remove(temp)

        # Both are written whole before either takes its name, so a failed
        # write leaves no results without their method.
        try:
            with open(results_temp, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(RESULTS_COLUMNS)
                for record in results:
                    writer.writerow(record[column] for column in RESULTS_COLUMNS)
        except OSError as e:
            discard_temps()
            raise PostProcessingFailedError(
                operation=CELL_COUNT_OPERATION,
                missing=f'The results could not be written to {results_file_path} ({e.strerror}).',
            ) from e
        try:
            save_cell_count_method(method, method_temp)
        except OSError as e:
            discard_temps()
            raise PostProcessingFailedError(
                operation=CELL_COUNT_OPERATION,
                missing=f'The method could not be written to {method_file_path} ({e.strerror}).',
            ) from e

        # The results go first: failing there names nothing new.
        try:
            os.replace(results_temp, results_file_path)
        except OSError as e:
            discard_temps()
            raise PostProcessingFailedError(
                operation=CELL_COUNT_OPERATION,
                missing=f'The results could not be written to {results_file_path} ({e.strerror}).',
            ) from e
        try:
            os.replace(method_temp, method_file_path)
        except OSError as e:
            discard_temps()
            raise PostProcessingFailedError(
                operation=CELL_COUNT_OPERATION,
                missing=(
                    f'The method these results were counted by could not be written to'
                    f' {method_file_path} ({e.strerror}).'
                ),
                produced_paths=(results_file_path,),
                output_root=str(path),
                errors=not_counted,
            ) from e

        if not_counted:
            raise PostProcessingFailedError(
                operation=CELL_COUNT_OPERATION,
                missing=(
                    f'{len(not_counted)} of {len(filenames)} image(s) could not be counted.'
                    f'{left_out_sentence}'
                ),
                produced_paths=(results_file_path, method_file_path),
                output_root=str(path),
                errors=not_counted,
            )
        return {
            'results_path': results_file_path,
            'method_path': method_file_path,
            'counted': len(results),
            'left_out': left_out,
            'message': (
                f'Counted cells in {len(results)} image(s).{left_out_sentence}'
                f' Results: {results_file_path}; method: {method_file_path}'
            ),
        }


# A count's results and its method share one stem, numbered together.
_RESULTS_STEM = 'results'
_RESULTS_ENDING = '.csv'
_METHOD_ENDING = '_method.json'

# What the outcome calls each reason an image was left out.
_LEFT_OUT_WORDS = {
    LeftOutReason.NOT_IN_RECORD: "not in the run's record",
    LeftOutReason.QUICK_ENHANCE_COPY: 'a Quick Enhance copy',
    LeftOutReason.OTHER_CHANNEL: 'not of the channel counted',
}


def _images_to_count(
    folder: pathlib.Path,
    listed: list[str],
    run: RunRecord | None,
    post_rows: pd.DataFrame | None,
) -> tuple[dict[str, dict | None], list[tuple[str, LeftOutReason]]]:
    """The images in *folder* to count, each with its record row, and the
    rest with why each is left out.

    With *run*, an image is counted when the run record names it (its row
    is the capture's) or the post-processing record does (its row is the
    build's); its name in either is relative to the run's root. With none,
    each image is counted with no row, but a Quick Enhance copy.
    """
    recorded: dict[str, dict | None] = {}
    left_out: list[tuple[str, LeftOutReason]] = []
    if run is None:
        for name in listed:
            if quick_enhance.recipe_of(folder / name).is_file():
                left_out.append((name, LeftOutReason.QUICK_ENHANCE_COPY))
            else:
                recorded[name] = None
        return recorded, left_out
    built = (
        {}
        if post_rows is None
        else {pathlib.Path(row['Filepath']): row for row in post_rows.to_dict('records')}
    )
    for name in listed:
        relative = pathlib.Path(os.path.relpath(folder / name, run.root))
        row = run.raw_row(relative)
        if row is None:
            row = built.get(relative)
        if row is None:
            left_out.append((name, LeftOutReason.NOT_IN_RECORD))
        else:
            recorded[name] = row
    return recorded, left_out


def _in_taken_order(
    rows: list[dict], taken_at: Mapping[str, datetime.datetime | None]
) -> list[dict]:
    """*rows* in the order their images were taken, then by name; when the
    times do not compare, those with a time by name, then the rest by name."""
    if image_utils.comparable_times(taken_at[row['file']] for row in rows) is not None:
        return sorted(rows, key=lambda row: (taken_at[row['file']], row['file']))
    return sorted(rows, key=lambda row: (taken_at[row['file']] is None, row['file']))


def _image_channel(facts: dict | None, row: dict | None) -> str | None:
    """The channel an image was taken in, as it names it: the one its file
    states, else its record row's; None when neither states one."""
    stated = facts.get('channel') if facts is not None else None
    if stated is None and row is not None:
        return row['channel']
    return stated


def _counted_as(facts: dict | None, row: dict | None) -> str | None:
    """The release channel an image counts as: its channel, an old name
    (an old run's ``Blue``) read as the channel it names today."""
    channel = _image_channel(facts, row)
    return None if channel is None else channel_record.stored_key(channel)


def _where_taken(facts: dict | None, row: dict | None) -> dict:
    """Where an image was taken, as ``results.csv`` writes it.

    *facts* is the image's file record, None for a file that carries none
    (a JPG), which states nothing. ``well``, ``channel`` and ``z`` are what
    the image states it was captured at; an image that states no channel
    takes its record row's, which a step's capture never moves. The
    record's well and Z are the step's plan, which an inserted or moved step
    leaves empty or stale and autofocus moves from, so an image that states
    neither leaves them blank. ``scan`` (counted from 0, as the file names
    are) and ``tile`` are the record row's; both are blank with no record.
    """
    stated = facts if facts is not None else {}
    plate = stated.get('plate')
    well = plate.get('well_label') if plate is not None else None
    channel = _image_channel(facts, row)
    z = stated.get('z_pos_um')
    return {
        'well': '' if well is None else well,
        'channel': '' if channel is None else channel,
        'scan': '' if row is None else row['Scan Count'],
        'tile': '' if row is None else row['Tile'],
        'z': '' if z is None else z,
    }
