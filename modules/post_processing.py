#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import contextlib
import csv
import os
import time
import uuid

import modules.image_utils as image_utils

from lvp_logger import logger
from modules.cell_count import CellCount
from modules.exceptions import PostProcessingFailedError, PostProcessingRefusedError
from modules.protocol_post_processor import ProgressCallback

# The operation's name as a person reads it, in its refusals and failures.
CELL_COUNT_OPERATION = 'Cell Count'


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

    def preview_cell_count(self, image, settings, significant_bits: int):
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
            # One read returns pixels AND their payload depth together, so a
            # right-aligned 12-bit TIFF scales to 8-bit by its true depth and
            # the two can never be read out of sync.
            try:
                image, significant_bits = image_utils.load_pixels(file_path)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f'[LVP Main  ] Skipping unreadable image {filename}: {e}')
                unreadable.append(f'{filename}: {e}')
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
