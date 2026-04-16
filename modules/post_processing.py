#!/usr/bin/python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import csv
import os
import time
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import ConciseDateFormatter
import pandas
import modules.image_utils as image_utils

from modules.cell_count import CellCount

class PostProcessing:

    SUPPORTED_IMAGE_TYPES = (
        '.jpg',
        '.jpeg',
        '.png',
        '.bmp',
        '.tif',
        '.tiff'
    )

    def __init__(self):
        self._cell_count = CellCount()


    def convert_to_avi(self, filepath):
        pass

    def stitch(self, filepath):
        pass

    
    def preview_cell_count(self, image, settings):
        preview_images, cell_stats = self._cell_count.process_image(
            image=image,
            settings=settings
        )

        return preview_images['filtered_contours'], cell_stats


    def get_num_images_in_folder(self, path):
        num_images = 0
        for filename in os.listdir(path):
            if filename.endswith(self.SUPPORTED_IMAGE_TYPES):
                num_images += 1
        
        return num_images


    def apply_cell_count_to_folder(self, path, settings):
        fields = ['file', 'time', 'num_cells', 'total_object_area (um2)', 'total_object_intensity']
        results = []

        for filename in os.listdir(path):
            if filename.endswith(self.SUPPORTED_IMAGE_TYPES):
                file_path = os.path.join(path, filename)
                image = image_utils.image_file_to_image(image_file=file_path)
                if image is None:
                    continue
                    
                _, region_info = self.preview_cell_count(
                    image=image,
                    settings=settings
                )

                time_created_raw = os.path.getctime(file_path)
                time_created = time.ctime(time_created_raw)

                results.append({
                    'filename': os.path.basename(filename),
                    'time': time_created,
                    'num_cells': region_info['summary']['num_regions'],
                    'total_object_area (um2)': region_info['summary']['total_object_area'],
                    'total_object_intensity': region_info['summary']['total_object_intensity'],
                })
                
                yield {
                    'filename': filename
                }

        results_file_path = os.path.join(path, 'results.csv')
        try:
            with open(results_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                for record in results:
                    writer.writerow(record.values())
        except OSError as e:
            logger.error(f'[LVP Main  ] Failed to write results CSV: {e}')

