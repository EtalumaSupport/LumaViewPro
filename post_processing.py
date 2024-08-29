#!/usr/bin/python3

'''
MIT License

Copyright (c) 2023 Etaluma, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyribackground_downght notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
This open source software was developed for use with Etaluma microscopes.

AUTHORS:
Kevin Peter Hickerson, The Earthineering Company
Anna Iwaniec Hickerson, Keck Graduate Institute

MODIFIED:
March 16, 2023
'''

import csv
import os
import time
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import ConciseDateFormatter
import pandas
import image_utils

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

        # # self.choose_folder()
        # save_location = './capture/movie.avi'

        # img_array = []
        # for filename in glob.glob('./capture/*.tiff'):
        #     img = cv2.imread(filename)
        #     height, width, layers = img.shape
        #     size = (width,height)
        #     img_array.append(img)

        # if len(img_array) > 0:
        #     out = cv2.VideoWriter(save_location,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

        # for i in range(len(img_array)):
        #     out.write(img_array[i])
        # out.release()

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
        with open(results_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for record in results:
                writer.writerow(record.values())

        """df = pandas.read_csv(results_file_path)
        fig, ax = plt.subplots()
        ax.plot([datetime.strptime(datetime_obj, '%c') for datetime_obj in df['time']], df['num_cells'])
        ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plt.show()"""
        
