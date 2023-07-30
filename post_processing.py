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

from kivy.graphics.texture import Texture

import image_utils

from modules.cell_count import CellCount

class PostProcessing:

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

    
    def preview_cell_count(self, image, fluorescent_mode, threshold, size_min, size_max):
        """
        Takes a file path, processes an image for cell counting, and returns
        a Kivy texture preview of the processed image
        """
        preview_img, num_cells = self._cell_count.preview_image(
            image=image,
            fluorescent_mode=fluorescent_mode,
            threshold=threshold,
            size_min=size_min,
            size_max=size_max
        )

        return preview_img, num_cells


    def apply_cell_count_to_folder(self, path, settings):
        print(f"Apply cell counts to folder: {path} with settings {settings}")

        SUPPORTED_IMAGE_TYPES = (
            '.jpg',
            '.jpeg',
            '.png',
            '.bmp',
            '.tif',
            '.tiff'
        )

        fields = ['file', 'num_cells']
        results = []

        for file_name in os.listdir(path):
            if file_name.endswith(SUPPORTED_IMAGE_TYPES):
                file_path = os.path.join(path, file_name)
                image = image_utils.image_file_to_image(image_file=file_path)
                _, num_cells = self.preview_cell_count(
                    image=image,
                    fluorescent_mode=settings['fluorescent_mode'],
                    threshold=settings['threshold'],
                    size_min=settings['size']['min'],
                    size_max=settings['size']['max']
                )
                results.append({
                    'file': os.path.basename(file_name),
                    'num_cells': num_cells
                })


        results_file_path = os.path.join(path, 'results.csv')
        with open(results_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for record in results:
                writer.writerow(record.values())
