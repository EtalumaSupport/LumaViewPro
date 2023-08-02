
import cv2
import scipy
import skimage
import numpy as np

import image_utils

from modules.settings_transformer import SettingsTransformer

class CellCount:

    def __init__(self):
        self._settings_transformer = SettingsTransformer()
        

    def process_image(self, image, settings):

        print(f"Before settings: {settings}")
        settings = self._settings_transformer.transform_to_digital(settings=settings)
        # settings = self._transform_settings(settings)
        print(f"Transformed settings: {settings}")

        gray_image = image_utils.rgb_image_to_gray(image=image)

        if settings['context']['fluorescent_mode'] is False:
            # Invert the image
            gray_image = cv2.bitwise_not(gray_image)

        # Denoise
        denoised_img = cv2.fastNlMeansDenoising(gray_image)

        # Threshold
        th, threshold_img = cv2.threshold(denoised_img, settings['segmentation']['parameters']['threshold'], 255, cv2.THRESH_BINARY)

        # Countours
        contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contour Area filtering
        filtered_contours = []
        for contour in contours:    
            area = cv2.contourArea(contour)
            if (area >= settings['filters']['area']['min']):
                if (settings['filters']['area']['max'] is None) or (area <= settings['filters']['area']['max']):
                    filtered_contours.append(contour)

        # contours = filtered_contours.copy()
        # Perimeter filtering
        # for contour in contours:



        contoursImg = image.copy()
        cv2.drawContours(contoursImg, filtered_contours, -1, (255,100,0), 3)

        region_stats = {
            'num_cells': len(filtered_contours)
        }

        return contoursImg, region_stats

    
    # def process_image(self, image, settings):

    #     if 'crop_bottom' in settings:
    #         image = image[:-1*settings['crop_bottom'],:,:]

    #     gray_image = image_utils.rgb_image_to_gray(image=image)

    #     if settings['fluorescent_mode'] is False:
    #         gray_image = cv2.bitwise_not(gray_image)

    #     denoised_img = cv2.fastNlMeansDenoising(gray_image)

    #     ret, threshold_img = cv2.threshold(denoised_img, settings['threshold'], 255, cv2.THRESH_BINARY)

    #     kernel = np.ones((3,3), np.uint8)
    #     eroded_img = cv2.erode(threshold_img, kernel, iterations=1)
    #     dilated_img = cv2.dilate(eroded_img, kernel, iterations=1)

    #     mask = dilated_img == 255
    #     structure = [
    #         [1,1,1],
    #         [1,1,1],
    #         [1,1,1]
    #     ]

    #     labeled_mask, num_labels = scipy.ndimage.label(mask, structure=structure)
    #     labeled_img = skimage.color.label2rgb(labeled_mask, bg_label=0)

    #     region_stats = self._extract_region_properties(
    #         labeled_mask=labeled_mask,
    #         intensity_image=gray_image,
    #         pixels_per_um=settings['pixels_per_um']
    #     )

    #     return labeled_img, region_stats

    
    # @staticmethod
    # def _extract_region_properties(labeled_mask, intensity_image, pixels_per_um):
    #     regions = skimage.measure.regionprops(labeled_mask, intensity_image=intensity_image)

    #     prop_list = [
    #         'Area',
    #         # 'equivalent_diameter',
    #         'Perimeter',
    #         'MinIntensity',
    #         'MeanIntensity',
    #         'MaxIntensity'
    #     ]

    #     for region_idx, region in enumerate(regions):
    #         print(f"Region {region_idx}")
    #         for prop in prop_list:
    #             if (prop == 'Area'):
    #                 val = f"{round(region[prop]*pixels_per_um**2,2)} um^2"
    #             elif (prop == 'Perimeter'):
    #                 val = f"{round(region[prop]*pixels_per_um,2)} um"
    #             elif (prop == 'equivalent_diameter'):
    #                 val = f"{round(region[prop]*pixels_per_um,2)} um"
    #             elif (prop in ['MinIntensity','MeanIntensity','MaxIntensity']):
    #                 val = np.round(region[prop],2)
    #             else:
    #                 # val = round(region[prop],2)
    #                 val = region[prop]
                    
    #             print(f" - {prop}: {val}")

    #     return {
    #         'num_cells': len(regions)
    #     }
