
import cv2
import scipy
import skimage
import numpy as np

import image_utils

class CellCount:

    def __init__(self):
        pass


    def _transform_intensity_settings(self, settings):

        scale_ratio = 255/100
      
        return {
            'min': {
                'min': settings['min']['min']*scale_ratio,
                'max': settings['min']['max']*scale_ratio,
            },
            'mean': {
                'min': settings['mean']['min']*scale_ratio,
                'max': settings['mean']['max']*scale_ratio,
            },
            'max': {
                'min': settings['max']['min']*scale_ratio,
                'max': settings['max']['max']*scale_ratio,
            }
        }

    def _transform_perimeter_settings(self, settings, pixels_to_um):
        return {
            'min': settings['min'] / pixels_to_um,
            'max': settings['max'] / pixels_to_um
        }

    def _transform_area_settings(self, settings, pixels_to_um):
        return {
            'min': settings['min'] / (pixels_to_um**2),
            'max': settings['max'] / (pixels_to_um**2)
        }

    def _transform_threshold_settings(self, setting):
        scale_ratio = 255/100
        return setting * scale_ratio
        

    def _transform_settings(self, settings):
        transformed_settings = {
            'fluorescent_mode': settings['fluorescent_mode'],
            'pixels_to_um': settings['pixels_to_um'],
            'intensity': self._transform_intensity_settings(settings['intensity']),
            'perimeter': self._transform_perimeter_settings(settings['perimeter'], settings['pixels_to_um']),
            'threshold': self._transform_threshold_settings(settings['threshold']),
            'area': self._transform_area_settings(settings['area'], settings['pixels_to_um'])
        }
        return transformed_settings

    
   



    def process_image(self, image, settings):

        print(f"Before settings: {settings}")
        settings = self._transform_settings(settings)
        print(f"Transformed settings: {settings}")

        gray_image = image_utils.rgb_image_to_gray(image=image)

        if settings['fluorescent_mode'] is False:
            # Invert the image
            gray_image = cv2.bitwise_not(gray_image)

        # Denoise
        denoised_img = cv2.fastNlMeansDenoising(gray_image)

        # Threshold
        th, threshold_img = cv2.threshold(denoised_img, settings['threshold'], 255, cv2.THRESH_BINARY)

        # Countours
        contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contour Area filtering
        filtered_contours = []
        for contour in contours:    
            area = cv2.contourArea(contour)
            if (area >= settings['area']['min']):
                if (settings['area']['max'] is None) or (area <= settings['area']['max']):
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
    #         pixels_to_um=settings['pixels_to_um']
    #     )

    #     return labeled_img, region_stats

    
    # @staticmethod
    # def _extract_region_properties(labeled_mask, intensity_image, pixels_to_um):
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
    #                 val = f"{round(region[prop]*pixels_to_um**2,2)} um^2"
    #             elif (prop == 'Perimeter'):
    #                 val = f"{round(region[prop]*pixels_to_um,2)} um"
    #             elif (prop == 'equivalent_diameter'):
    #                 val = f"{round(region[prop]*pixels_to_um,2)} um"
    #             elif (prop in ['MinIntensity','MeanIntensity','MaxIntensity']):
    #                 val = np.round(region[prop],2)
    #             else:
    #                 # val = round(region[prop],2)
    #                 val = region[prop]
                    
    #             print(f" - {prop}: {val}")

    #     return {
    #         'num_cells': len(regions)
    #     }
