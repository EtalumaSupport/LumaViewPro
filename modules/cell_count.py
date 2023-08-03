
import math

import cv2
import scipy
import skimage
import numpy as np

import image_utils

from modules.settings_transformer import SettingsTransformer

class CellCount:

    def __init__(self):
        self._settings_transformer = SettingsTransformer(to_direction='digital')
        

    def _get_region_info(self, labeled_mask, intensity_image, pixels_per_um):

        def _sphericity(area, perimeter):
            if perimeter==0:
                return -1
            return (4 * math.pi * area) / (perimeter**2)
            
        regions = skimage.measure.regionprops(labeled_mask, intensity_image=intensity_image, extra_properties=(_sphericity,))
        
        region_prop_params = [
            'Area',
            # 'equivalent_diameter',
            'Perimeter',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity',
        ]

        region_info = {}
        for region_idx, region in enumerate(regions):
            
            region_param_info = {}
            for param in region_prop_params:
                if (param == 'Area'):
                    val = round(region[param]/(pixels_per_um**2),2)
                    units = 'μm^2'
                elif (param == 'Perimeter'):
                    val = round(region[param]/(pixels_per_um),2)
                    units = 'μm'
                elif (param == 'equivalent_diameter'):
                    val = round(region[param]/(pixels_per_um),2)
                    units = 'μm'
                elif (param in ['MinIntensity','MeanIntensity','MaxIntensity']):
                    val = np.round(region[param]/(255/100),2)
                    units = '%'
                else:
                    val = region[param]
                    units = None
                    
                region_param_info[param.lower()] = {
                        'val': val,
                        'units': units
                }

            region_param_info['sphericity'] = {
                'val': _sphericity(area=region_param_info['area']['val'], perimeter=region_param_info['perimeter']['val']),
                'units': None
            }
            
            region_info[region_idx] = region_param_info

        return {
            'summary': {
                'num_regions': len(region_info)
            },
            'regions': region_info
        }


    def _filter_regions(self, region_info, contours, settings):
    
        filtered_regions = {}
        filtered_contours = []

        def _within_bounds(subject, subject_key, criteria, criteria_key):
            if (criteria[criteria_key]['min'] is not None) and (subject[subject_key]['val'] < criteria[criteria_key]['min']):
                return False

            if (criteria[criteria_key]['max'] is not None) and (subject[subject_key]['val'] > criteria[criteria_key]['max']):
                return False

            return True
            

        for (region_idx, region), contour in zip(region_info['regions'].items(), contours):

            if not _within_bounds(region, 'area', settings, 'area'):
                continue

            if not _within_bounds(region, 'perimeter', settings, 'perimeter'):
                continue

            if not _within_bounds(region, 'sphericity', settings, 'sphericity'):
                continue

            if not _within_bounds(region, 'minintensity', settings['intensity'], 'min'):
                continue

            if not _within_bounds(region, 'meanintensity', settings['intensity'], 'mean'):
                continue

            if not _within_bounds(region, 'maxintensity', settings['intensity'], 'max'):
                continue
                
            filtered_regions[region_idx] = region
            filtered_contours.append(contour)

        filtered_region_info = {
            'summary': {
                'num_regions': len(filtered_regions)
            },
            'regions': filtered_regions
        }
        
        return filtered_region_info, filtered_contours


    def process_image(self, image, settings):

        digital_settings = self._settings_transformer.transform(settings=settings)

        gray_image = image_utils.rgb_image_to_gray(image=image)

        if digital_settings['context']['fluorescent_mode'] is False:
            # Invert the image
            gray_image = cv2.bitwise_not(gray_image)

        # Denoise
        denoised_img = cv2.fastNlMeansDenoising(gray_image)

        # Threshold
        th, threshold_img = cv2.threshold(denoised_img, digital_settings['segmentation']['parameters']['threshold'], 255, cv2.THRESH_BINARY)

        # Countours
        contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_img = gray_image.copy()
        cv2.drawContours(contours_img, contours, -1, (255,100,0), 3)


        # Labeling
        label_map = np.zeros_like(contours_img, dtype = 'uint8')
        for i, contour in enumerate(contours):
            label_map = cv2.drawContours(label_map, [contour], -1, i+1, -1)


        region_info = self._get_region_info(
            labeled_mask=label_map,
            intensity_image=gray_image,
            pixels_per_um=digital_settings['context']['pixels_per_um']
        )

        filtered_region_info, filtered_contours = self._filter_regions(
            region_info=region_info,
            contours=contours,
            settings=settings['filters']
        )

        filtered_contours_img_for_display = image.copy()
        cv2.drawContours(filtered_contours_img_for_display, filtered_contours, -1, (255,100,0), 3)

        return filtered_contours_img_for_display, filtered_region_info

    
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
