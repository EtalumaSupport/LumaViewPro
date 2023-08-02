
import cv2

import image_utils

class CellCount:

    def __init__(self):
        pass

    @staticmethod
    def preview_image(image, settings): #fluorescent_mode, threshold, area_min, area_max=None):
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

        return contoursImg, len(filtered_contours)
