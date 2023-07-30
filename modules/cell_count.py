
import cv2

import image_utils

class CellCount:

    def __init__(self):
        pass

    @staticmethod
    def preview_image(image, fluorescent_mode, threshold, size_min, size_max=None):
        gray_image = image_utils.rgb_image_to_gray(image=image)

        if fluorescent_mode is False:
            # Invert the image
            gray_image = cv2.bitwise_not(gray_image)

        # Denoise
        denoised_img = cv2.fastNlMeansDenoising(gray_image)

        # Threshold
        th, threshold_img = cv2.threshold(denoised_img, threshold, 255, cv2.THRESH_BINARY)

        # Countours
        contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # Contour Area filtering
        filtered_contours = []
        for contour in contours:    
            area = cv2.contourArea(contour)
            if (area >= size_min):
                if (size_max is None) or (area <= size_max):
                    filtered_contours.append(contour)

        contoursImg = image.copy()
        cv2.drawContours(contoursImg, filtered_contours, -1, (255,100,0), 3)

        return contoursImg, len(filtered_contours)
