
import cv2

class CellCount:

    def __init__(self):
        pass

    @staticmethod
    def preview_image(filepath, threshold):
        # img = cv2.imread(filepath)#, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Denoise
        denoised_img = cv2.fastNlMeansDenoising(img)

        # Threshold
        # th, threshold_img = cv2.threshold(denoised_img, 200, 255,cv2.THRESH_BINARY_INV)#|cv2.THRESH_OTSU)
        th, threshold_img = cv2.threshold(denoised_img, 200, 255,cv2.THRESH_BINARY_INV)#|cv2.THRESH_OTSU)

        # th, threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#|cv2.THRESH_OTSU)

        # Countours
        contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contoursImg = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contoursImg, contours, -1, (255,100,0), 3)

        return contoursImg, len(contours)