### Preprocessing of the image, text localization

import cv2
import numpy as np

class PreProcessor:
    def __init__(self,image,  process_name="Image Pre-Processor") -> None:
        self.process_name = process_name
        self.image = image

    # get grayscale image
    def getGrayScale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Noise Removal
    def removeNoise(self, image):
        return cv2.medianBlur(image, 5)
    
    # Thresholding
    def thresholding(self, image):
        threshhold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Threshold : {threshhold}" )
        return threshhold[1]
    
    """
    DILATION adds pixels to the boundaries of objects in an image,
    while EROSION removes pixels on object boundaries. The number
    of pixels added or removed from the objects in an image depends 
    on the size and shape of the structuring element used to process
    the image.
    """
    
    # Dilation
    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations = 1)
    
    # Erosion
    def erode(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations = 1)
    
    # Opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    # Canny Edge detection
    def canny(self, image):
        return cv2.Canny(image, 100, 200)
    
    # Skew correction
    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)

        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags = cv2.INTER_CUBIC,
                                 borderMode = cv2.BORDER_REPLICATE)
        
        return rotated
    
    # Template Matching
    def match_template(image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)