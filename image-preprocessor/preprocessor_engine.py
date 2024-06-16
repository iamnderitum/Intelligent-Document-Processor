### Preprocessing of the image, text localization

import cv2
import numpy as np
#from sklearn import preprocessing
#import sklearn
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

        (image_height, image_width) = image.shape[:2]
        center = (image_width // 2, image_height // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image_width, image_height),
                                 flags = cv2.INTER_CUBIC,
                                 borderMode = cv2.BORDER_REPLICATE)
        
        return rotated
    
    # Template Matching
    def match_template(image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    

class ImagePreProcessor:
    def __init__(self, image, process_name="Image Preprocessor") -> None:
        self.image = image

    def resize(self, image):
        image_height, image_width, color_channels = image.shape
        image_type = image.dtype
        print("Initial Image Information: ")
        print("-----------------------------\n")
        print(f"Height: {image_height}\n Width:{image_width} \n Color Channels: {color_channels}")
        print(f"Image Type: {image_type}")

        new_width = 400
        ration = new_width / image_width

        new_height = int(image_height * ration)
        new_dim = (new_width, new_height)
        print('Dimensions of the new image will be: \n height: {}'' \n width: {}'
            .format(new_dim[1], new_dim[0]))
        
        resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

        return resized
    
    def normalize(self, image, new_min=0, new_max=255):
        image_size, image_type, image_shape= image.size, image.dtype, image.shape
        print(f"Image Properties: \nShape: {image_shape} \n Size: {image_size}, \n Type: {image_type}\n\n")
        print("Image Array - Pixel\n",image[:1])

        image_array = np.asarray(image)
        
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = (image - min_val) / max_val - min_val
        normalized_image = normalized_image * (new_max - new_min) + new_min
        print("\n\n", normalized_image[:1])

        return normalized_image

        
