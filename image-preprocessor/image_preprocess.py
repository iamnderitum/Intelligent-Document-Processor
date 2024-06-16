import cv2
import os
from preprocessor_engine import ImagePreProcessor

def imagePreprocessor(image, output_directory, output_filename):
    img = cv2.imread(image)

    if img is None:
        raise ValueError("Image Not Found")
    
    image_preprocessor = ImagePreProcessor(img)
    image_resized = image_preprocessor.resize(image_preprocessor.image)

    output_path = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_path, image_resized)
    return image_resized

def normalizeImage(image):
    img = cv2.imread(image)

    image_preprocessor = ImagePreProcessor(img)
    normalized_image = image_preprocessor.normalize(image_preprocessor.image)

    return normalized_image


image = "../data/images/title_deed.png"
#output_directory = "../data/output-images/"
#output_filename = "title_deed_resized.png"

#preprocessed_image = imagePreprocessor(image, output_directory, output_filename)
#cv2.imshow("Preprocessed Image", preprocessed_image)
#cv2.waitKey(0)

#cv2.destroyAllWindows()
ni = normalizeImage(image)