from preprocessor_engine import PreProcessor

import cv2

image_path = "../data/images/ID_FRONT.png"
image = cv2.imread(image_path)


def basicImagePreprocess(image_path):
    
    image = cv2.imread(image_path)

    if image is None:
        raise  ValueError('Image File Not Found or Unable to Read')
    
    # Instance of PreProcessor
    preprocessor = PreProcessor(image)

    # Apply preprocessing steps GrayScale, Remove Noise, Threshold Image
    gray_image = preprocessor.getGrayScale(preprocessor.image)
    noise_removed_image = preprocessor.thresholding(gray_image)
    thresholded_image = preprocessor.opening(noise_removed_image)
    
    return thresholded_image

def fullImagePreprocess(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise  ValueError('Image File Not Found or Unable to Read')
    
    preprocessor = PreProcessor(image)

    # Apply preprocessing 
    gray_image = preprocessor.getGrayScale(preprocessor.image)
    noise_removed_image = preprocessor.removeNoise(gray_image)
    thresholded_image = preprocessor.thresholding(noise_removed_image)
    dilated_image = preprocessor.dilate(thresholded_image)
    eroded_image = preprocessor.erode(dilated_image)
    opened_image = preprocessor.opening(eroded_image)
    edge_detected_image = preprocessor.canny(opened_image)
    deskewed_image = preprocessor.deskew(edge_detected_image)
  
    return deskewed_image

def defaultImagePreprocess(input_image_path, output_image_path): 
    image = cv2.imread(input_image_path)
    if image is None:
        raise  ValueError('Image File Not Found or Unable to Read')
    
    preprocessor = PreProcessor(image)

    gray_image = preprocessor.getGrayScale(preprocessor.image)
    cv2.imwrite(output_image_path, gray_image)

    #noise_free_image = preprocessor.removeNoise(gray_image)

    #deskewed_image = preprocessor.deskew(noise_free_image)

    #canny_detected_image = preprocessor.canny(deskewed_image)


    return gray_image



if __name__ == "__main__":
    image_input_path = "../data/images/ID_FRONT.png"
    image_output_path = "../data/output-images/ID_FRONT.png"
    
    processing_level = str(input("Enter Image Processing Level: "))

    if processing_level == "Basic":
        basic_processed_image = basicImagePreprocess(image_path)
        cv2.imshow("Processed Image", basic_processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif processing_level == "Full":
        full_processed_image = fullImagePreprocess(image_path)
        cv2.imshow("Processed Image", full_processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif processing_level == "Default":
        default_processed_image = defaultImagePreprocess(image_input_path, image_output_path)
        cv2.imshow("Default Processed Image", default_processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Processing Level only Include Basic | Default | Full")

    # Display the final processed image
    #cv2.imshow("Processed Image", basic_processed_image)
    
   