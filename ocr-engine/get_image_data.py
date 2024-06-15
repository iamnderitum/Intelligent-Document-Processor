from ocr_engine import OcrEngine

import cv2

def imageData(image_path):
    engine = OcrEngine(image_path)
    image = cv2.imread(image_path)
    if image is None:
            raise  ValueError('Image File Not Found or Unable to Read')
    
    image_data = engine.getImageData(image_path)

    #image_keys = image_data.keys()
    #image_text = image_data.data["text"]
    
    return image_data.shape, image_data.dtype, image_data

if __name__ == "__main__":
      image_path = "../data/output-images/ID_FRONT.png"

      image_data = imageData(image_path)
      print(image_data)