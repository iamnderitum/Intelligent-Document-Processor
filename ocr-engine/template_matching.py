from ocr_engine import TemplateMatcher

import cv2

def matchTemplate(image):
    template_matcher = TemplateMatcher(image)
    img = cv2.imread(image)

    if img is None:
            raise  ValueError('Image File Not Found or Unable to Read')
    
    matched_data = template_matcher.dateMatcher(image)
     
    image_text = matched_data["text"]
    return image_text

if __name__ == "__main__": 
      image = "../data/output-images/ID_FRONT.png"

      template_matched = matchTemplate(image)
      print(template_matched)