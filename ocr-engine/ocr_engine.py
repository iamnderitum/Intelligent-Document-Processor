### Character segmentation, Character Recognition, Post Processing
import re
import cv2
import pytesseract

from pytesseract import Output

class OcrEngine:
    def __init__(self, image_path, process_name="OCR-ENGINE") -> None:
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError("Image File Not Found Or Unable to Read")

    def getBoxText(self, image):
        img = cv2.imread(image)

        image_height, image_width, color_channels = img.shape

        boxes = pytesseract.image_to_boxes(img)

        for b in boxes.splitlines():
            b = b.split("")
            img = cv2.rectangle(img,
                                (int(b[1]), image_height, int(b[2])),
                                (int(b[3]), image_height - int(b[4])),
                                (0, 255, 0), 2)
            
        cv2.imshow("Image", img)
        cv2.waitKey()

    def getImageData(self, image_path):
        processed_image = cv2.imread(image_path)
    
        img_data = pytesseract.image_to_data(processed_image, output_type=Output.DICT)

        print(img_data.keys())

        n_boxes = len(img_data["text"])
        for i in range(n_boxes):
            if int(img_data["conf"][i]) > 60:
                (x, y, w, h) = (img_data["left"][i],
                                img_data["top"][i],
                                img_data["width"][i],
                                img_data["height"][i])
                
                img = cv2.rectangle(processed_image,
                                    (x, y),
                                    (x + w, y + h),
                                    (0, 255, 0),
                                    2 )
                
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        return img
    

class TemplateMatcher:
    def __init__(self, image_path, class_name="Template Matcher") -> None:
        self.image_path = image_path

    def dateMatcher(self, image_path):
        image = cv2.imread(image_path)
        img_data = pytesseract.image_to_data(image, output_type=Output.DICT)
        #keys = list(img_data.keys())

        date_pattern = r"^(0[1-9] | [12] [0-9] | 3[01]) / (0[1-9] | 1[012]) / (19|20)\d\d$ "
        id_number_pattern = r"\d"
        date_boxes = len(img_data["text"])

        # Initialize img with the original image
        img = image.copy()
        for i in range(date_boxes):
            if int(img_data["conf"][i]) > 60:
                if re.match(date_pattern, img_data["text"][i]):
                    (x ,y, w, h) = (img_data["left"][i],
                                    img_data["top"][i],
                                    img_data["width"][i],
                                    img_data["height"][i])
                    
                    img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if re.match(id_number_pattern, img_data["text"][i]):
                    (x ,y, w, h) = (img_data["left"][i],
                                    img_data["top"][i],
                                    img_data["width"][i],
                                    img_data["height"][i])
                    
                    img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Date Matcher", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img_data