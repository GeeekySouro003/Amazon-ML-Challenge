import pytesseract
from PIL import Image
import os

def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist")
    image=Image.open(image_path)
    extracted_text=pytesseract.image_to_string(image)
    return extracted_text

def preprocess_ocr_text(text):
    return text.lower().strip()

if __name__ == "__main__":
    image_path = '../images/417SThj+SrL.jpg'
    text=extract_text_from_image(image_path)
    cleaned_text = preprocess_ocr_text(text)
    print(f"Extracted Text : {cleaned_text}")