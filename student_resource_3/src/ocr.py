import pytesseract
from PIL import Image
import os

def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist")
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text

def preprocess_ocr_text(text):
    return text.lower().strip()

if __name__ == "__main__":
    # Use the absolute path to the image file
    image_path = r'D:\Amazon ML Challenge\66e31d6ee96cd_student_resource_3\student_resource_3\images\test_images\test.jpg'
    try:
        text = extract_text_from_image(image_path)
        cleaned_text = preprocess_ocr_text(text)
        print(f"Extracted Text: {cleaned_text}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
