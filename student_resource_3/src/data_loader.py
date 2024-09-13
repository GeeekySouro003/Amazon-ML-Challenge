import torch
from torchvision import transforms
from PIL import Image

def get_transforms () : ## tranforms for resizing the image,normalizing positon etc
    return transforms.Compose([
        transforms.Resize((224,224)), ## size of 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),  
    ])
def load_image(image_path): ## preprocessing the image
    image=Image.open(image_path)
    transform = get_transforms()
    return transform(image)

##def preprocess_ocr_text(text):
  ##  text=text.lower().strip()
    ##return text
    
if __name__ == "__main__":
    img=load_image('../sample_images/test.jpg')
    print(img.shape)