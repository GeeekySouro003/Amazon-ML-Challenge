import os
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO

class ProductImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.valid_images = []

        print(f"DataFrame shape: {self.df.shape}")
        print("First few corrected image paths:")
        print(self.df['corrected_image_path'].head())
        print(f"Image directory: {self.img_dir}")

        for idx, row in self.df.iterrows():
            img_path = row['corrected_image_path']

            # Check if it's a URL or a local path
            if img_path.startswith('http'):
                try:
                    response = requests.get(img_path)
                    if response.status_code == 200:
                        self.valid_images.append(idx)
                    else:
                        print(f"Failed to access URL: {img_path}")
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching image from URL {img_path}: {e}")
            else:
                local_img_path = os.path.join(self.img_dir, img_path)
                if os.path.exists(local_img_path):
                    self.valid_images.append(idx)
                else:
                    print(f"Image not found: {local_img_path}")

            if idx % 1000 == 0:
                print(f"Processed {idx} images...")

        print(f"Total valid images found: {len(self.valid_images)}")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_idx = self.valid_images[idx]
        img_path = self.df.iloc[img_idx]['corrected_image_path']

        # Check if it's a URL or local path
        if img_path.startswith('http'):
            try:
                response = requests.get(img_path)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                print(f"Error loading image from URL {img_path}: {e}")
                raise
        else:
            local_img_path = os.path.join(self.img_dir, img_path)
            try:
                image = Image.open(local_img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image at {local_img_path}: {e}")
                raise

        if self.transform:
            image = self.transform(image)

        return image
