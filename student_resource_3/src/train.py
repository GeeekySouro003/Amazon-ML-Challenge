import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import FeatureExtraction
from data_loader import get_transforms
from PIL import Image
import pandas as pd
import os

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 1])
        image = Image.open(img_name)
        label = self.annotations.iloc[idx, 3]  # Assuming entity_value is in column 3
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    # Define paths
    train_csv = 'dataset/train.csv'
    train_image_dir = 'images/train_images/'

    # Prepare data
    transform = get_transforms()
    train_dataset = CustomDataset(csv_file=train_csv, root_dir=train_image_dir, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # Initialize the model, criterion, and optimizer
    model = FeatureExtraction()
    criterion = torch.nn.CrossEntropyLoss()  # Use appropriate loss function based on your task
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

    # Save the model
    torch.save(model.state_dict(), 'output/cnn_model.pth')

if __name__ == "__main__":
    main()
