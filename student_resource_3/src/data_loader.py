from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

def get_data_loaders(train_dir, val_dir=None, batch_size=32):
    if not os.path.isdir(train_dir):
        raise ValueError(f"Training directory {train_dir} does not exist")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_dir and os.path.isdir(val_dir):
        val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Usage
train_loader, val_loader = get_data_loaders('path_to_training_data', 'path_to_validation_data')
