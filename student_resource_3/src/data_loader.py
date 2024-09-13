import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Optional: Normalize as per standard practice
    ])

def get_data_loaders(train_dir, val_dir=None, batch_size=32):
    if not os.path.isdir(train_dir):
        raise ValueError(f"Training directory {train_dir} does not exist")
    
    transform = get_transforms()  # Use the get_transforms function for both training and validation datasets

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_dir and os.path.isdir(val_dir):
        val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Usage in the script
train_loader, val_loader = get_data_loaders('D:/Amazon ML Challenge/66e31d6ee96cd_student_resource_3/student_resource_3/images/train_images')
