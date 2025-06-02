from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_dir='chest_xray', batch_size=32, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
