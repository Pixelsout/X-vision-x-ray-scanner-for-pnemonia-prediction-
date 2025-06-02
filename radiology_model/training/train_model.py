import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from radiology_model.training.model import get_model

def get_data_loaders(data_dir, batch_size=32, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    # Handle class imbalance using weights
    targets = [label for _, label in train_dataset]
    class_count = Counter(targets)
    class_weights = [1.0 / class_count[label] for label in targets]
    sampler = WeightedRandomSampler(class_weights, len(class_weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.classes

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'chest_xray'
    train_loader, val_loader, class_names = get_data_loaders(data_dir)

    model = get_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter()  # TensorBoard

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total * 100
        val_accuracy = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    torch.save(model.state_dict(), 'radiology_model.pth')
    print("✅ Training complete. Model saved as 'radiology_model.pth'.")

if __name__ == "__main__":
    train_model()
