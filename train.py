"""
train.py
Full training script for Xception deepfake detector.
Trains on the 140k Real and Fake Faces dataset.
Run this on a GPU machine or Colab — not on CPU.

Dataset: 140k Real and Fake Faces
Source: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
Download (run once in Colab):
    import os, json
    os.makedirs('/root/.kaggle', exist_ok=True)
    with open('/root/.kaggle/kaggle.json', 'w') as f:
        json.dump({"username": "RamadhanZome", "key": "your_kaggle_key"}, f)
    os.chmod('/root/.kaggle/kaggle.json', 0o600)
    os.system('kaggle datasets download -d xhlulu/140k-real-and-fake-faces --unzip')

Usage:
    python train.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.xception import Xception


#  Config 
DATASET_PATH = 'dataset/real_vs_fake/real-vs-fake'
MODEL_SAVE_PATH = 'models/best_xception.pth'
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dataset = datasets.ImageFolder(root=f'{DATASET_PATH}/train', transform=transform)
    valid_dataset = datasets.ImageFolder(root=f'{DATASET_PATH}/valid', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Classes: {train_dataset.classes}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(valid_loader)}")
    return train_loader, valid_loader


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")

    model     = Xception(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, valid_loader = get_dataloaders()

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0.0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc  = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc  = 100. * val_correct / val_total
        val_loss = val_running_loss / len(valid_loader)

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> Best model saved! Val Acc: {best_val_acc:.2f}%")

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    plot_metrics(train_losses, train_accs, val_losses, val_accs)


def plot_metrics(train_losses, train_accs, val_losses, val_accs):
    """Plot and save loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs,   label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('utilities/training_curves.png')
    plt.show()
    print("Training curves saved to utilities/training_curves.png")


if __name__ == '__main__':
    train()