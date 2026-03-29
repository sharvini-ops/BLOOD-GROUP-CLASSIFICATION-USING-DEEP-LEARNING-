import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------------------------
# 📁 Paths
# ---------------------------
DATA_DIR = "dataset"
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# 🔄 Transforms (IMPORTANT)
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ---------------------------
# 📦 Dataset
# ---------------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Classes:", dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 🧠 Train Function
# ---------------------------
def train_model(model, name, epochs=10):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        running_loss = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"{name} Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), f"{SAVE_DIR}/{name}_model.pth")
    print(f"✅ {name} model saved\n")

# ---------------------------
# 🚀 VGG16
# ---------------------------
vgg = models.vgg16(weights='IMAGENET1K_V1')
vgg.classifier[6] = nn.Linear(4096, 8)
train_model(vgg, "vgg16")

# ---------------------------
# 🚀 ResNet18
# ---------------------------
resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet.fc = nn.Linear(resnet.fc.in_features, 8)
train_model(resnet, "resnet")

# ---------------------------
# 🚀 AlexNet
# ---------------------------
alex = models.alexnet(weights='IMAGENET1K_V1')
alex.classifier[6] = nn.Linear(4096, 8)
train_model(alex, "alexnet")