import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------------
# SETTINGS
# -----------------------------
DATA_DIR = "dataset"       # folder containing class subfolders
BATCH_SIZE = 4             # small for fast CPU run
EPOCHS = 10                 # reduced for quick test
NUM_CLASSES = 8
USE_PRETRAINED = False     # no pretrained weights → faster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # convert BMP to 3 channels
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# DATASET & DATALOADER
# -----------------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Classes found:", dataset.classes)

# -----------------------------
# MODEL SELECTION
# -----------------------------
MODEL_NAME = "vgg16"   # options: "alexnet", "resnet", "vgg16"

if MODEL_NAME == "alexnet":
    weights = models.AlexNet_Weights.DEFAULT if USE_PRETRAINED else None
    model = models.alexnet(weights=weights)
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)

elif MODEL_NAME == "resnet":
    weights = models.ResNet18_Weights.DEFAULT if USE_PRETRAINED else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

elif MODEL_NAME == "vgg16":
    weights = models.VGG16_Weights.DEFAULT if USE_PRETRAINED else None
    model = models.vgg16(weights=weights)
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)

else:
    raise ValueError("Unsupported MODEL_NAME. Choose alexnet, resnet, or vgg16.")

model = model.to(device)

# -----------------------------
# LOSS & OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -----------------------------
# TRAINING LOOP
# -----------------------------
print(f"\nTraining {MODEL_NAME}...\n")

for epoch in range(EPOCHS):
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("saved_models", exist_ok=True)
save_path = f"saved_models/{MODEL_NAME}_model.pth"
torch.save(model.state_dict(), save_path)

print(f"\n✅ Model saved at: {save_path}")