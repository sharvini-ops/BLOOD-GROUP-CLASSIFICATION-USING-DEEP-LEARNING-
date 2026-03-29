import os
import cv2
import torch
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
from collections import Counter
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ---------------------------
# 📁 Create results folder
# ---------------------------
os.makedirs("results", exist_ok=True)

# ---------------------------
# 🧬 Classes
# ---------------------------
classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# ---------------------------
# 🔄 Transforms
# ---------------------------
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

gray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# ---------------------------
# 🧠 LeNet (grayscale)
# ---------------------------
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 53 * 53, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 8)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ---------------------------
# 📦 Load Models
# ---------------------------
def load_models():
    models_list = []

    try:
        vgg = models.vgg16(weights=None)
        vgg.classifier[6] = nn.Linear(4096, 8)
        vgg.load_state_dict(torch.load("saved_models/vgg16_model.pth", map_location='cpu'))
        vgg.eval()
        models_list.append(vgg)
        print("✅ VGG16 Loaded")
    except Exception as e:
        print("❌ VGG16 Error:", e)

    try:
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Linear(resnet.fc.in_features, 8)
        resnet.load_state_dict(torch.load("saved_models/resnet_model.pth", map_location='cpu'))
        resnet.eval()
        models_list.append(resnet)
        print("✅ ResNet Loaded")
    except Exception as e:
        print("❌ ResNet Error:", e)

    try:
        alex = models.alexnet(weights=None)
        alex.classifier[6] = nn.Linear(4096, 8)
        alex.load_state_dict(torch.load("saved_models/alexnet_model.pth", map_location='cpu'))
        alex.eval()
        models_list.append(alex)
        print("✅ AlexNet Loaded")
    except Exception as e:
        print("❌ AlexNet Error:", e)

    try:
        lenet = LeNet()
        lenet.load_state_dict(torch.load("saved_models/lenet_model.pth", map_location='cpu'))
        lenet.eval()
        models_list.append(lenet)
        print("✅ LeNet Loaded")
    except Exception as e:
        print("❌ LeNet Error:", e)

    return models_list

models_list = load_models()

# ---------------------------
# 📂 File Browser
# ---------------------------
def browse_file(file_type="all"):
    Tk().withdraw()
    if file_type == "bmp":
        return askopenfilename(filetypes=[("BMP files", "*.bmp")])
    else:
        return askopenfilename(filetypes=[("All files", "*.*")])

# ---------------------------
# 📷 Camera with Box
# ---------------------------
def camera_capture():
    cap = cv2.VideoCapture(0)
    print("Place finger in box and press SPACE")

    while True:
        ret, frame = cap.read()
        h, w, _ = frame.shape

        size = 250
        x1 = w//2 - size//2
        y1 = h//2 - size//2
        x2 = x1 + size
        y2 = y1 + size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == 32:
            roi = frame[y1:y2, x1:x2]
            path = "captured.jpg"
            cv2.imwrite(path, roi)
            break

    cap.release()
    cv2.destroyAllWindows()
    return path

# ---------------------------
# 🧠 Fingerprint Processing
# ---------------------------
def preprocess_fingerprint(img_path):
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    edges = cv2.Canny(gray, 50, 150)

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    path = "processed.jpg"
    cv2.imwrite(path, edges)
    return path

# ---------------------------
# 🔍 Prediction
# ---------------------------
def predict(img_path):
    img_rgb = Image.open(img_path).convert('RGB')
    img_gray = Image.open(img_path).convert('L')

    img_rgb = rgb_transform(img_rgb).unsqueeze(0)
    img_gray = gray_transform(img_gray).unsqueeze(0)

    results = []

    print("\n🔍 Model Outputs with Confidence:")

    for i, model in enumerate(models_list, 1):
        with torch.no_grad():
            if isinstance(model, LeNet):
                output = model(img_gray)
            else:
                output = model(img_rgb)

            prob = torch.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)

            label = classes[pred.item()]
            print(f"Model {i}: {label} ({conf.item():.2f})")
            results.append(label)

    return results

# ---------------------------
# 🧠 Majority Vote
# ---------------------------
def majority_vote(preds):
    return Counter(preds).most_common(1)[0][0]

# ---------------------------
# 🔄 Format Output
# ---------------------------
def format_output(label):
    base = label.replace('+','').replace('-','')
    return f"{base}+ / {base}-"

# ---------------------------
# 🖼️ Show + Save
# ---------------------------
def show_and_save(img_path, text):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300,300))

    out = cv2.copyMakeBorder(img,0,50,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
    cv2.putText(out, f"Prediction: {text}", (10,330),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("Result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_path = os.path.join("results","result_"+os.path.basename(img_path))
    cv2.imwrite(save_path, out)

    print("✅ Saved:", save_path)

# ---------------------------
# 🚀 MAIN LOOP
# ---------------------------
def main():
    while True:
        print("\n1. Choose from the Dataset")
        print("2. Choose from the Storage")
        print("3. Choose from the Camera Input")

        choice = input("Enter choice: ")

        if choice == '1':
            path = browse_file("bmp")

        elif choice == '2':
            path = browse_file("all")

        elif choice == '3':
            raw = camera_capture()
            path = preprocess_fingerprint(raw)

        else:
            print("Invalid choice")
            continue

        if not path:
            print("No file selected")
            continue

        preds = predict(path)
        final = format_output(majority_vote(preds))

        print("\n✅ Final Output:", final)

        show_and_save(path, final)

        if input("\nContinue? (yes/no): ").lower() != "yes":
            break

# ---------------------------
if __name__ == "__main__":
    main()