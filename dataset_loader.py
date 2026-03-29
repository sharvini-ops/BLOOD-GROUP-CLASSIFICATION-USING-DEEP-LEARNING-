import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# label mapping
label_map = {
    "A+":0,
    "A-":1,
    "B+":2,
    "B-":3,
    "AB+":4,
    "AB-":5,
    "O+":6,
    "O-":7
}

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class FingerprintDataset(Dataset):

    def __init__(self,folder):

        self.images=[]
        self.labels=[]

        for blood_group in os.listdir(folder):

            group_path=os.path.join(folder,blood_group)

            if blood_group in label_map:

                for file in os.listdir(group_path):

                    if file.lower().endswith((".bmp",".png",".jpg",".jpeg")):

                        self.images.append(os.path.join(group_path,file))
                        self.labels.append(label_map[blood_group])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):

        img=Image.open(self.images[index])
        img=transform(img)

        label=self.labels[index]

        return img,label