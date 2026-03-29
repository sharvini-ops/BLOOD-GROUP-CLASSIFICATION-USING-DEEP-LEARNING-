import torch.nn as nn
from torchvision import models

class VGG16Model(nn.Module):

    def __init__(self):

        super(VGG16Model,self).__init__()

        self.model=models.vgg16(pretrained=False)

        self.model.classifier[6]=nn.Linear(4096,8)

    def forward(self,x):

        return self.model(x)