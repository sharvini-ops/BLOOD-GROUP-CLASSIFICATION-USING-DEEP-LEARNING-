import torch.nn as nn
from torchvision import models

class ResNetModel(nn.Module):

    def __init__(self):

        super(ResNetModel,self).__init__()

        self.model=models.resnet18(pretrained=False)

        self.model.fc=nn.Linear(self.model.fc.in_features,8)

    def forward(self,x):

        return self.model(x)