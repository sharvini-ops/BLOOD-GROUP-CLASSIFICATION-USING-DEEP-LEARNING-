import torch
import torch.nn as nn
from torchvision import models

class AlexNetModel(nn.Module):

    def __init__(self):

        super(AlexNetModel,self).__init__()

        self.model=models.alexnet(pretrained=False)

        self.model.classifier[6]=nn.Linear(4096,8)

    def forward(self,x):

        return self.model(x)