import torch
import torch.nn as nn

class LeNet(nn.Module):

    def __init__(self):

        super(LeNet,self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Linear(16*53*53,120),
            nn.ReLU(),

            nn.Linear(120,84),
            nn.ReLU(),

            nn.Linear(84,8)
        )

    def forward(self,x):

        x=self.conv(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)

        return x