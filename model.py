import torch.nn as nn


class Net(nn.Module):
    def __init__(self, mode):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        if mode == 'dropout' or mode == 'bothreg':
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 10)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(True),
                nn.Linear(512, 10)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
