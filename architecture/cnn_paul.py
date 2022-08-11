import torch.nn as nn

class Net(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        modules = []

        modules.append(
            nn.Conv2d(in_channels=img_dim[0], out_channels=16, kernel_size=3)
        )
        modules.append(nn.ReLU())
        
        modules.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3))
        modules.append(nn.ReLU())

        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(nn.Dropout())

        modules.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3))
        modules.append(nn.ReLU())
        
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(nn.Dropout(p=0.25))
        modules.append(nn.BatchNorm2d(64))

        modules.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3))
        modules.append(nn.ReLU())#

        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(nn.Dropout(p=0.25))
        modules.append(nn.BatchNorm2d(128))

        modules.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3))
        modules.append(nn.ReLU())
        
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))


        modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(nn.BatchNorm2d(256))
        
        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features=256, out_features=64))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=64, out_features=16))
        modules.append(nn.ReLU())

        self.model = nn.Sequential(*modules)

    def forward(self, imgs):
        return self.model(imgs).flatten()