import torch.nn as nn

class Net(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        modules = []

        # input_shape=(3,96,96), output_shape=[16,94,94]
        modules.append(nn.Conv2d(in_channels=img_dim[0], out_channels=16, kernel_size=3))
        modules.append(nn.ReLU())
        # input_shape=(16,94,94), output_shape=[16,47,47]
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(nn.BatchNorm2d(16))
        
        # input_shape=(16,47,47), output_shape=[32,45,45]
        modules.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3))
        modules.append(nn.ReLU())
        # input_shape=(32,45,45), output_shape=[32,22,22]
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # GlobalAveragePooling
        # input_shape=(32,22,22), output_shape=[32,1,1]
        modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(nn.BatchNorm2d(32))
        # input_shape=(32,1,1), output_shape=[32]
        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features=32, out_features=16))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=16, out_features=8))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=8, out_features=1))
        modules.append(nn.Sigmoid())

        self.model = nn.Sequential(*modules)

    
    def forward(self, imgs):
        return self.model(imgs).flatten()