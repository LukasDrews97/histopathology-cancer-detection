import torch.nn as nn

class Net(nn.Module):
    def __init__(self, img_dim, l1=120, l2=84):
        super().__init__()
        modules = []

        # input_shape=[3,96,96], output_shape=[16,94,94]
        modules.append(nn.Conv2d(in_channels=img_dim[0], out_channels=16, kernel_size=3))
        modules.append(nn.ReLU())
        # input_shape=[16,94,94], output_shape=[32,92,92]
        modules.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout2d(0.2))

        # input_shape=[32,92,92], output_shape=[32,46,46]
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(nn.BatchNorm2d(32))
        
        # input_shape=[32,46,46], output_shape=[64,44,44]
        modules.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3))
        modules.append(nn.ReLU())
        # input_shape=[64,44,44], output_shape=[128,42,42]
        modules.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3))
        modules.append(nn.ReLU())
        
        # input_shape=[128,42,42], output_shape=[128,21,21]
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(nn.Dropout2d(0.2))
        # GlobalAveragePooling
        # input_shape=[128,21,21], output_shape=[128,1,1]
        modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(nn.BatchNorm2d(128))
        
        # input_shape=[128,1,1], output_shape=[128]
        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features=128, out_features=64))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=64, out_features=32))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.2))
        modules.append(nn.Linear(in_features=32, out_features=16))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(in_features=16, out_features=1))
        modules.append(nn.Sigmoid())

        self.model = nn.Sequential(*modules)
    
    def forward(self, imgs):
        return self.model(imgs).flatten()