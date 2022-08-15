import torch.nn as nn

class MlpBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        modules = []

        modules.append(nn.Linear(in_features=dim, out_features=hidden_dim))
        modules.append(nn.GELU())
        modules.append(nn.Linear(in_features=hidden_dim, out_features=dim))
        modules.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim, dropout=0.0):
        super().__init__()   
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)

        self.token_mixing_block = MlpBlock(num_patches, token_dim, dropout=dropout)
        self.channel_mixing_block = MlpBlock(dim, channel_dim, dropout=dropout)

    def forward(self, x):
        _x = self.layer_norm_1(x)
        _x = _x.permute(0,2,1)
        _x = self.token_mixing_block(_x)
        _x = _x.permute(0,2,1)
        _x = _x + x
        _x = _x + self.channel_mixing_block(self.layer_norm_2(_x))
        return _x

class Net(nn.Module):
    def __init__(self, img_dim):
        super().__init__()   
        self.batch_size = img_dim[0]
        self.img_size = img_dim[1]
        self.patch_size = 16
        self.num_classes = 1
        self.num_blocks = 8
        self.dim = 512
        self.token_dim = 256
        self.channel_dim = 2048
        self.dropout = 0.2

        # Assert square images
        assert img_dim[1] == img_dim[2]

        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.create_patches = nn.Conv2d(self.batch_size, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.mixer_blocks = nn.Sequential(
            *[MixerBlock(self.dim, self.num_patches, self.token_dim, self.channel_dim, self.dropout) for _ in range(self.num_blocks)])
        self.layer_norm = nn.LayerNorm(self.dim)
        self.classifier = nn.Linear(self.dim, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.create_patches(x)
        b, c, _, _ = x.shape
        x = x.view(b,c,-1).transpose(1,2)
        x = self.mixer_blocks(x)
        x = self.layer_norm(x)
        # Global Average Pooling
        x = x.mean(dim=1)
        x = self.classifier(x)
        x = self.sigmoid(x)
        x = x.flatten()
        return x

