import torch
import torch.nn as nn

class Discriminator3D(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(Discriminator3D, self).__init__()
        self.main = nn.Sequential(
            # input is 160 x 64 x 64
            nn.Conv3d(1, 32, 3, stride=2, padding=1, bias=False), # output size: 80 x 32 x 32
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout_rate),
            
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False), # output size: 40 x 16 x 16
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout_rate),

            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False), # output size: 20 x 8 x 8
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout_rate),

            nn.Conv3d(128, 256, 3, stride=2, padding=1, bias=False), # output size: 10 x 4 x 4
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout_rate),

            nn.Conv3d(256, 512, 3, stride=2, padding=1, bias=False), # output size: 5 x 2 x 2
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout_rate),

            # Final layer to output a single value (real or fake)
            nn.Conv3d(512, 1, (5, 1, 1), stride=1, padding=0, bias=False), # output size: 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).reshape(-1, 1)
