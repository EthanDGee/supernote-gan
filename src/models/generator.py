import torch.nn as nn

from src.constants import NOTE_SIZE, LATENT_SIZE


class Generator(nn.Module):
    def __init__(
        self,
        latent_size: int = LATENT_SIZE,
        img_channels: int = 3,
        img_size: tuple = NOTE_SIZE,
    ):
        super(Generator, self).__init__()
        self.latent_dim = latent_size
        self.img_channels = img_channels
        self.img_size = img_size

        self.init_size = (4, 6)
        self.init_channels = 512

        # Initial dense layer
        self.fc = nn.Linear(
            latent_size, self.init_channels * self.init_size[0] * self.init_size[1]
        )

        # Reshape and batch norm
        self.reshape = lambda x: x.view(x.size(0), self.init_channels, *self.init_size)
        self.batch_norm = nn.BatchNorm2d(self.init_channels)

        # Transposed convolution layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                self.init_channels, 256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # 32x48 -> 64x96

        # Additional conv layers to reach target size
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Upsample to target size
        self.upsample = nn.Upsample(size=img_size, mode="bilinear", align_corners=False)

        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(16, img_channels, kernel_size=3, stride=1, padding=1), nn.Tanh()
        )

    def forward(self, z):
        # Dense layer
        x = self.fc(z)
        x = self.reshape(x)
        x = self.batch_norm(x)

        # Transposed conv layers
        x = self.deconv(x)
        # Additional conv layers
        x = self.conv(x)
        # Upsample to target size
        x = self.upsample(x)

        # Final layer
        x = self.final(x)

        return x
