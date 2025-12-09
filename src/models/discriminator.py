import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channels: int = 3, img_size: tuple = (351, 468)):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )  # 44x59 -> 22x30

        # Adaptive pool to get consistent size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Flatten and dense layers
        self.flatten_size = 256 * 8 * 8

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv(x)
        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten and dense layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
