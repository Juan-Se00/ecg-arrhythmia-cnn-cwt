"""The custom CNN (paper §III.E, Eq. 10-12). Single source of truth."""
import torch
import torch.nn as nn


class ECGCNN(nn.Module):
    def __init__(
        self,
        dropout: float,
        filters: int,
        conv_layers: int,
        dense_units: int,
        img_size: tuple[int, int] = (128, 128),
        num_classes: int = 5,
    ):
        super().__init__()
        layers = []
        in_channels = 1
        for i in range(conv_layers):
            out_channels = filters * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout / 2),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *img_size)
            conv_out = self.conv_layers(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc(self.flatten(self.conv_layers(x)))
