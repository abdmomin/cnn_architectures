import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(ResidualBlock, self).__init__()

        self.identity_downsample = identity_downsample
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 4,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels * 4)

        )

    def forward(self, x):
        identity = x.clone()
        x = self.resblock(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        return F.relu(torch.add(x, identity))


class ResNet(nn.Module):
    def __init__(self, layers, image_channels, out_classes):
        super(ResNet, self).__init__()
        assert len(layers) == 4
        self.in_channels = 64

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=64,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.resnet_block = nn.Sequential(
            self.make_layers(residual_blocks=layers[0], out_channels=64, stride=1),
            self.make_layers(residual_blocks=layers[1], out_channels=128, stride=2),
            self.make_layers(residual_blocks=layers[2], out_channels=256, stride=2),
            self.make_layers(residual_blocks=layers[3], out_channels=512, stride=2),

            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(in_features=512 * 4, out_features=out_classes)
        )

    def forward(self, x):
        return self.resnet_block(self.conv_block(x))


    def make_layers(self, residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels * 4,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels * 4),
            )

        layers.append(ResidualBlock(in_channels=self.in_channels, out_channels=out_channels,
                                    identity_downsample=identity_downsample, stride=stride))

        self.in_channels = out_channels * 4

        for _ in range(residual_blocks - 1):
            layers.append(ResidualBlock(in_channels=self.in_channels, out_channels=out_channels))

        return nn.Sequential(*layers)



def ResNet50(image_channels=3, out_classes=1000):
    return ResNet(layers=[3, 4, 6, 3], image_channels=image_channels, out_classes=out_classes)

def ResNet101(image_channels=3, out_classes=1000):
    return ResNet(layers=[3, 4, 23, 3], image_channels=image_channels, out_classes=out_classes)


def ResNet152(image_channels=3, out_classes=1000):
    return ResNet(layers=[3, 8, 36, 3], image_channels=image_channels, out_classes=out_classes)



if __name__ == '__main__':
    test = torch.randn(2, 3, 224, 224)
    resnet = ResNet101(image_channels=3, out_classes=1000)
    print(resnet(test).shape)
