import torch
from torch import nn


class GoogLeNet(nn.Module):
    def __init__(self, auxiliary: bool, out_classes: int=1000):
        super(GoogLeNet, self).__init__()
        assert auxiliary == True or auxiliary == False
        self.auxiliary = auxiliary

        ### Convolution blocks
        self.conv_1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_2  = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ### Inception blocks
        ## parameters: in_channels, out_1x1, reduced_3x3, out_3x3, reduced_5x5, out_5x5, out_1x1_maxpool
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=1024, out_features=out_classes)

        if self.auxiliary:
            self.aux_1 = Auxiliary(in_channels=512, out_classes=out_classes)
            self.aux_2 = Auxiliary(in_channels=528, out_classes=out_classes)
        else:
            self.aux_1 = None
            self.aux_2 = None

    def forward(self, x):
        x = self.maxpool_2(self.conv_2(self.maxpool_1(self.conv_1(x))))
        x = self.maxpool_3(self.inception3b(self.inception3a(x)))
        x = self.inception4a(x)

        ## Auxiliary output 1 after 4a
        if self.auxiliary and self.training:
            aux1 = self.aux_1(x)

        x = self.inception4d(self.inception4c(self.inception4b(x)))

        ## Auxiliary output 2 after 4d
        if self.auxiliary and self.training:
            aux2 = self.aux_2(x)

        x = self.inception5b(self.inception5a(self.maxpool_4(self.inception4e(x))))

        out = self.linear(self.dropout(self.flat(self.avgpool(x))))

        if self.auxiliary and self.training:
            return aux1, aux2, out

        return out    




class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, reduced_3x3, out_3x3, reduced_5x5, out_5x5, out_1x1_maxpool):
        super(InceptionBlock, self).__init__()

        self.branch_1 = ConvBlock(in_channels=in_channels, out_channels=out_1x1, kernel_size=1)

        self.branch_2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=reduced_3x3, kernel_size=1),
            ConvBlock(in_channels=reduced_3x3, out_channels=out_3x3, kernel_size=3, padding=1)
        )

        self.branch_3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=reduced_5x5, kernel_size=1),
            ConvBlock(in_channels=reduced_5x5, out_channels=out_5x5, kernel_size=5, padding=2)
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=out_1x1_maxpool, kernel_size=1)
        )

    def forward(self, X):
        return torch.cat([self.branch_1(X), self.branch_2(X), self.branch_3(X), self.branch_4(X)], dim=1)




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.relu(self.batchnorm(self.conv(X)))


class Auxiliary(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(Auxiliary, self).__init__()

        self.aux = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            ConvBlock(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=1024, out_features=out_classes)
        )
    
    def forward(self, X):
        return self.aux(X)


if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224)
    model = GoogLeNet(auxiliary=True, out_classes=1000)
    print(model(x)[2].shape) ## output --> [3, 1000]