import torch

# VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}

class VGGNet(torch.nn.Module):
  def __init__(self, in_channels=3, output_classes=1000) -> None:
    super(VGGNet, self).__init__()
    self.in_channels = in_channels
    self.conv_layers = self.create_conv_layers(VGG_types['VGG16'])

    self.fc_layers = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=512*7*7, out_features=4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=4096, out_features=4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=4096, out_features=output_classes)
    )

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    return self.fc_layers(self.conv_layers(X))

  def create_conv_layers(self, architecture):
    layers = []
    in_channels = self.in_channels
    for x in architecture:
      if type(x) == int:
        out_channels = x
        layers += [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=3, stride=1, padding=1),
                   torch.nn.BatchNorm2d(x),
                   torch.nn.ReLU()]
        in_channels = x
      elif x == 'M':
        layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]

    return torch.nn.Sequential(*layers)


if __name__ == "__main__":
    vgg16 = VGGNet()
    test = torch.randn(1, 3, 224, 224)
    print(vgg16(test).shape) ## output shape [1, 1000]