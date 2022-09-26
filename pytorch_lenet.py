import torch


class LeNet(torch.nn.Module):
  def __init__(self, out_classes=10) -> None:
    super(LeNet, self).__init__()
    self.conv_block = torch.nn.Sequential( ## input --> [N, 1, 32, 32]
        torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0), ## output --> [N, 6, 28, 28]
        torch.nn.ReLU(), # original --> Tanh
        torch.nn.AvgPool2d(kernel_size=2, stride=2), ## output --> [N, 6, 14, 14]

        torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0), ## output --> [N, 6, 10, 10]
        torch.nn.ReLU(), # original --> Tanh
        torch.nn.AvgPool2d(kernel_size=2, stride=2), ## output --> [N, 6, 5, 5]

        torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0), ## output --> [N, 120, 1, 1]
        torch.nn.ReLU(), # original --> Tanh
    )

    self.fc_block = torch.nn.Sequential(
        torch.nn.Flatten(), ## output --> [N, 120]
        torch.nn.Linear(in_features=120, out_features=84), ## output --> [N, 84]
        torch.nn.ReLU(), # original --> Tanh
        torch.nn.Linear(in_features=84, out_features=out_classes) ## output --> [N, classes]
    )

  def forward(self, X):
    return self.fc_block(self.conv_block(X))



if __name__ == "__main__":
    lenet = LeNet()
    test = torch.randn(3, 1, 32, 32)
    print(lenet(test).shape) ## output shape [3, 10]