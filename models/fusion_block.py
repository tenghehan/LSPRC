import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class fusion_block(nn.Module):
    def __init__(self, attr_num):
        super(fusion_block, self).__init__()

        self.attr_num = attr_num
        self.conv1_1 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=1)
        self.conv1_3 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv1_4 = nn.Conv2d(512, 128, kernel_size=1)

        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)
        self.conv3_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1)
        self.relu = nn.BatchNorm2d(256)

        self.classifier = nn.Linear(256, attr_num)
        init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.classifier.bias, 0)

    def forward(self, x_1, x_2, x_3, x_4):
        x_1 = self.conv1_1(x_1)
        x_1 = self.conv3_1(x_1)
        x_1 = self.bn1(x_1)
        x_1 = self.relu1(x_1)

        x_2 = self.conv1_2(x_2)
        x_2 = F.interpolate(x_2, size=[64, 48], mode='bilinear')
        x_2 = self.conv3_2(x_2)
        x_2 = self.bn2(x_2)
        x_2 = self.relu2(x_2)

        x_3 = self.conv1_3(x_3)
        x_3 = F.interpolate(x_3, size=[64, 48], mode='bilinear')
        x_3 = self.conv3_3(x_3)
        x_3 = self.bn3(x_3)
        x_3 = self.relu3(x_3)

        x_4 = self.conv1_4(x_4)
        x_4 = F.interpolate(x_4, size=[64, 48], mode='bilinear')
        x_4 = self.conv3_4(x_4)
        x_4 = self.bn4(x_4)
        x_4 = self.relu4(x_4)

        x = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        x = self.conv1(x)
        x = self.relu(x)

        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    module = fusion_block(20)
    module.eval()
    x_1 = torch.zeros([64, 64, 64, 48]).float()
    x_2 = torch.zeros([64, 128, 32, 24]).float()
    x_3 = torch.zeros([64, 256, 16, 12]).float()
    x_4 = torch.zeros([64, 512, 8, 6]).float()

    x = module(x_1, x_2, x_3, x_4)
