import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class self_mask_block(nn.Module):
    def __init__(self, inplanes, planes=1):
        super(self_mask_block, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def init_fc(self, fc: nn.Linear, std=0.01):
        init.normal_(fc.weight, std=std)
        init.constant_(fc.bias, 0)

    def forward(self, x):
        origin_x = x

        atten_map = self.conv1(x)
        atten_map = self.bn1(atten_map)
        atten_map = self.relu(atten_map)

        atten_map = self.conv2(atten_map)
        atten_map = self.bn2(atten_map)
        atten_map = self.relu(atten_map)

        atten_map = self.conv3(atten_map)
        atten_map = self.bn3(atten_map)

        x = (atten_map + 1) * origin_x
        return x

if __name__ == "__main__":
    model = self_mask_block(64)
    model.eval()
    batch_tensor = torch.zeros([32, 64, 224, 224]).float()
    output = model(batch_tensor)