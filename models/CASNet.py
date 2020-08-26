import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.resnet import resnet34
from models.CAS import CAS
from models.self_mask_block import self_mask_block

class CAS_ResNet34(nn.Module):
    def __init__(
            self,
            ratio,
            is_global: list
    ):
        super(CAS_ResNet34, self).__init__()

        """
        task_a: global attributes
        task_b: part attributes
        """
        self.is_global = is_global
        self.attr_num_a = is_global.count(True)
        self.attr_num_b = is_global.count(False)
        self.pretrained = True
        self.drop_pool = True
        self.drop_pool_rate = 0.5

        self.features_a = resnet34(pretrained=self.pretrained)
        self.features_b = resnet34(pretrained=self.pretrained)

        self.classifier_a = nn.Linear(512, self.attr_num_a)
        self.classifier_b = nn.Linear(512, self.attr_num_b)
        init.normal_(self.classifier_a.weight, std=0.001)
        init.constant_(self.classifier_a.bias, 0)
        init.normal_(self.classifier_b.weight, std=0.001)
        init.constant_(self.classifier_b.bias, 0)

        self.CAS_module1 = CAS(64, ratio)
        self.CAS_module2 = CAS(128, ratio)
        self.CAS_module3 = CAS(256, ratio)
        self.CAS_module4 = CAS(512, ratio)

        self.spatial_attention = True

        self.self_mask_block_layer1_a = self_mask_block(64)
        self.self_mask_block_layer1_b = self_mask_block(64)
        self.self_mask_block_layer2_a = self_mask_block(128)
        self.self_mask_block_layer2_b = self_mask_block(128)
        self.self_mask_block_layer3_a = self_mask_block(256)
        self.self_mask_block_layer3_b = self_mask_block(256)
        self.self_mask_block_layer4_a = self_mask_block(512)
        self.self_mask_block_layer4_b = self_mask_block(512)


    def ResNet_Layer1(self, x_a, x_b):
        x_a = self.features_a.conv1(x_a)
        x_a = self.features_a.bn1(x_a)
        x_a = self.features_a.relu(x_a)
        x_a = self.features_a.maxpool(x_a)
        x_a = self.features_a.layer1(x_a)

        x_b = self.features_b.conv1(x_b)
        x_b = self.features_b.bn1(x_b)
        x_b = self.features_b.relu(x_b)
        x_b = self.features_b.maxpool(x_b)
        x_b = self.features_b.layer1(x_b)

        return x_a, x_b


    def ResNet_Layer2(self, x_a, x_b):
        x_a = self.features_a.layer2(x_a)
        x_b = self.features_b.layer2(x_b)

        return x_a, x_b


    def ResNet_Layer3(self, x_a, x_b):
        x_a = self.features_a.layer3(x_a)
        x_b = self.features_b.layer3(x_b)

        return x_a, x_b


    def ResNet_Layer4(self, x_a, x_b):
        x_a = self.features_a.layer4(x_a)
        x_b = self.features_b.layer4(x_b)

        return x_a, x_b

    def combine_a_b(self, x_a, x_b, x):
        i_a, i_b = 0, 0
        for step, flag in enumerate(self.is_global):
            if flag:
                x[:, step] = x_a[:, i_a]
                i_a += 1
            else:
                x[:, step] = x_b[:, i_b]
                i_b += 1
        return x

    def classification(self, x_a, x_b):
        x_a = F.avg_pool2d(x_a, x_a.shape[2:])
        x_a = x_a.view(x_a.size(0), -1)

        x_b = F.avg_pool2d(x_b, x_b.shape[2:])
        x_b = x_b.view(x_b.size(0), -1)

        if self.drop_pool:
            x_a = F.dropout(x_a, p=self.drop_pool_rate, training=self.training)
            x_b = F.dropout(x_b, p=self.drop_pool_rate, training=self.training)
        x_a = self.classifier_a(x_a)
        x_b = self.classifier_b(x_b)

        x = torch.zeros(x_a.shape[0], self.attr_num_a + self.attr_num_b)
        x = self.combine_a_b(x_a, x_b, x)
        return x


    def forward(self, x):
        x_a, x_b = self.ResNet_Layer1(x_a=x, x_b=x)
        if self.spatial_attention:
            x_a = self.self_mask_block_layer1_a(x_a)
            x_b = self.self_mask_block_layer1_b(x_b)
        x_a, x_b = self.CAS_module1(x_a, x_b)

        x_a, x_b = self.ResNet_Layer2(x_a=x_a, x_b=x_b)
        if self.spatial_attention:
            x_a = self.self_mask_block_layer2_a(x_a)
            x_b = self.self_mask_block_layer2_b(x_b)
        x_a, x_b = self.CAS_module2(x_a, x_b)

        x_a, x_b = self.ResNet_Layer3(x_a=x_a, x_b=x_b)
        if self.spatial_attention:
            x_a = self.self_mask_block_layer3_a(x_a)
            x_b = self.self_mask_block_layer3_b(x_b)
        x_a, x_b = self.CAS_module3(x_a, x_b)

        x_a, x_b = self.ResNet_Layer4(x_a=x_a, x_b=x_b)
        if self.spatial_attention:
            x_a = self.self_mask_block_layer4_a(x_a)
            x_b = self.self_mask_block_layer4_b(x_b)
        x_a, x_b = self.CAS_module4(x_a, x_b)

        x = self.classification(x_a=x_a, x_b=x_b)

        return x


if __name__ == "__main__":
    model = CAS_ResNet34(32, [True] * 11 + [False] * 35 + [True] * 9)
    model.eval()
    batch_tensor = torch.ones([64, 3, 256, 192]).float()
    output = model(batch_tensor)