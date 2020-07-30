import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.resnet import resnet50


class DeepMAR_ResNet50(nn.Module):
    def __init__(
            self,
            attr_num,
    ):
        super(DeepMAR_ResNet50, self).__init__()

        self.attr_num = attr_num
        self.last_conv_stride = 2
        self.drop_pool5 = True
        self.drop_pool5_rate = 0.5
        self.pretrained = True

        self.features = resnet50(pretrained=self.pretrained, last_conv_stride=self.last_conv_stride)
        self.classifier = nn.Linear(2048, self.attr_num)
        init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.drop_pool5:
            x = F.dropout(x, p=self.drop_pool5_rate, training=self.training)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = DeepMAR_ResNet50(55)
    model.eval()
    batch_tensor = torch.zeros([64, 3, 224, 224], dtype=torch.int32)
    output = model(batch_tensor.float())
    print(output.size())