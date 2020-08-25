import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.resnet import resnet50


class CoCNN_ResNet50_Combine(nn.Module):
    def __init__(
            self,
            attr_num,
            attr_part_list,
    ):
        super(CoCNN_ResNet50_Combine, self).__init__()

        self.attr_num = attr_num
        self.attr_part_list = attr_part_list
        self.attr_num_whole = attr_part_list.count(0)
        self.attr_num_head = attr_part_list.count(1)
        self.attr_num_upper = attr_part_list.count(2)
        self.attr_num_lower = attr_part_list.count(3)

        self.pretrained = True

        self.features = resnet50(pretrained=self.pretrained)

        self.dropout = True
        self.dropout_rate = 0.5

        self.fc_whole = nn.Linear(2048, self.attr_num_whole)
        self.init_fc(self.fc_whole)
        self.fc_head = nn.Linear(2048, self.attr_num_head)
        self.init_fc(self.fc_head)
        self.fc_upper = nn.Linear(2048, self.attr_num_upper)
        self.init_fc(self.fc_upper)
        self.fc_lower = nn.Linear(2048, self.attr_num_lower)
        self.init_fc(self.fc_lower)

    def init_fc(self, fc: nn.Linear, std=0.001):
        init.normal_(fc.weight, std=std)
        init.constant_(fc.bias, 0)

    def whole_body_classifier(self, x):
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_whole(x)
        return x

    def head_classifier(self, x):
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_head(x)
        return x

    def upper_body_classifier(self, x):
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_upper(x)
        return x

    def lower_body_classifier(self, x):
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_lower(x)
        return x

    def combine(self, whole, head, upper, lower):
        i_w, i_h, i_u, i_l = 0, 0, 0, 0
        x = torch.zeros(whole.shape[0], self.attr_num)
        for step, flag in enumerate(self.attr_part_list):
            if flag == 0:
                x[:, step] = whole[:, i_w]
                i_w += 1
            elif flag == 1:
                x[:, step] = head[:, i_h]
                i_h += 1
            elif flag == 2:
                x[:, step] = upper[:, i_u]
                i_u += 1
            else:
                x[:, step] = lower[:, i_l]
                i_l += 1
        return x

    def forward(self, x):
        x = self.features(x)
        logits_whole = self.whole_body_classifier(x)
        logits_head = self.head_classifier(x[:, :, 0:4, :])
        logits_upper = self.upper_body_classifier(x[:, :, 3:9, :])
        logits_lower = self.lower_body_classifier(x[:, :, 8:, :])
        x = self.combine(logits_whole, logits_head, logits_upper, logits_lower)
        return x


class CoCNN_ResNet50_Max(nn.Module):
    def __init__(
            self,
            attr_num,
            attr_part_list,
    ):
        super(CoCNN_ResNet50_Max, self).__init__()

        self.attr_num = attr_num
        self.attr_part_list = attr_part_list

        self.pretrained = True

        self.features = resnet50(pretrained=self.pretrained)

        self.dropout = True
        self.dropout_rate = 0.5

        self.attention = True

        self.fc_whole = nn.Linear(2048, self.attr_num)
        self.init_fc(self.fc_whole)
        self.fc_head = nn.Linear(2048, self.attr_num)
        self.init_fc(self.fc_head)
        self.fc_upper = nn.Linear(2048, self.attr_num)
        self.init_fc(self.fc_upper)
        self.fc_lower = nn.Linear(2048, self.attr_num)
        self.init_fc(self.fc_lower)

        self.attention_whole = nn.Linear(2048, 2048)
        self.init_fc(self.attention_whole)
        self.attention_head = nn.Linear(2048, 2048)
        self.init_fc(self.attention_head)
        self.attention_upper = nn.Linear(2048, 2048)
        self.init_fc(self.attention_upper)
        self.attention_lower = nn.Linear(2048, 2048)
        self.init_fc(self.attention_lower)

    def init_fc(self, fc: nn.Linear, std=0.001):
        init.normal_(fc.weight, std=std)
        init.constant_(fc.bias, 0)

    def whole_body_classifier(self, x):
        if self.attention:
            atten = F.avg_pool2d(x, x.shape[2:])
            atten = atten.view(atten.size(0), -1)
            atten = self.attention_whole(atten)

            x = (atten.view(atten.size(0), -1, 1, 1)) * x

        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_whole(x)
        return x

    def head_classifier(self, x):
        if self.attention:
            atten = F.avg_pool2d(x, x.shape[2:])
            atten = atten.view(atten.size(0), -1)
            atten = self.attention_head(atten)

            x = (atten.view(atten.size(0), -1, 1, 1)) * x

        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_head(x)
        return x

    def upper_body_classifier(self, x):
        if self.attention:
            atten = F.avg_pool2d(x, x.shape[2:])
            atten = atten.view(atten.size(0), -1)
            atten = self.attention_upper(atten)

            x = (atten.view(atten.size(0), -1, 1, 1)) * x

        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_upper(x)
        return x

    def lower_body_classifier(self, x):
        if self.attention:
            atten = F.avg_pool2d(x, x.shape[2:])
            atten = atten.view(atten.size(0), -1)
            atten = self.attention_lower(atten)

            x = (atten.view(atten.size(0), -1, 1, 1)) * x

        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_lower(x)
        return x

    def combine(self, whole, head, upper, lower):
        x, _ = torch.max(torch.stack((whole, head, upper, lower), dim=0), dim=0)
        return x

    def forward(self, x):
        x = self.features(x)
        logits_whole = self.whole_body_classifier(x)
        logits_head = self.head_classifier(x[:, :, 0:4, :])
        logits_upper = self.upper_body_classifier(x[:, :, 3:9, :])
        logits_lower = self.lower_body_classifier(x[:, :, 8:, :])
        x = self.combine(logits_whole, logits_head, logits_upper, logits_lower)
        return x

if __name__ == "__main__":
    model = CoCNN_ResNet50_Max(55, [0] * 11 + [1] * 5 + [2] * 10 + [3] * 12 + [0] * 17)
    model.eval()
    batch_tensor = torch.zeros([32, 3, 448, 224]).float()
    output = model(batch_tensor)