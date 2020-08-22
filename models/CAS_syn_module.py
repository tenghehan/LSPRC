import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CASSynModule(nn.Module):
    def __init__(self, C):
        super(CASSynModule, self).__init__()

        self.conv_syn_a = nn.Conv2d(C * 2, C, kernel_size=1)
        self.conv_syn_b = nn.Conv2d(C * 2, C, kernel_size=1)

        self.conv_M_a = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.conv_M_b = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        self.dropout_M = False
        self.dropout_M_rate = 0.6

    def forward(self, V_a_a, feature_sh_a, V_a_b, feature_sh_b):
        feature_cat = torch.cat((feature_sh_a, feature_sh_b), 1)

        feature_syn_a = self.conv_syn_a(feature_cat)
        feature_syn_b = self.conv_syn_b(feature_cat)

        avg = torch.mean(feature_cat, dim=1, keepdim=True)
        max, _ = torch.max(feature_cat, dim=1, keepdim=True)
        M_cat = torch.cat((avg, max), 1)
        M_a = F.sigmoid(self.conv_M_a(M_cat))
        M_b = F.sigmoid(self.conv_M_b(M_cat))

        if self.dropout_M:
            M_a = F.dropout(M_a, p=self.dropout_M_rate, training=self.training)
            M_b = F.dropout(M_b, p=self.dropout_M_rate, training=self.training)

        A_a = (V_a_a.view(V_a_a.size(0), -1, 1, 1)) * M_a
        A_b = (V_a_b.view(V_a_b.size(0), -1, 1, 1)) * M_b

        return A_a, feature_syn_a, feature_syn_b, A_b