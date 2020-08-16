import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CASPreModule(nn.Module):
    def __init__(self, C, ratio=0.6):
        super(CASPreModule, self).__init__()

        self.ratio = ratio
        reduced_C = int(C * ratio)

        self.W_m = nn.Linear(C, reduced_C)
        self.init_fc(self.W_m)

        self.W_t = nn.Linear(reduced_C, C)
        self.init_fc(self.W_t)

        self.W_a = nn.Linear(reduced_C, C)
        self.init_fc(self.W_a)

        self.W_sh = nn.Linear(reduced_C, C)
        self.init_fc(self.W_sh)

    def init_fc(self, fc: nn.Linear, std=0.001):
        init.normal_(fc.weight, std=std)
        init.constant_(fc.bias, 0)

    def pre_task_specific_branch(self, V_m):
        V_t = F.sigmoid(self.W_t(V_m))
        feature_t = (V_t.view(V_t.size(0), -1, 1, 1)) * self.feature

        return feature_t

    def pre_attentive_branch(self, V_m):
        V_a = F.sigmoid(self.W_a(V_m))

        return V_a

    def pre_synergetic_branch(self, V_m):
        V_sh = F.sigmoid(self.W_sh(V_m))
        feature_sh = (V_sh.view(V_sh.size(0), -1, 1, 1)) * self.feature

        return feature_sh

    def forward(self, x):
        self.feature = x
        V_g = F.avg_pool2d(x, x.shape[2:]).view(x.size(0), -1)
        V_m = F.relu(self.W_m(V_g), inplace=True)

        feature_t = self.pre_task_specific_branch(V_m)
        V_a = self.pre_attentive_branch(V_m)
        feature_sh = self.pre_synergetic_branch(V_m)

        return feature_t, V_a, feature_sh


if __name__ == "__main__":
    module = CASPreModule(128)
    module.eval()
    x = torch.zeros([64, 128, 64, 48]).float()

    x = module(x)
