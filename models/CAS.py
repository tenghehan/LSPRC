import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CAS_pre_module import CASPreModule
from models.CAS_syn_module import CASSynModule


class CAS(nn.Module):
    def __init__(self, C):
        super(CAS, self).__init__()

        self.pre_a = CASPreModule(C)
        self.pre_b = CASPreModule(C)

        self.syn = CASSynModule(C)


    def forward(self, x_a, x_b):
        feature_a = x_a
        feature_b = x_b

        feature_t_a, V_a_a, feature_sh_a = self.pre_a(feature_a)
        feature_t_b, V_a_b, feature_sh_b = self.pre_b(feature_b)

        A_a, feature_syn_a, feature_syn_b, A_b = \
            self.syn(V_a_a, feature_sh_a, V_a_b, feature_sh_b)

        feature_out_a = (feature_a + feature_t_a + feature_syn_a) * A_a
        feature_out_b = (feature_b + feature_t_b + feature_syn_b) * A_b

        return feature_out_a, feature_out_b


if __name__ == "__main__":
    module = CAS(512)
    module.eval()
    x_a = torch.zeros([64, 512, 64, 48]).float()
    x_b = torch.zeros([64, 512, 64, 48]).float()

    x_a, x_b = module(x_a, x_b)