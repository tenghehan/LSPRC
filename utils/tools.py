import numpy as np
import random
import torch
from torchvision import transforms


def set_seed(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.enabled = True
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


def show_PILImage(tensor):
    image = transforms.ToPILImage()(tensor)
    image.show()
