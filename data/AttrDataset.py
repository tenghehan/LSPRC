import torch
import torch.utils.data as data
import numpy as np
import torchvision.transforms as T
import os
from PIL import Image
from typing import List, Tuple, Optional


class AttrDataset(data.Dataset):

    def __init__(
            self,
            image_dir_path,
            annotation_data: List[Tuple[str, Optional[np.ndarray]]],
            attr_names_cn,
            attr_names_en,
            transform=None,
    ):
        self.annotation = annotation_data
        self.transform = transform
        self.image_dir_path = image_dir_path
        self.attr_names_cn = attr_names_cn
        self.attr_names_en = attr_names_en

    def __getitem__(self, index):
        image_filename, gt_label = self.annotation[index]
        image_path = os.path.join(self.image_dir_path, image_filename)
        image = Image.open(image_path)

        if gt_label is not None:
            gt_label = gt_label.astype(np.float32)
        else:
            gt_label = 0

        if self.transform is not None:
            image = self.transform(image)

        return image, gt_label, image_filename

    def __len__(self):
        return len(self.annotation)


def get_transform(args):
    height = args.height
    width = args.width
    noise_std = 0.1
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        # T.RandomRotation(degrees=(-30, 30)),
        T.ToTensor(),
        normalize,
        # T.RandomApply([
        #     T.Lambda(lambda data: data + noise_std * torch.randn_like(data)),
        # ], p=0.5),
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform


def get_transform_cocnn(args):
    noise_std = 0.1
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((args.resize_height, args.resize_width)),
        T.RandomCrop((args.input_height, args.input_width)),
        T.RandomHorizontalFlip(),
        # T.RandomRotation(degrees=(-30, 30)),
        T.ToTensor(),
        normalize,
        T.RandomApply([
            T.Lambda(lambda data: data + noise_std * torch.randn_like(data)),
        ], p=0.5),
    ])

    valid_transform = T.Compose([
        T.Resize((args.input_height, args.input_width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform