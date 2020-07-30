import torch.utils.data as data
import numpy as np
import torchvision.transforms as T
import os
from PIL import Image
from typing import List, Tuple, Optional


class AttrDataset(data.Dataset):

    def __init__(
            self,
            args,
            annotation_data: List[Tuple[str, Optional[np.ndarray]]],
            attr_names_cn,
            attr_names_en,
            transform=None,
    ):
        self.annotation = annotation_data
        self.transform = transform
        self.image_dir_path = args.train_val_images_dir
        self.attr_names_cn = attr_names_cn
        self.attr_names_en = attr_names_en

    def __getitem__(self, index):
        image_filename, gt_label = self.annotation[index]
        image_path = os.path.join(self.image_dir_path, image_filename)
        image = Image.open(image_path)

        gt_label = gt_label.astype(np.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, gt_label, image_filename

    def __len__(self):
        return len(self.annotation)


def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform