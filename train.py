import argparse
import torch
from data.load_rap_attributes_data_mat import *
from data.AttrDataset import AttrDataset, get_transform
from torch.utils.data import DataLoader
from config import argument_parser
from utils.tools import set_seed

set_seed(100)

def main():

    parser = argument_parser()

    args = parser.parse_args()

    use_gpu = True
    if use_gpu:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(args.gpu_id)
    else:
        device = torch.device("cpu")

    annotation_data = loadRAPAttr(args.annotation_file)
    annotation_data = processRAPAttr(annotation_data)
    annotation_data = cleanRAPAttr(annotation_data)

    attr_names_cn = annotation_data['attr_names_cn']
    attr_names_en = annotation_data['attr_names_en']

    train_transform, valid_transform = get_transform(args)

    train_dataset = AttrDataset(
        args=args,
        annotation_data=annotation_data['training_set'],
        transform=train_transform,
        attr_names_cn=attr_names_cn,
        attr_names_en=attr_names_en
    )
    valid_dataset = AttrDataset(
        args=args,
        annotation_data=annotation_data['validation_set'],
        transform=train_transform,
        attr_names_cn=attr_names_cn,
        attr_names_en=attr_names_en
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    for index, (images, gt_labels, image_filenames) in enumerate(valid_loader):
        print(images.size())
        break


if __name__ == '__main__':
    main()