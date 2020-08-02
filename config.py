import argparse


def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_file', type=str, default='data/RAP/RAP_attributes_data.mat', \
                        help='file path of attributes annotation results')
    parser.add_argument('--train_val_images_dir', type=str, default='data/RAP/training_validation_images', \
                        help='directory path of training and validation images')
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--lr_ft', type=float, default=0.01, help='learning rate of feature extractor')
    parser.add_argument('--lr_new', type=float, default=0.1, help='learning rate of classifier_base')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--model_name', type=str, required=True)

    return parser

def inference_argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_file', type=str, default='data/RAP/RAP_attributes_data.mat', \
                        help='file path of attributes annotation results')
    parser.add_argument('--train_val_images_dir', type=str, default='data/RAP/training_validation_images', \
                        help='directory path of training and validation images')
    parser.add_argument('--test_images_dir', type=str, default='data/RAP/test_images')
    parser.add_argument('--dataset', type=str, required=True, choices=['validation_set','test_set'])
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
    parser.add_argument('--model_path', type=str, default='results/deepMAR/max.pth')

    return parser

