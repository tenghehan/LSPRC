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
    parser.add_argument('--weighted_loss', default=False, action='store_true')
    parser.add_argument('--joint_loss', default=False, action='store_true')
    parser.add_argument('--data_augment', default=False, action='store_true')

    return parser

def argument_cas_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_file', type=str, default='data/RAP/RAP_attributes_data.mat', \
                        help='file path of attributes annotation results')
    parser.add_argument('--train_val_images_dir', type=str, default='data/RAP/training_validation_images', \
                        help='directory path of training and validation images')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=192)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=70)
    parser.add_argument('--lr_ft', type=float, default=0.0001, help='learning rate of feature extractor')
    parser.add_argument('--lr_new', type=float, default=0.0005, help='learning rate of classifier_base')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--weighted_loss', default=False, action='store_true')
    parser.add_argument('--joint_loss', default=False, action='store_true')
    parser.add_argument('--data_augment', default=False, action='store_true')
    parser.add_argument('--ratio', type=int, default=32)
    parser.add_argument('--nesterov', default=False, action='store_true')

    return parser


def argument_cocnn_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_file', type=str, default='data/RAP/RAP_attributes_data.mat', \
                        help='file path of attributes annotation results')
    parser.add_argument('--train_val_images_dir', type=str, default='data/RAP/training_validation_images', \
                        help='directory path of training and validation images')
    parser.add_argument('--resize_height', type=int, default=512)
    parser.add_argument('--resize_width', type=int, default=256)
    parser.add_argument('--input_height', type=int, default=448)
    parser.add_argument('--input_width', type=int, default=224)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_ft', type=float, default=0.0001, help='learning rate of feature extractor')
    parser.add_argument('--lr_new', type=float, default=0.0005, help='learning rate of classifier_base')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--weighted_loss', default=False, action='store_true')
    parser.add_argument('--joint_loss', default=False, action='store_true')
    parser.add_argument('--loss', type=str, required=True, choices=['cross_entropy', 'L2'])
    parser.add_argument('--data_augment', default=False, action='store_true')
    parser.add_argument('--nesterov', default=False, action='store_true')

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
    parser.add_argument('--model_path', type=str, default='results/deepMAR_jointweightedloss/max.pth')
    parser.add_argument('--probs', type=str, required=True, choices=['sigmoid', 'joint'])

    return parser

