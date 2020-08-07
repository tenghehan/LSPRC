import numpy as np
import random
import torch
from torch import tensor
from torchvision import transforms
from torchvision.utils import make_grid
from easydict import EasyDict
import matplotlib.pyplot as plt
import os
import json
import math


def set_seed(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.enabled = True
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


def show_PILImage(tensor):
    image = transforms.ToPILImage()(tensor)
    image.show()


def to_scalar(vt):
    """
    preprocess a 1-length pytorch Variable or Tensor to scalar
    """
    # if isinstance(vt, Variable):
    #     return vt.data.cpu().numpy().flatten()[0]
    if torch.is_tensor(vt):
        if vt.dim() == 0:
            return vt.detach().cpu().numpy().flatten().item()
        else:
            return vt.detach().cpu().numpy()
    elif isinstance(vt, np.ndarray):
        return vt
    else:
        raise TypeError('Input should be a ndarray or tensor')


def get_pedestrian_metrics(gt_label, preds_probs, threshold=0.5):
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result


def show_image_model_to_tensorboard(writer, model, train_loader):
    dataiter = iter(train_loader)
    images, labels, image_filenames = dataiter.next()

    img_grid = make_grid(images)

    # plt.imshow(img_grid.numpy())

    writer.add_image('pedestrian_attributes_images', img_grid)
    writer.add_graph(model, images)
    writer.close()


def get_attr_weights(training_set):
    data = np.concatenate([attr.reshape([1, -1]) for filename, attr in training_set], axis=0)
    pos_rate = list(data.sum(axis=0) * 1.0 / data.shape[0])
    weights_attr = [(math.exp(1 - rate), math.exp(rate)) for rate in pos_rate]
    return weights_attr


def get_weights(weights_attr, gt_labels):
    pos_weights_attr = tensor([[pos for pos, neg in weights_attr]])
    neg_weights_attr = tensor([[neg for pos, neg in weights_attr]])
    weights = pos_weights_attr * gt_labels + neg_weights_attr * (1 - gt_labels)
    return weights


def show_scalars_to_tensorboard(writer, epoch, train_loss, valid_loss, train_result, valid_result):
    i = epoch
    writer.add_scalar('Loss/train', train_loss, i)
    writer.add_scalar('Loss/valid', valid_loss, i)
    writer.add_scalar('ma/train', train_result.ma, i)
    writer.add_scalar('ma/valid', valid_result.ma, i)
    writer.add_scalar('Accuracy/train', train_result.instance_acc, i)
    writer.add_scalar('Accuracy/valid', valid_result.instance_acc, i)
    writer.add_scalar('Precision/train', train_result.instance_prec, i)
    writer.add_scalar('Precision/valid', valid_result.instance_prec, i)
    writer.add_scalar('Recall/train', train_result.instance_recall, i)
    writer.add_scalar('Recall/valid', valid_result.instance_recall, i)
    writer.add_scalar('F1/train', train_result.instance_f1, i)
    writer.add_scalar('F1/valid', valid_result.instance_f1, i)


def output_results_to_screen(epoch, train_loss, valid_loss, train_result, valid_result):
    print(f'epoch {epoch} \n',
          'training loss: {:.6f}, validate loss: {:.6f} \n'.format(train_loss, valid_loss),
          'training ma: {:.4f}, Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f} \n'.format(
              train_result.ma, train_result.instance_acc, train_result.instance_prec,
              train_result.instance_recall, train_result.instance_f1
          ),
          'validation ma: {:.4f}, Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f} \n'.format(
              valid_result.ma, valid_result.instance_acc, valid_result.instance_prec,
              valid_result.instance_recall, valid_result.instance_f1
          ))


def save_results_to_json(best_epoch, loss_list, result_list, save_result_path):
    result = []
    for i in range(len(loss_list)):
        result.append({
            'epoch': i,
            'training_loss': loss_list[i][0],
            'validation_loss': loss_list[i][1],
            'training_ma': result_list[i][0].ma,
            'validation_ma': result_list[i][1].ma,
            'training_acc': result_list[i][0].instance_acc,
            'validation_acc': result_list[i][1].instance_acc,
            'training_prec': result_list[i][0].instance_prec,
            'validation_prec': result_list[i][1].instance_prec,
            'training_recall': result_list[i][0].instance_recall,
            'validation_recall': result_list[i][1].instance_recall,
            'training_f1': result_list[i][0].instance_f1,
            'validation_f1': result_list[i][1].instance_f1,
        })
    result_json = open(os.path.join(save_result_path, 'result_data.json'), 'w+')
    json.dump({
        'best_epoch': best_epoch,
        'result_data': result,
    }, result_json)


class AverageMeter(object):
    """
    Computes and stores the average and current value

    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-20)


if __name__ == '__main__':
    weights_attr = [(0.9, 0.3), (0.4, 0.5), (0.2, 0.3)]
    gt_labels = np.array([[1,1,1],[1,0,0],[0,1,1]])
    weights = get_weights(weights_attr, gt_labels)