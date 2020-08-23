import numpy as np
import random
import torch
from torch import tensor
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import WeightedRandomSampler
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


def get_attr_weights(training_set, attr_augment_rate):
    attr_augment_rate = torch.ones([55]).float()
    data = np.concatenate([attr.reshape([1, -1]) for filename, attr in training_set], axis=0)
    pos_num = list(data.sum(axis=0) * np.asarray(attr_augment_rate))
    pos_rate = list(data.sum(axis=0) * np.asarray(attr_augment_rate) * 1.0 / data.shape[0])
    weights_attr = [(math.exp(1 - rate), math.exp(rate)) for rate in pos_rate]
    return pos_num, weights_attr


def get_weights(weights_attr, gt_labels):
    pos_weights_attr = tensor([[pos for pos, neg in weights_attr]]).to(gt_labels.device)
    neg_weights_attr = tensor([[neg for pos, neg in weights_attr]]).to(gt_labels.device)
    weights = pos_weights_attr * gt_labels + neg_weights_attr * (1 - gt_labels)
    return weights


def get_attr_augment_rate(attr_augment_rate, training_set):
    attr_augment_rate[2] = 3
    attr_augment_rate[5] = 3
    attr_augment_rate[11] = 8
    attr_augment_rate[14] = 2
    attr_augment_rate[25] = 10
    attr_augment_rate[35] = 2
    attr_augment_rate[37] = 10
    attr_augment_rate[43] = 2
    attr_augment_rate[50] = 3
    attr_augment_rate[51] = 2
    attr_augment_rate[54] = 5
    return attr_augment_rate


def get_sampler(attr_augment_rate, training_set):
    gt_labels = np.concatenate([attr.reshape([1, -1]) for filename, attr in training_set], axis=0)
    example_weights = ((attr_augment_rate - 1) * tensor(gt_labels) + 1).prod(axis=1)
    sampler = WeightedRandomSampler(list(example_weights), len(training_set))
    return sampler

def sigmoid_CE_loss_function(train_logits, gt_labels, weight=None, pos_num=None):
    criterion = F.binary_cross_entropy_with_logits
    return criterion(train_logits, gt_labels, weight)


def L2_loss_function(train_logits, gt_labels, weight=None, pos_num=None):
    criterion = F.mse_loss
    return criterion(torch.sigmoid(train_logits), gt_labels)


def joint_loss_function(train_logits, gt_labels, weight=None, pos_num=None):
    # exclusive_groups = [(0, 1), (2, 5), (6, 8), (9, 10), (32, 37)]
    exclusive_groups = [(6, 8), (9, 10)]
    batchsize, attr_num = gt_labels.shape
    non_exclusive_attr_indexes = [
        i for i in range(attr_num)
        if all(i < x or i > y for x, y in exclusive_groups)
    ]

    softmax_loss = 0.0

    for start, end in exclusive_groups:
        logits = train_logits[:, start:(end+1)]
        labels = gt_labels[:, start:(end+1)]
        labels = torch.argmax(labels, axis=1)
        if pos_num is None:
            w = None
        else:
            p = tensor(pos_num[start:(end+1)]).float()
            w = torch.exp(1 - p / p.sum())
            w = w.to(gt_labels.device)
        softmax_loss += F.cross_entropy(logits, labels, weight=w, reduction='sum')

    logits = train_logits[:, non_exclusive_attr_indexes]
    labels = gt_labels[:, non_exclusive_attr_indexes]
    if weight is None:
        w = None
    else:
        w = weight[:, non_exclusive_attr_indexes]
    sigmoid_loss = F.binary_cross_entropy_with_logits(logits, labels, weight=w, reduction='sum')

    return (softmax_loss + sigmoid_loss) / (float(attr_num) * float(batchsize))


def logits_to_probs_joint(logits):
    exclusive_groups = [(0, 1), (2, 5), (6, 8), (9, 10), (32, 37)]
    attr_num = 55
    non_exclusive_attr_indexes = [
        i for i in range(attr_num)
        if all(i < x or i > y for x, y in exclusive_groups)
    ]

    probs = torch.zeros(logits.shape).to(logits.device)
    for start, end in exclusive_groups:
        group_logits = logits[:, start:(end+1)]
        group_probs = torch.softmax(group_logits, axis=1)
        probs[:, start:(end+1)] = group_probs

    probs[:, non_exclusive_attr_indexes] = torch.sigmoid(logits[:, non_exclusive_attr_indexes])

    return probs


def show_scalars_to_tensorboard(writer, epoch, train_loss, valid_loss, train_result, valid_result, lr_ft, lr_new):
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
    writer.add_scalar('Lr/lr_ft', lr_ft, i)
    writer.add_scalar('Lr/lr_new', lr_new, i)


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
    logits = torch.randn((10, 55))
    probs = logits_to_probs_joint(logits)