import argparse
from evaluate_PR_A import evalQueryRes, generateQueryGt
from inference import load_query
from typing import Optional, Union, List

import torch
from data.load_rap_attributes_data_mat import *
from data.AttrDataset import AttrDataset, get_transform
from torch.utils.data import DataLoader
from config import argument_parser, argument_cas_parser
from utils.tools import *
from models.CASNet_fusion import CAS_Fusion_ResNet34
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

set_seed(100)


class AttLR:
    def __init__(
            self,
            optimizer,
            warmup: int,
            max_lr: Union[float, List[float]],
            drop_lr: Optional[List[float]] = None,
    ):
        self.optimizer = optimizer
        self.last_epoch = -1

        self.warmup = warmup
        if isinstance(max_lr, float):
            self.max_lrs = [max_lr]
        else:
            self.max_lrs = list(max_lr)
        assert len(self.max_lrs) == len(self.optimizer.param_groups)
        self.init_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]

        if drop_lr is not None:
            assert(len(drop_lr) == len(self.max_lrs))
            self.drop_lrs = drop_lr
        else:
            self.drop_lrs = self.max_lrs
        # lr = linear(init_lr, max_lr) when epoch < warmup
        # lr = multi * epoch ** -0.5 when epoch >= warmup
        # ==> multi = drop_lr / warmup^(-0.5)
        self.multipliers = [_drop_lr * np.sqrt(warmup) for _drop_lr in self.drop_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for i, param_group in enumerate(self.optimizer.param_groups):
            init_lr, multi, max_lr = self.init_lrs[i], self.multipliers[i], self.max_lrs[i]
            if epoch < self.warmup:
                lr = init_lr + float(epoch) / self.warmup * (max_lr - init_lr)
            else:
                lr = multi * np.power(epoch, -0.5)
            param_group['lr'] = lr

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


QUERY_FILE_NAME = "attr_query_index.txt"
QUERY_LIST = load_query(QUERY_FILE_NAME)


def evaluate_ap(gt_label: np.ndarray, preds_probs: np.ndarray) -> float:
    num_images = preds_probs.shape[0]
    query_res_list = []
    for query in QUERY_LIST:
        confidence_list = np.ones(num_images)
        for index in query:
            confidence_list = confidence_list * preds_probs[:, index]
        query_res_list.append(confidence_list.reshape((1, num_images)))
    query_res = np.concatenate(query_res_list, axis=0)

    query_gt = generateQueryGt(QUERY_FILE_NAME, gt_label.astype(int))
    return evalQueryRes(
        query_res=query_res, 
        image_index=np.array(range(num_images), dtype=int),
        query_gt=query_gt,
    )



def train_model(start_epoch, epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler, path, writer, weights_attr, pos_num):

    def train(model, train_loader, criterion, optimizer, weights_attr):
        model.train()
        loss_meter = AverageMeter()

        gt_list = []
        preds_probs = []

        for step, (images, gt_labels, image_filenames) in enumerate(tqdm(train_loader)):
            images, gt_labels = images.to(device), gt_labels.to(device)
            train_logits = model(images).to(device)
            weights = get_weights(weights_attr, gt_labels)
            train_loss = criterion(train_logits, gt_labels, weight=weights.to(device), pos_num=pos_num)

            train_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()
            loss_meter.update(to_scalar(train_loss))

            gt_list.append(gt_labels.cpu().numpy())
            if args.joint_loss:
                train_probs = logits_to_probs_joint(train_logits)
            else:
                train_probs = torch.sigmoid(train_logits)
            preds_probs.append(train_probs.detach().cpu().numpy())

            # log_interval = 20
            # if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            #     print(f'Step {step}/{len(train_loader)} in Ep {epoch} ',
            #           f'train_loss: {loss_meter.avg: 4f}')
        train_loss = loss_meter.avg

        gt_labels = np.concatenate(gt_list, axis=0)
        preds_probs = np.concatenate(preds_probs, axis=0)

        return train_loss, gt_labels, preds_probs

    def valid(model, valid_loader, criterion):
        model.eval()
        loss_meter = AverageMeter()

        gt_list = []
        preds_probs = []

        with torch.no_grad():
            for step, (images, gt_labels, image_filenames) in enumerate(tqdm(valid_loader)):
                images, gt_labels = images.to(device), gt_labels.to(device)
                gt_list.append(gt_labels.cpu().numpy())
                valid_logits = model(images).to(device)
                valid_loss = criterion(valid_logits, gt_labels)
                if args.joint_loss:
                    valid_probs = logits_to_probs_joint(valid_logits)
                else:
                    valid_probs = torch.sigmoid(valid_logits)
                preds_probs.append(valid_probs.cpu().numpy())
                loss_meter.update(to_scalar(valid_loss))

        valid_loss = loss_meter.avg

        gt_labels = np.concatenate(gt_list, axis=0)
        preds_probs = np.concatenate(preds_probs, axis=0)
        return valid_loss, gt_labels, preds_probs

    maximum = float(-np.inf)
    best_epoch = 0

    result_list = []
    loss_list = []

    for i in range(start_epoch, epoch):

        train_loss, train_gt, train_probs = train(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            weights_attr=weights_attr
        )

        valid_loss, valid_gt, valid_probs = valid(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion
        )

        lr_ft = optimizer.param_groups[0]['lr']
        lr_new = optimizer.param_groups[1]['lr']
        # lr_scheduler.step(metrics=valid_loss)
        lr_scheduler.step()

        train_result = get_pedestrian_metrics(train_gt, train_probs)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)
        valid_result.ap = evaluate_ap(valid_gt, valid_probs)

        output_results_to_screen(i, train_loss, valid_loss, train_result, valid_result)
        show_scalars_to_tensorboard_lr2(writer, i, train_loss, valid_loss, train_result, valid_result, lr_ft, lr_new)

        if valid_result.ap > maximum:
            maximum = valid_result.ap
            best_epoch = i
            best_model = model
            torch.save({
                'epoch': i,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                }, os.path.join(path, 'max.pth'))

        result_list.append((train_result, valid_result))
        loss_list.append((train_loss, valid_loss))

    return best_epoch, loss_list, result_list


def main():

    save_result_path = os.path.join('results', args.model_name)

    annotation_data = loadRAPAttr(args.annotation_file)
    annotation_data = processRAPAttr(annotation_data)
    annotation_data = cleanRAPAttr(annotation_data)

    attr_names_cn = annotation_data['attr_names_cn']
    attr_names_en = annotation_data['attr_names_en']

    train_transform, valid_transform = get_transform(args)

    train_dataset = AttrDataset(
        image_dir_path=args.train_val_images_dir,
        annotation_data=annotation_data['training_set'],
        transform=train_transform,
        attr_names_cn=attr_names_cn,
        attr_names_en=attr_names_en
    )
    valid_dataset = AttrDataset(
        image_dir_path=args.train_val_images_dir,
        annotation_data=annotation_data['validation_set'],
        transform=valid_transform,
        attr_names_cn=attr_names_cn,
        attr_names_en=attr_names_en
    )

    attr_augment_rate = torch.ones([55]).float()
    sampler = None
    shuffle = True
    if args.data_augment:
        attr_augment_rate = get_attr_augment_rate(attr_augment_rate, annotation_data['training_set'])
        sampler = get_sampler(attr_augment_rate, annotation_data['training_set'])
        shuffle = False

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f'training set: {len(train_loader.dataset)}, '
          f'validation set: {len(valid_loader.dataset)}, '
          f'attr_num: {len(train_dataset.attr_names_cn)}')

    weights_attr = [(1, 1) for i in range(len(attr_names_cn))]
    pos_num = [1 for i in range(len(attr_names_cn))]
    if args.weighted_loss:
        pos_num, weights_attr = get_attr_weights(annotation_data['training_set'], attr_augment_rate)

    writer = SummaryWriter(os.path.join('runs', args.model_name))

    model = CAS_Fusion_ResNet34(args.ratio, [True] * 11 + [False] * 35 + [True] * 9)

    # show_image_model_to_tensorboard(writer, model, train_loader)
    model = model.to(device)

    criterion = sigmoid_CE_loss_function
    if args.joint_loss:
        criterion = joint_loss_function

    finetuned_params = []
    new_params = []
    for n, p in model.named_parameters():
        if n.find('classifier') >= 0 or n.find('CAS') >= 0 or n.find('self_mask_block') >=0 \
                or n.find('channel_att') >=0 or n.find('fusion_block') >=0:
            # print(f'Learning rate: {n} -> {args.lr_new}')
            new_params.append(p)
        else:
            # print(f'Learning rate: {n} -> {args.lr_ft}')
            finetuned_params.append(p)
    param_groups = [{'params': finetuned_params, 'lr': args.lr_ft / 2},
                    {'params': new_params, 'lr': args.lr_new / 2}]
    # optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    # lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=0.00001)
    lr_scheduler = AttLR(
        optimizer,
        warmup=8,
        max_lr=[args.lr_ft, args.lr_new],
        drop_lr=[args.lr_ft, args.lr_new],
    )

    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(os.path.join(save_result_path, 'max.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    best_epoch, loss_list, result_list = train_model(
        start_epoch=start_epoch,
        epoch=args.epoch,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        path=save_result_path,
        writer=writer,
        weights_attr=weights_attr,
        pos_num=pos_num
    )

    save_results_to_json(best_epoch, loss_list, result_list, save_result_path)


if __name__ == '__main__':

    parser = argument_cas_parser()

    args = parser.parse_args()

    use_gpu = True
    if use_gpu:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(args.gpu_id)
    else:
        device = torch.device("cpu")

    main()