import argparse
import torch
from data.load_rap_attributes_data_mat import *
from data.AttrDataset import AttrDataset, get_transform
from torch.utils.data import DataLoader
from config import argument_parser
from utils.tools import *
from models.DeepMAR import DeepMAR_ResNet50
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

set_seed(100)


def train_model(start_epoch, epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler, path, writer, weights_attr, pos_num):

    def train(model, train_loader, criterion, optimizer, weights_attr):
        model.train()
        loss_meter = AverageMeter()

        gt_list = []
        preds_probs = []

        for step, (images, gt_labels, image_filenames) in enumerate(tqdm(train_loader)):
            images, gt_labels = images.to(device), gt_labels.to(device)
            train_logits = model(images)
            weights = get_weights(weights_attr, gt_labels)
            train_loss = criterion(train_logits, gt_labels, weight=weights.to(device), pos_num=pos_num.to(device))

            train_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()
            loss_meter.update(to_scalar(train_loss))

            gt_list.append(gt_labels.cpu().numpy())
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
                valid_logits = model(images)
                valid_loss = criterion(valid_logits, gt_labels)
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

        # lr_scheduler.step(metrics=valid_loss, epoch=i)
        lr_scheduler.step(metrics=valid_loss)

        train_result = get_pedestrian_metrics(train_gt, train_probs)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

        output_results_to_screen(i, train_loss, valid_loss, train_result, valid_result)
        show_scalars_to_tensorboard(writer, i, train_loss, valid_loss, train_result, valid_result)

        if valid_result.ma > maximum:
            maximum = valid_result.ma
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

    print(f'training set: {len(train_loader.dataset)}, '
          f'validation set: {len(valid_loader.dataset)}, '
          f'attr_num: {len(train_dataset.attr_names_cn)}')

    weights_attr = [(1, 1) for i in range(len(attr_names_cn))]
    pos_num = [1 for i in range(len(attr_names_cn))]
    if args.weighted_loss:
        pos_num, weights_attr = get_attr_weights(annotation_data['training_set'])

    writer = SummaryWriter(os.path.join('runs', args.model_name))

    model = DeepMAR_ResNet50(len(train_dataset.attr_names_cn))

    show_image_model_to_tensorboard(writer, model, train_loader)
    model = model.to(device)

    criterion = sigmoid_CE_loss_function
    if args.joint_loss:
        criterion = joint_loss_function

    finetuned_params = []
    new_params = []
    for n, p in model.named_parameters():
        if n.find('classifier') >= 0:
            new_params.append(p)
        else:
            finetuned_params.append(p)
    param_groups = [{'params': finetuned_params, 'lr': args.lr_ft},
                    {'params': new_params, 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

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

    parser = argument_parser()

    args = parser.parse_args()

    use_gpu = True
    if use_gpu:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(args.gpu_id)
    else:
        device = torch.device("cpu")

    main()