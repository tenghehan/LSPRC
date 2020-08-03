import torch
import argparse
from config import inference_argument_parser
from data.load_rap_attributes_data_mat import *
from data.AttrDataset import AttrDataset, get_transform
from torch.utils.data import DataLoader
from models.DeepMAR import DeepMAR_ResNet50
from tqdm import tqdm


def load_data():
    annotation_data = loadRAPAttr(args.annotation_file)
    annotation_data = processRAPAttr(annotation_data)
    annotation_data = cleanRAPAttr(annotation_data)

    attr_names_cn = annotation_data['attr_names_cn']
    attr_names_en = annotation_data['attr_names_en']

    _, valid_transform = get_transform(args)

    image_dir_path = {
        'validation_set': args.train_val_images_dir,
        'test_set': args.test_images_dir
    }
    dataset = AttrDataset(
        image_dir_path=image_dir_path['validation_set'],
        annotation_data=annotation_data['validation_set'][0:100],
        transform=valid_transform,
        attr_names_cn=attr_names_cn,
        attr_names_en=attr_names_en
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return data_loader


def attributes_recognition_results(preds_probs):
    with open('attributes_recognition_val_gt.txt', 'w') as file:
        for i in range(len(preds_probs)):
            print(i, end='', file=file)
            for prob in preds_probs[i]:
                print(',%.0f' % prob, end='', file=file)
            print('', file=file)


def query_results(query_list, preds_probs):
    results = []
    for query in query_list:
        confidence_list = np.ones(preds_probs.shape[0])
        for index in query:
            confidence_list = confidence_list * preds_probs[:, index]
        results.append(sorted(list(enumerate(confidence_list.tolist())), key=lambda t:t[1], reverse=True))

    with open('query_val_gt.txt', 'w') as file:
        for i in range(len(results)):
            print(i, end='', file=file)
            for (image_index, confidence) in results[i]:
                print(',%d' % image_index, end='', file=file)
                print(',%.0f' % confidence, end='', file=file)
            print('', file=file)


def get_groundtruth(dataloader):
    gt_list = []

    with torch.no_grad():
        for step, (images, gt_labels, image_filenames) in enumerate(tqdm(dataloader)):
            gt_labels = gt_labels.to(device)
            gt_list.append(gt_labels.cpu().numpy())

    return np.concatenate(gt_list, axis=0)


def load_query(query_file_path):
    query_list = []
    file = open(query_file_path, 'r')
    lines = file.readlines()
    for line in lines:
        query_list.append([int(x) for x in line.strip().split(' ')])

    return query_list


def main():
    dataloader = load_data()
    gt_labels = get_groundtruth(dataloader)
    attributes_recognition_results(gt_labels)

    query_list = load_query('attr_query_index.txt')
    query_results(query_list, gt_labels)


if __name__ == '__main__':

    parser = inference_argument_parser()
    args = parser.parse_args()

    use_gpu = False
    if use_gpu:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(args.gpu_id)
    else:
        device = torch.device("cpu")

    main()



