import os
import argparse
import yaml
import numpy as np
import torch
import json
import torch.nn.functional as F
from dataset import TransUnetDataSet
from torch.utils.data import DataLoader
from models.MedNextV1 import get_MedNeXt_model
from util import save_checkpoint, seeding
from criterion import CrossEntropy, OhemCrossEntropy
from omegaconf import OmegaConf
from util import DataUpdater, get_confusion_matrix
from medpy.metric.binary import dc, hd95, asd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='MedNeXt', type=str, help='model name')
    parser.add_argument('--model_config', default='config/transUnet.yaml', type=str, help='model config files')
    parser.add_argument('--num_class', default=8, type=int, help="class num")

    # parameters for data
    parser.add_argument('--frame_height', type=int, default=512, help='the frame height we used during training.')
    parser.add_argument('--frame_width', type=int, default=1024, help='the frame width we used during training.')
    parser.add_argument('--ignore_label', type=int, default=-1, help='ignoring the label of pixels')

    # folder num，以便用于内部数据集的交叉验证
    parser.add_argument('--fold_num', type=int, default=2, choices=[0, 1, 2, 3, 4],
                        help='the fold num we used for cross validation.')
    parser.add_argument('--train_pattern', type=str, default="train",
                        choices=["train", "distill_label", "distill_unlabel"])
    parser.add_argument('--test_pattern', type=str, default="test")
    parser.add_argument('--pre_train', default=False, type=bool,
                        help="weight initialized by the weight pretrained from "
                             "imageNet")

    # test configuration
    parser.add_argument('--batch_size', type=int, default=1, help="training batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="data loader thread")

    parser.add_argument('--gpu', type=int, default=1, help='the gpu number will be used')
    parser.add_argument('--checkpoint', type=str, default='checkpoint',
                        help='the directory to save the model weights.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--result_dir', default='result', type=str,
                        help='the directory of the model testing performance')

    # loss
    parser.add_argument('--use_ohem', default=False, type=bool, help='whether use ohem for cross entropy')
    parser.add_argument('--ohemthres', default=0.9, type=float, help='threshold for ohem')
    parser.add_argument('--ohemkeep', type=int, default=125000, help='minimal numbers of ohem')

    args = parser.parse_args()
    return args


def calculate_dice(tp, pos, res):
    """
    计算 Dice 系数
    :param tp: True Positives 数组
    :param pos: Ground Truth 总数
    :param res: 预测结果总数
    :return: Dice 系数
    """
    return (2 * tp) / np.maximum(1.0, pos + res)


def path_determine(args):
    param_pattern = 'ade20k_pretrain' if args.pre_train else 'random_initial'
    # model_name = args.model_config.split('/')[-1].split('.')[0]
    model_name = args.model_name
    args.directory_path = os.path.join(args.train_pattern, model_name, param_pattern, str(args.fold_num))
    args.weight_path = os.path.join('checkpoint', args.directory_path, 'MedNeXt_checkpoint_huafen337.pth')
    args.result_dir = os.path.join('result', args.directory_path)
    os.makedirs(args.result_dir, exist_ok=True)
    return args


def merge_config(args):
    """
    将yaml的模型配置文件与arg parser合并在一起
    :param args:
    :return:
    """
    with open(args.model_config, 'r') as f:
        yaml_data = yaml.safe_load(f)
    # 使用 OmegaConf 将 YAML 数据转换为 ConfigNode 对象
    config = OmegaConf.create(yaml_data)
    config.update(vars(args))
    return config


def load_weight(model, weight_path):
    # 加载模型模型权重
    print('==> loading teacher model')
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_dict = checkpoint['state_dict']
    state = model.load_state_dict(model_dict, strict=False)
    print('loading checkpoint from {}'.format(weight_path))
    print(state)
    return model


def test_seg(model, test_loader, criterion, config):
    model.eval()
    ave_loss = DataUpdater()
    nums = config.MODEL.NUM_OUTPUTS
    device = next(model.parameters()).device
    confusion_matrix = np.zeros((config.num_class, config.num_class, nums))

    # 分别存储每个类别的 HD95 和 ASD
    hd95_scores = [[] for _ in range(config.num_class)]
    asd_scores = [[] for _ in range(config.num_class)]

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images, masks, labels, _, _ = batch
            H, W = images.size(2), images.size(3)
            images = images.to(device, non_blocking=True)
            masks = masks.long().to(device, non_blocking=True)  # [B, H, W]
            #labels = labels.clone().detach().long().to(device, non_blocking=True)

            pred_seg, _, _, _, _ = model(images)
            pred_seg = F.interpolate(input=pred_seg, size=(masks.size(1), masks.size(2)), mode='bilinear',
                                     align_corners=True)
            seg_losses = criterion(pred_seg, masks)

            if not isinstance(pred_seg, (list, tuple)):
                pred_seg = [pred_seg]
            for i, x in enumerate(pred_seg):
                confusion_matrix[..., i] += get_confusion_matrix(
                    masks,
                    x,
                    (H, W),
                    config.num_class,
                    config.ignore_label,
                )

                pred_np = torch.argmax(x, dim=1).cpu().numpy()
                masks_np = masks.cpu().numpy()

                for cls in range(1, config.num_class):  # skipping background (class 0)
                    pred_bin = (pred_np == cls).astype(np.uint8)
                    mask_bin = (masks_np == cls).astype(np.uint8)

                    if np.sum(mask_bin) > 0:  # Ensure class exists in the mask
                        if np.sum(pred_bin) > 0:  # Ensure class exists in the prediction
                            hd95_score = hd95(pred_bin, mask_bin)
                            asd_score = asd(pred_bin, mask_bin)
                            hd95_scores[cls].append(hd95_score)
                            asd_scores[cls].append(asd_score)
                        else:
                            print(f"类别 {cls} 在预测中未找到前景对象，跳过 HD95 和 ASD 计算。")
                    else:
                        print(f"类别 {cls} 在标签中未找到前景对象，跳过 HD95 和 ASD 计算。")

            if idx % 10 == 0:
                print(f"Processing batch {idx} / {len(test_loader)}")
            loss = seg_losses.mean()
            ave_loss.update(loss.item())

    print("平均损失:", ave_loss.avg)
    results = {}
    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])

        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()

        dice_array = calculate_dice(tp, pos, res)
        mean_dice = dice_array.mean()

        results[f"seg_output_{i}"] = {
            "IoU_array": IoU_array.tolist(),
            "mean_IoU": mean_IoU,
            "dice_array": dice_array.tolist(),
            "mean_dice": mean_dice,
        }
        print(
            '{} 类别 IoU: {} 平均 IoU: {} Dice: {} 平均 Dice: {}'.format(i, IoU_array, mean_IoU, dice_array, mean_dice))

    hd95_scores = [np.mean(scores) if scores else 0 for scores in hd95_scores]
    asd_scores = [np.mean(scores) if scores else 0 for scores in asd_scores]

    mean_hd95 = np.mean(hd95_scores)
    mean_asd = np.mean(asd_scores)
    print(f"平均 HD95: {mean_hd95}")
    print(f"平均 ASD: {mean_asd}")

    results['overall'] = {
        "mean_hd95": mean_hd95,
        "mean_asd": mean_asd,
        "hd95_scores": hd95_scores,
        "asd_scores": asd_scores
    }



    # Save the results to a JSON file
    results_file = os.path.join(config.result_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            "average_loss": ave_loss.avg,
            "results": results
        }, f, indent=4)

    return ave_loss.avg, mean_IoU, IoU_array, mean_dice, dice_array, mean_hd95, mean_asd


def worker(args):
    # 合并yaml文件配置参数
    args = path_determine(args)
    args = merge_config(args)
    # 定义dataloader
    test_data = TransUnetDataSet(args=args, pattern=args.test_pattern)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # 定义模型
    model = get_MedNeXt_model(args)
    print(f"fold_num: {args.fold_num}")
    model = load_weight(model, args.weight_path)

    # label weight，应该为长度为8的列表，表示每个分割类的权重
    # class_weight = torch.FloatTensor([1, 1, 1.5, 10.6, 4.1, 3, 10, 0.87])
    class_weight = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1])
    # 定义损失函数
    if args.use_ohem:
        criterion = OhemCrossEntropy(ignore_label=args.ignore_label,
                                     thres=args.ohemthres,
                                     min_kept=args.ohemkeep,
                                     weight=class_weight)
    else:
        criterion = CrossEntropy(ignore_label=args.ignore_label, weight=class_weight)

    device = torch.device('cuda:' + str(args.gpu))
    model = model.to(device)
    criterion = criterion.to(device)
    results = test_seg(model, test_loader, criterion, args)

    avg_loss, mean_IoU, IoU_array, mean_dice, dice_array, mean_hd95, mean_asd = results

    print(f"平均 Dice: {mean_dice}")


if __name__ == '__main__':
    # set seed for reproduce
    args = parse_option()
    seeding(args.seed)
    #for fold_num in [2]:
    #args.fold_num = fold_num
    worker(args)
