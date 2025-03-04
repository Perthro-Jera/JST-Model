import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from dataset import TransUnetDataSet
from models.MedNextV1 import get_MedNeXt_model
from util import seeding
from omegaconf import OmegaConf


# 自定义颜色映射：为每个类别分配一种颜色
COLOR_MAP = [
    (128, 128, 128),  # background (灰色)
    (0, 0, 0),        # 背景 1 (黑色)
    (255, 255, 255),  # 上皮 2 (白色)
    (128, 0, 128),    # 间隙 3 (紫色)
    (255, 0, 0),      # 保护套 4 (红色)
    (0, 0, 255),      # 凸起 5 (蓝色)
    (42, 42, 165),    # 囊肿 6 (棕色)
    (0, 255, 0),      # 基质 7 (绿色)
]


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='MedNeXt', type=str, help='model name')
    parser.add_argument('--model_config', default='config/transUnet.yaml', type=str, help='model config files')
    parser.add_argument('--num_class', default=8, type=int, help="class num")
    parser.add_argument('--frame_height', type=int, default=512, help='the frame height during training.')
    parser.add_argument('--frame_width', type=int, default=1024, help='the frame width during training.')
    parser.add_argument('--fold_num', type=int, default=2, help='fold num for cross-validation.')
    parser.add_argument('--train_pattern', type=str, default="train",choices=["train", "distill_label", "distill_unlabel"])
    parser.add_argument('--test_pattern', type=str, default="predict")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
    parser.add_argument('--batch_size', type=int, default=1, help="batch size.")
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers for DataLoader.")
    parser.add_argument('--pre_train', default=False, type=bool,help="weight initialized by the weight pretrained from ""imageNet")
    parser.add_argument('--checkpoint', type=str, default='checkpoint', help='directory of model weights.')
    parser.add_argument('--result_dir', type=str, default='result', help='directory to save results.')
    parser.add_argument('--seed', default=42, type=int, help='random seed.')

    args = parser.parse_args()
    return args


def merge_config(args):
    """
    将 YAML 配置文件与 argparse 参数合并
    """
    with open(args.model_config, 'r') as f:
        yaml_data = yaml.safe_load(f)
    config = OmegaConf.create(yaml_data)
    config.update(vars(args))
    return config


def path_determine(args):
    param_pattern = 'ade20k_pretrain' if args.pre_train else 'random_initial'
    # model_name = args.model_config.split('/')[-1].split('.')[0]
    model_name = args.model_name
    args.directory_path = os.path.join(args.train_pattern, model_name, param_pattern, str(args.fold_num))
    args.weight_path = os.path.join('checkpoint', args.directory_path, 'MedNeXt_checkpoint_huafen337.pth')
    args.result_dir = os.path.join('result', args.directory_path)
    os.makedirs(args.result_dir, exist_ok=True)
    return args

def load_weight(model, weight_path):
    """
    加载模型权重
    """
    print('==> loading model weights...')
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f'Weights loaded from {weight_path}')
    return model


def colorize_mask(mask, color_map):
    """
    根据类别掩码生成彩色分割图像
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(color_map):
        color_mask[mask == cls_idx] = color
    return color_mask




def predict_segmentation(args):
    """
    预测分割结果并保存
    """
    # 加载配置和数据
    args = merge_config(args)
    args = path_determine(args)
    test_data = TransUnetDataSet(args=args, pattern=args.test_pattern)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # 定义模型
    model = get_MedNeXt_model(args)
    model = load_weight(model, args.weight_path)
    model.eval()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    save_dir = os.path.join(args.result_dir, "colored_predictions")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images, _= batch
            images = images.to(device, non_blocking=True)

            # 获取模型预测结果
            preds, _, _, _, _ = model(images)
            preds = torch.argmax(preds, dim=1)  # 获取预测类别

            # 保存文件时带上批次和样本索引
            for i in range(images.shape[0]):
                original_img_path = os.path.join(save_dir, f"batch_{idx}_img_{i}_original.png")
                pred_img_path = os.path.join(save_dir, f"batch_{idx}_img_{i}_pred.png")

                img = images[i].permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                pred = preds[i].cpu().numpy()
                color_pred = colorize_mask(pred, COLOR_MAP)

                Image.fromarray(img).save(original_img_path)
                Image.fromarray(color_pred).save(pred_img_path)

            if idx % 10 == 0:
                print(f"Processed batch {idx}/{len(test_loader)}")

    print(f"所有预测结果已保存到 {save_dir}")


if __name__ == "__main__":
    args = parse_option()
    seeding(args.seed)
    predict_segmentation(args)
