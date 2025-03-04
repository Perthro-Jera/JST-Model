import os
import cv2
import argparse
import yaml
import numpy as np
import torch
import json
import timm
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from dataset import TransformerDataSet
from dataset import TransformerLBPDataSet
import models.resnet as resnet
import models.resnet1 as resnet1
import models.convnext as convnext
from models.ae_densenet121 import ModifiedDenseNet121
import models.swin_transformer as swin_transformer
import models.cswin as cswin
from models.Unet import UNetClassifier
from models.drae import DRAE
from models.seg_drae import FusionModel
from models.fusion import FeatureFusion
from torch.utils.data import DataLoader
from models.MedNextV1 import get_MedNeXt_model
from models.Multi_GlaucNet import ClassifierWithSegmentation
from util import save_checkpoint, seeding
from criterion import CrossEntropy, OhemCrossEntropy
from omegaconf import OmegaConf
from util import DataUpdater, get_confusion_matrix
from medpy.metric.binary import dc, hd95, asd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_auc_score
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray  # 导入灰度转换函数
#from GradCAM import GradCAM
import matplotlib.pyplot as plt
from gradcam_utils import GradCAM, show_cam_on_image
from GradCAMfc import GradCAM_FC
#from CAM import generate_grad_cam


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='seg_drae', type=str,
                        choices=['resnet18', 'convnext_pico', 'vit_base', 'vmamba_tiny', 'GlaucNet', 'ae_densenet121'],
                        help='the model we used.')
    parser.add_argument('--test_file', default='huaxi',
                        choices=['internal', 'huaxi', 'xiangya', 'xiangyang'],
                        help="the test file from different center", type=str)
    parser.add_argument('--train_pattern', type=str, default="seg_drae",
                        choices=["train", "st_train"])
    parser.add_argument('--model_config', default='config/transUnet.yaml', type=str, help='model config files')
    parser.add_argument('--num_class', default=5, type=int, help="class num")

    # parameters for data
    parser.add_argument('--frame_height', type=int, default=512, help='the frame height we used during training.')
    parser.add_argument('--frame_width', type=int, default=1024, help='the frame width we used during training.')
    parser.add_argument('--ignore_label', type=int, default=-1, help='ignoring the label of pixels')

    # folder num，以便用于内部数据集的交叉验证
    parser.add_argument('--fold_num', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='the fold num we used for cross validation.')

    parser.add_argument('--test_pattern', type=str, default="test")
    parser.add_argument('--pre_train', default=True, type=bool,
                        help="weight initialized by the weight pretrained from "
                             "imageNet")

    # test configuration
    parser.add_argument('--batch_size', type=int, default=1, help="training batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="data loader thread")

    parser.add_argument('--gpu', type=int, default=0, help='the gpu number will be used')
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


def path_determine(args):
    param_pattern = 'pretrain' if args.pre_train else 'random_initial'
    # model_name = args.model_config.split('/')[-1].split('.')[0]
    model_name = args.model_name
    args.directory_path = os.path.join(args.train_pattern, model_name, param_pattern, str(args.fold_num))
    args.weight_path = os.path.join('checkpoint', args.directory_path)
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


# 加载分割模型训练权重
def load_seg_weight(model):
    directory_path = os.path.join('train', 'MedNeXt', 'random_initial', '2')
    weight_path = os.path.join('checkpoint', directory_path, 'MedNeXt_checkpoint_huafen337.pth')
    # weight_path = os.path.join(weight_path, 'segmodel_checkpoint.pth')
    # 加载模型权重
    print('==> loading teacher model')
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_dict = checkpoint['state_dict']
    state = model.load_state_dict(model_dict, strict=False)
    print('loading checkpoint from {}'.format(weight_path))
    print(state)
    return model


def load_cls_weight(model, weight_path):
    weight_path = 'checkpoint/drae'
    weight_path = os.path.join(weight_path, 'checkpoint0222.pth')
    # 加载模型模型权重
    print('==> loading teacher model')
    checkpoint = torch.load(weight_path, map_location='cpu')
    print(checkpoint.keys())  # 查看所有保存的键
    model_dict = checkpoint['state_dict']
    state = model.load_state_dict(model_dict, strict=False)
    print('loading checkpoint from {}'.format(weight_path))
    print(state)
    return model


def convert_to_serializable(obj):
    """将对象转换为可 JSON 序列化的原生 Python 数据类型"""
    if isinstance(obj, (np.float32, np.float64, torch.Tensor)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    else:
        return obj

def test_cls(segmodel, clsmodel, test_loader, config):
    segmodel.eval()
    clsmodel.eval()
    ave_loss = DataUpdater()
    device = next(segmodel.parameters()).device
    num_class = 5  # 你的分类数是 5
    resnet_layers = clsmodel.resnet18.encoder[7]
    print(clsmodel)
    # 获取 Encoder 层
    encoder_layers = list(clsmodel.drae_encoder.encoder.children())  # 访问 encoder 部分


    # 初始化 FC-CAM

    # 打印层数和层的内容
    # print(f"Number of layers in encoder: {len(encoder_layers)}")
    # print(f"Layers in encoder: {encoder_layers}")

    # 选择多个层进行 Grad-CAM
    autoencoder_layers = [
        encoder_layers[4],  # Conv2d(64, 128)
        encoder_layers[6],  # Conv2d(128, 256)
        encoder_layers[8],  # Conv2d(128, 256)
    ]

    #cam = GradCAM(model=clsmodel, target_layers=resnet_layers, device=device, reshape_transform= None)
    scam = GradCAM(model=clsmodel, target_layers=resnet_layers, device=device, reshape_transform= None)
    tcam = GradCAM(model=clsmodel, target_layers=autoencoder_layers, device=device, reshape_transform=None)



    for idx, batch in enumerate(test_loader):
        images, texture, labels, images_path, _ = batch
        images = images.to(device, non_blocking=True)
        labels = labels.clone().detach().long().to(device, non_blocking=True)

        # 分类模型预测
        if args.train_pattern in ['mask_train', 'train']:
            #images = images[:, 0:1, :, :]
            pred_cls = clsmodel(images)
        elif args.train_pattern == 'seg_drae':
            seg_output, _, _, _, _ = segmodel(images)
            # print(f"Image shape: {images.shape}")  # 输出图像形状
            # 转为灰度图
            gray_image = images[:, 0:1, :, :]  # 取 B 通道
            pred_mask = torch.argmax(seg_output, dim=1)  # [batch_size, height, width]
            pred_mask = pred_mask.unsqueeze(1).float()
            # pred_mask = pred_mask.repeat(1, 2, 1, 1) # 扩展通道，变为 [12, 3, 512, 1024]
            # pred_cls = clsmodel(gray_image, pred_mask)
            pred_mask = pred_mask.repeat(1, 3, 1, 1)  # 扩展通道，变为 [12, 3, 512, 1024]


        #生成热力图
        s_cam = scam(input_tensor=gray_image, pred_mask=pred_mask, target_category=labels)
        t_cam = tcam(input_tensor=gray_image, pred_mask=pred_mask, target_category=labels)


        s_cam = s_cam[0, :]
        t_cam = t_cam[0, :]

        cam = 0.5 * s_cam + 0.5 * t_cam
        img = cv2.imread(images_path[0])  # [H, W, 3]
        img = cv2.resize(img, (config.frame_width, config.frame_height))

        visualization1 = show_cam_on_image(img / 255., s_cam, use_rgb=True)
        visualization2 = show_cam_on_image(img / 255., t_cam, use_rgb=True)
        visualization3 = show_cam_on_image(img / 255, cam, use_rgb=True)

        # 获取保存文件的基本信息
        name = images_path[0].split('/')[-1]  # 提取文件名
        target_category = str(labels.item())  # 将目标类别转换为字符串
        heatmap_dir1 = os.path.join('result', 'heatmaps1', config.test_file, 'resnet_heatmaps', target_category)
        heatmap_dir2 = os.path.join('result', 'heatmaps1', config.test_file, 'ae_heatmaps', target_category)
        heatmap_dir3 = os.path.join('result', 'heatmaps1', config.test_file, 'fusion_heatmaps', target_category)
        # 确保保存目录存在
        os.makedirs(heatmap_dir1, exist_ok=True)
        os.makedirs(heatmap_dir2, exist_ok=True)
        os.makedirs(heatmap_dir3, exist_ok=True)

        # 保存热力图
        heatmap_path1 = os.path.join(heatmap_dir1, name.replace('.png', '_heatmap.jpg'))
        cv2.imwrite(heatmap_path1, cv2.cvtColor(visualization1, cv2.COLOR_RGB2BGR))
        # 打印保存信息
        print(f"Heatmap saved to: {heatmap_path1}")
        heatmap_path2 = os.path.join(heatmap_dir2, name.replace('.png', '_heatmap.jpg'))
        cv2.imwrite(heatmap_path2, cv2.cvtColor(visualization2, cv2.COLOR_RGB2BGR))
        # 打印保存信息
        print(f"Heatmap saved to: {heatmap_path2}")
        heatmap_path3 = os.path.join(heatmap_dir3, name.replace('.png', '_heatmap.jpg'))
        cv2.imwrite(heatmap_path3, cv2.cvtColor(visualization3, cv2.COLOR_RGB2BGR))
        # 打印保存信息
        print(f"Heatmap saved to: {heatmap_path3}")
        # plt.imshow(visualization)
        # plt.show()

        if idx % 10 == 0:
            print(f"Processing batch {idx} / {len(test_loader)}")
    return 0


def worker(args):
    # 合并yaml文件配置参数
    args = path_determine(args)
    args = merge_config(args)
    # 定义dataloader
    test_data = TransformerDataSet(args=args, pattern=args.test_pattern)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # 定义模型
    segmodel = get_MedNeXt_model()
    if args.model_name == 'resnet18':
        clsmodel = getattr(resnet, args.model_name)(pretrained=args.pre_train, num_classes=args.num_class)
    elif args.model_name in ['convnext_pico', 'convnext_little', 'convnext_lite']:
        clsmodel = getattr(convnext, args.model_name)(pretrained=args.pre_train, num_classes=args.num_class)
    elif args.model_name == 'unet':
        clsmodel = UNetClassifier(in_channels=3, num_classes=args.num_class)
    elif args.model_name in ['vmamba_tiny']:
        clsmodel = getattr(vmamba, args.model_name)(pretrained=args.pre_train, num_classes=args.num_class)
    elif args.model_name in ['vit_tiny']:
        clsmodel = timm.create_model(model_name='vit_tiny_patch16_224', pretrained=False, num_classes=args.num_class,
                                     img_size=(args.frame_height, args.frame_width))
    elif args.model_name in ['vit_base']:
        clsmodel = timm.create_model(model_name='vit_base_patch16_224', pretrained=False, num_classes=args.num_class,
                                     img_size=(args.frame_height, args.frame_width))
    elif args.model_name in ['vim_tiny']:
        clsmodel = getattr(vim, args.model_name)(pretrained=args.pre_train, num_classes=args.num_class)
    elif args.model_name == 'efficientnet_v2':
        clsmodel = timm.create_model('efficientnetv2_rw_s', pretrained=False, num_classes=args.num_class)
    elif args.model_name == 'densenet121':
        clsmodel = timm.create_model('densenet121', pretrained=False, num_classes=args.num_class, )
    elif args.model_name == 'swin_base':
        clsmodel = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=args.num_class,
                                     img_size=(args.frame_height, args.frame_width))
    elif args.model_name == 'seg_drae':
        clsmodel = FusionModel()
    elif args.model_name == 'GlaucNet':
        clsmodel = ClassifierWithSegmentation()
    elif args.model_name == 'ae_densenet121':
        clsmodel = ModifiedDenseNet121(pretrained=True, num_classes=5)

    segmodel = load_seg_weight(segmodel)
    clsmodel = load_cls_weight(clsmodel, args.weight_path)


    device = torch.device('cuda:' + str(args.gpu))

    segmodel = segmodel.to(device)
    clsmodel = clsmodel.to(device)


    results = test_cls(segmodel, clsmodel, test_loader, args)


if __name__ == '__main__':
    # set seed for reproduce
    args = parse_option()
    seeding(args.seed)
    # for fold_num in [0]:
    #     args.fold_num = fold_num
    #     worker(args)
    worker(args)