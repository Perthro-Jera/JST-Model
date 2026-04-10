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
from dataset import TransformerDataSet,EyeOCTClsDataset
import models.resnet as resnet
import models.convnext as convnext
from models.ae_densenet121 import ModifiedDenseNet121
from models.seg_drae import FusionModel
from torch.utils.data import DataLoader
from models.MedNextV1 import get_MedNeXt_model
from models.semi_MedNeXt import KnowSAM
from models.Multi_GlaucNet import ClassifierWithSegmentation
from util import save_checkpoint, seeding
from criterion import CrossEntropy, OhemCrossEntropy
from omegaconf import OmegaConf
from util import DataUpdater, get_confusion_matrix
from medpy.metric.binary import dc, hd95, asd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray  # 导入灰度转换函数
import matplotlib.pyplot as plt
from torchsummary import summary
import time


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='seg_drae', type=str,
                        choices=['resnet18', 'convnext_pico',  'vit_base', 'vmamba_tiny', 'GlaucNet', 'ae_densenet121'],
                        help='the model we used.')
    parser.add_argument('--dataset', default='eye', choices=['eye', 'cervix'],
                        help='选择数据集类型，用于自动设置类别数等参数')
    parser.add_argument('--test_file', default='Retinal',
                        choices=['internal', "Retinal" ,"OCT2017"],
                        help="the test file from different center", type=str)
    parser.add_argument('--train_pattern', type=str, default="seg_drae",
                        choices=["train", "st_train","Glaucnet"])
    parser.add_argument('--model_config', default='config/transUnet.yaml', type=str, help='model config files')
    parser.add_argument('--num_class', default=2, type=int, help="class num")


    # parameters for data
    parser.add_argument('--frame_height', type=int, default=256, help='the frame height we used during training.')
    parser.add_argument('--frame_width', type=int, default=512, help='the frame width we used during training.')
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

    parser.add_argument('--seg_model', default='KnowSAM', choices=['MedNeXt', 'KnowSAM'], help='选择分割模型')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=8)

    args = parser.parse_args()
    # ---------------- 自动适配 dataset ----------------
    if args.dataset == 'eye':
        args.num_class = 2  # AMD / DME / NORMAL
        args.seg_num_classes = 6  # 眼科 OCT 分割 6 类
    elif args.dataset == 'cervix':
        args.num_class = 5  # 宫颈 5 类
        args.seg_num_classes = 8  # 宫颈 OCT 分割 8 类
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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


#加载分割模型训练权重
def load_seg_weight(model):
    directory_path = os.path.join('train', 'MedNeXt', 'random_initial', '2')
    #weight_path = os.path.join('checkpoint',directory_path, 'MedNeXt_checkpoint_huafen337.pth')
    weight_path = os.path.join('checkpoint', directory_path, 'MedNeXt_checkpoint_256_300.pth')
    #weight_path = os.path.join(weight_path, 'segmodel_checkpoint.pth')
    # 加载模型权重
    print('==> loading teacher model')
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_dict = checkpoint['state_dict']
    state = model.load_state_dict(model_dict, strict=False)
    print('loading checkpoint from {}'.format(weight_path))
    print(state)
    return model

def load_KnowSAM_weight(model):

    weight_path = os.path.join('checkpoint/train/KnowSAM/eye/oct_full/SGDL_best_model.pth')
    # 加载模型权重
    print('==> loading seg model')
    state = model.load_state_dict(torch.load(weight_path), strict=True)
    print('loading checkpoint from {}'.format(weight_path))
    print(state)
    return model

def load_cls_weight(model, weight_path):
    weight_path = 'checkpoint/baseline'
    weight_path = os.path.join(weight_path, 'checkpoint_eye_oct5k_2_30_01_fpn_cross.pth')
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    total = 0
    print("模块参数统计如下：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:60s} \t参数量: {param.numel()}")
            total += param.numel()
    print(f"\n总可训练参数量：{total:,}")
    return total

def test_cls(segmodel, clsmodel, test_loader, criterion, config):
    segmodel.eval()
    clsmodel.eval()
    # print(f"Segmentation model parameters: {count_parameters(segmodel):,}")
    # print(f"Classification model parameters: {count_parameters(clsmodel):,}")
    ave_loss = DataUpdater()
    device = next(segmodel.parameters()).device
    num_class = config.num_class

    total_infer_time = 0
    total_samples = 0
    total_seg_infer_time = 0
    total_cls_infer_time = 0

    all_labels = []
    all_predictions = []
    all_probs = []

    all_labels_binary = []
    all_predictions_binary = []
    all_probs_binary = []

    #多类别
    #cm_n = torch.zeros(num_class, num_class, dtype=torch.int64)

    cm_2 = torch.zeros(2, 2, dtype=torch.int64)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images, texture, labels, _, _ = batch
            images = images.to(device, non_blocking=True)
            texture = texture.to(device, non_blocking=True)
            labels = labels.clone().detach().long().to(device, non_blocking=True)

            start_total = time.perf_counter()

            if config.train_pattern == 'seg_drae':
                start_seg = time.perf_counter()
                seg_output, _, _, _, _ = segmodel(images)
                end_seg = time.perf_counter()
                total_seg_infer_time += (end_seg - start_seg)

                gray_image = images[:, 0:1, :, :]
                pred_mask = torch.argmax(seg_output, dim=1).unsqueeze(1).float()
                seg_image = pred_mask.repeat(1, 3, 1, 1)
                pred_mask[(pred_mask == 5)] = 0
                pred_mask[(pred_mask == 1) | (pred_mask == 2) | (pred_mask == 3) | (pred_mask == 4)] = 1
                extracted_image = gray_image * pred_mask
                #extracted_image = gray_image

                start_cls = time.perf_counter()
                pred_cls, _, _ = clsmodel(extracted_image, seg_image)
                end_cls = time.perf_counter()
                total_cls_infer_time += (end_cls - start_cls)
            elif config.train_pattern == 'Glaucnet':
                gray_image = images[:, 0:1, :, :]
                seg_output, _, _, _, _ = segmodel(images)
                pred_mask = torch.argmax(seg_output, dim=1)  # [batch_size, height, width]
                pred_mask = pred_mask.unsqueeze(1).float()
                pred_mask = pred_mask.repeat(1, 2, 1, 1)  # 扩展通道，变为 [12, 3, 512, 1024]
                # 其他训练模式
                pred_cls = clsmodel(gray_image, pred_mask)
            else:
                #images = images[:, 0:1, :, :]
                pred_cls = clsmodel(images)


            end_total = time.perf_counter()
            total_infer_time += (end_total - start_total)
            total_samples += images.size(0)

            # 分类损失
            cls_losses = criterion(pred_cls, labels)
            loss = cls_losses.mean()
            ave_loss.update(loss.item())

            # 多分类预测
            pred_cls_nd = torch.argmax(pred_cls, dim=1)
            probs_nd = F.softmax(pred_cls, dim=1)

            #计算 n 类混淆矩阵
            # for t, p in zip(labels.view(-1), pred_cls_nd.view(-1)):
            #     cm_n[t.long(), p.long()] += 1

            # 二分类标签和预测
            if config.dataset == 'eye':
                # NORMAL=0（阴性），AMD/DME=1（阳性）
                labels_binary = (labels > 0).long()  # 1 或 2 都算阳性
                pred_cls_binary = (pred_cls_nd > 0).long()
                probs_binary = probs_nd[:, 1:].sum(dim=1)  # AMD+DME 的概率求和
            elif config.dataset == 'cervix':
                # Class 3,4=阳性，其余=阴性
                labels_binary = (labels >= 3).long()
                pred_cls_binary = (pred_cls_nd >= 3).long()
                probs_binary = probs_nd[:, 3:5].sum(dim=1)

            for t, p in zip(labels_binary.view(-1), pred_cls_binary.view(-1)):
                cm_2[t.long(), p.long()] += 1

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(pred_cls_nd.cpu().numpy())
            all_probs.extend(probs_nd.cpu().numpy())

            all_labels_binary.extend(labels_binary.cpu().numpy())
            all_predictions_binary.extend(pred_cls_binary.cpu().numpy())
            all_probs_binary.extend(probs_binary.cpu().numpy())

            if idx % 10 == 0:
                print(f"Processing batch {idx} / {len(test_loader)}")

    # 多分类报告
    # target_names = [f"Class {i}" for i in range(config.num_class)]
    # acc_n = accuracy_score(all_labels, all_predictions)
    # report_n = classification_report(all_labels, all_predictions, target_names=target_names, digits=4)
    # print(f"{config.num_class}-class Confusion Matrix:")
    # print(cm_n.numpy())
    # print(f"{config.num_class}-class Accuracy: {acc_n * 100:.2f}%")
    # print(report_n)

    # 二分类指标
    acc_2 = accuracy_score(all_labels_binary, all_predictions_binary)
    tn, fp, fn, tp = cm_2.numpy().ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    auc = roc_auc_score(all_labels_binary, all_probs_binary) if len(set(all_labels_binary)) > 1 else 0

    print("2-class Confusion Matrix:")
    print(cm_2.numpy())
    print(f"2-class Accuracy: {acc_2 * 100:.2f}%")
    print(f"Sensitivity: {sensitivity * 100:.2f}%")
    print(f"Specificity: {specificity * 100:.2f}%")
    print(f"PPV (Precision): {ppv * 100:.2f}%")
    print(f"NPV: {npv * 100:.2f}%")
    print(f"AUC: {auc:.4f}")

    avg_infer_time = total_infer_time / total_samples
    avg_seg_time = total_seg_infer_time / total_samples
    avg_cls_time = total_cls_infer_time / total_samples
    print(f"Average inference time per image: {avg_infer_time * 1000:.2f} ms")
    print(f"Average segmentation model inference time per image: {avg_seg_time * 1000:.2f} ms")
    print(f"Average classification model inference time per image: {avg_cls_time * 1000:.2f} ms")

    return ave_loss.avg



def worker(args):
    # 合并yaml文件配置参数
    args = path_determine(args)
    args = merge_config(args)
    # 定义dataloader
    test_data = EyeOCTClsDataset(args=args, pattern=args.test_pattern)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)



    if args.model_name == 'resnet18':
        clsmodel = getattr(resnet, args.model_name)(pretrained=args.pre_train, num_classes=args.num_class)
    elif args.model_name in ['convnext_pico', 'convnext_little', 'convnext_lite']:
        clsmodel = getattr(convnext, args.model_name)(pretrained=args.pre_train, num_classes=args.num_class)
    elif args.model_name == 'unet':
        clsmodel = UNetClassifier(in_channels=3, num_classes=args.num_class)
    elif args.model_name in ['vmamba_tiny']:
        clsmodel = getattr(vmamba, args.model_name)(pretrained=args.pre_train, num_classes=args.num_class)
    elif args.model_name in ['vit_tiny']:
        clsmodel = timm.create_model(model_name='vit_tiny_patch16_224', pretrained=False, num_classes=args.num_class, img_size=(args.frame_height,args.frame_width))
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
    elif args.model_name =='seg_drae':
        clsmodel = FusionModel(
            input_channels=1,  # 原来的设置
            latent_dim=256,  # 原来的设置
            class_num=args.num_class, # ⚡关键：根据数据集自动设置类别数
            texture_extractor = 'fpn'  # 重构或者特征金字塔
        )
    elif args.model_name == 'GlaucNet':
        clsmodel = ClassifierWithSegmentation()
    elif args.model_name == 'ae_densenet121' :
        clsmodel = ModifiedDenseNet121(pretrained=True, num_classes=2)

        # 定义模型
    if args.seg_model == 'MedNeXt':
        segmodel = get_MedNeXt_model()
        segmodel = load_seg_weight(segmodel)
    else:
        segmodel = KnowSAM(in_channels=3, num_classes=args.seg_num_classes)
        segmodel = load_KnowSAM_weight(segmodel)
    # summary( clsmodel , input_size=(1,args.frame_height, args.frame_width))
    clsmodel = load_cls_weight(clsmodel, args.weight_path)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda:' + str(args.gpu))


    segmodel = segmodel.to(device)
    clsmodel = clsmodel.to(device)

    criterion = criterion.to(device)

    test_cls(segmodel, clsmodel, test_loader, criterion, args)



if __name__ == '__main__':
    # set seed for reproduce
    args = parse_option()
    seeding(args.seed)
    # for fold_num in [0]:
    #     args.fold_num = fold_num
    #     worker(args)
    worker(args)