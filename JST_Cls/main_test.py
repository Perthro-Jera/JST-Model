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
import models.resnet as resnet
import models.convnext as convnext
from models.ae_densenet121 import ModifiedDenseNet121
from models.seg_drae import FusionModel
from torch.utils.data import DataLoader
from models.MedNextV1 import get_MedNeXt_model
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


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='seg_drae', type=str,
                        choices=['resnet18', 'convnext_pico',  'vit_base', 'vmamba_tiny', 'GlaucNet','ae_densenet121'],
                        help='the model we used.')
    parser.add_argument('--test_file', default='xiangyang',
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


#加载分割模型训练权重
def load_seg_weight(model):
    directory_path = os.path.join('train', 'MedNeXt', 'random_initial', '2')
    weight_path = os.path.join('checkpoint',directory_path, 'MedNeXt_checkpoint_huafen337.pth')
    #weight_path = os.path.join(weight_path, 'segmodel_checkpoint.pth')
    # 加载模型权重
    print('==> loading teacher model')
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_dict = checkpoint['state_dict']
    state = model.load_state_dict(model_dict, strict=False)
    print('loading checkpoint from {}'.format(weight_path))
    print(state)
    return model

def load_cls_weight(model, weight_path):
    weight_path = 'checkpoint/jst'
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




def test_cls(segmodel, clsmodel, test_loader, criterion, config):
    segmodel.eval()
    clsmodel.eval()
    ave_loss = DataUpdater()
    device = next(segmodel.parameters()).device
    num_class = 5  # 你的分类数是 5


    all_labels = []
    all_predictions = []
    all_probs = []  # 用于存储五分类预测概率

    all_labels_binary = []  # 存储二分类标签
    all_predictions_binary = []  # 存储二分类预测结果
    all_probs_binary = []  # 存储二分类概率

    cm_5 = torch.zeros(num_class, num_class, dtype=torch.int64)  # 5分类混淆矩阵
    cm_2 = torch.zeros(2, 2, dtype=torch.int64)  # 2分类混淆矩阵


    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images, texture, labels, _, _ = batch
            images = images.to(device, non_blocking=True)
            texture = texture.to(device, non_blocking=True)
            labels = labels.clone().detach().long().to(device, non_blocking=True)


            # 分类模型预测
            if args.train_pattern in ['mask_train', 'train']:
                images = images[:, 0:1, :, :]
                pred_cls = clsmodel(images)
            elif args.train_pattern in ['drae_train']:
                seg_output, _, _, _, _ = segmodel(images)
                # print(f"Image shape: {images.shape}")  # 输出图像形状
                # 转为灰度图
                gray_image = images[:, 0:1, :, :]  # 取 B 通道
                # DRAE 编码器的纹理特征
                drae_features = drae_encoder(gray_image)  # [batch_size, latent_dim]

                # **特征融合**
                fused_features = fusion_model(drae_features, seg_output)
                print("Fused features mean:", fused_features.mean().item())
                pred_cls = clsmodel(fused_features)
            elif args.train_pattern == 'seg_drae':
                seg_output, _, _, _, _ = segmodel(images)
                # print(f"Image shape: {images.shape}")  # 输出图像形状
                # 转为灰度图
                gray_image = images[:, 0:1, :, :]  # 取 B 通道
                pred_mask = torch.argmax(seg_output, dim=1)  # [batch_size, height, width]
                pred_mask = pred_mask.unsqueeze(1).float()
                seg_image = pred_mask.repeat(1, 3, 1, 1)  # 扩展通道，变为 [12, 3, 512, 1024]
                # 将类别 1 和 5 的像素值设置为 0
                pred_mask[(pred_mask == 1) | (pred_mask == 5) ] = 0
                pred_mask[(pred_mask == 2) | (pred_mask == 3) | (pred_mask == 4) | (pred_mask == 6) | (pred_mask == 7)] = 1

                extracted_image = gray_image * pred_mask

                # pred_mask = pred_mask.repeat(1, 2, 1, 1) # 扩展通道，变为 [12, 3, 512, 1024]
                # pred_cls = clsmodel(gray_image, pred_mask)

                pred_cls, _, _ = clsmodel(extracted_image, seg_image)

            elif args.train_pattern in ['seg_train', 'texture_train', 'st_train']:
                pred_seg, _, _, _, _ = segmodel(images)
                pred_mask = torch.argmax(pred_seg, dim=1).unsqueeze(1).float()
                # pred_mask = F.interpolate(pred_mask, size=(256, 512), mode='bilinear', align_corners=True)
                # texture = F.interpolate(texture, size=(256, 512), mode='bilinear', align_corners=True)
                pred_mask = pred_mask.repeat(1, 3, 1, 1)  # 扩展通道，变为 [12, 3, 512, 1024]
                pred_cls = clsmodel(pred_mask)

            cls_losses = criterion(pred_cls, labels)
            loss = cls_losses.mean()
            ave_loss.update(loss.item())

            pred_cls_5d = torch.argmax(pred_cls, dim=1)  # 获取五分类预测结果
            probs_5d = F.softmax(pred_cls, dim=1)  # 获取五分类概率


            # 计算 5 分类混淆矩阵
            for t, p in zip(labels.view(-1), pred_cls_5d.view(-1)):
                cm_5[t.long(), p.long()] += 1

            # 计算 2 分类的标签与预测
            labels_binary = (labels >= 3).long()  # 3、4 类归为 1（阳性），0、1、2 归为 0（阴性）
            pred_cls_binary = (pred_cls_5d >= 3).long()
            probs_binary = probs_5d[:, 3:5].sum(dim=1)  # 3 和 4 归为阳性的概率

            # 计算 2 分类混淆矩阵
            for t, p in zip(labels_binary.view(-1), pred_cls_binary.view(-1)):
                cm_2[t.long(), p.long()] += 1

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(pred_cls_5d.cpu().numpy())
            all_probs.extend(probs_5d.cpu().numpy())

            all_labels_binary.extend(labels_binary.cpu().numpy())
            all_predictions_binary.extend(pred_cls_binary.cpu().numpy())
            all_probs_binary.extend(probs_binary.cpu().numpy())

            if idx % 10 == 0:
                print(f"Processing batch {idx} / {len(test_loader)}")

    # 计算 5 分类准确率和报告
    acc_5 = accuracy_score(all_labels, all_predictions)
    report_5 = classification_report(all_labels, all_predictions, target_names=["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"], digits=4)
    print("5-class Confusion Matrix:")
    print(cm_5.numpy())
    print("5-class Accuracy:", acc_5)
    # print("5-class Classification Report:")
    # print(report_5)

    # 计算 2 分类指标
    acc_2 = accuracy_score(all_labels_binary, all_predictions_binary)
    tn, fp, fn, tp = cm_2.numpy().ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 灵敏度
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性
    auc = roc_auc_score(all_labels_binary, all_probs_binary) if len(set(all_labels_binary)) > 1 else 0

    print("2-class Confusion Matrix:")
    print(cm_2.numpy())
    print(f"2-class Accuracy: {acc_2:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"AUC: {auc:.4f}")

    # 保存结果到 JSON
    results = {
        "5-class": {
            "accuracy": acc_5,
            "confusion_matrix": cm_5.numpy().tolist(),
            "classification_report": report_5
        },
        "2-class": {
            "accuracy": acc_2,
            "confusion_matrix": cm_2.numpy().tolist(),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc": auc
        }
    }
    results_file = os.path.join(config.result_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    return ave_loss.avg, results


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
        clsmodel = FusionModel()
    elif args.model_name == 'GlaucNet':
        clsmodel = ClassifierWithSegmentation()
    elif args.model_name == 'ae_densenet121' :
        clsmodel = ModifiedDenseNet121(pretrained=True, num_classes=5)


    segmodel = load_seg_weight(segmodel)
    clsmodel = load_cls_weight(clsmodel, args.weight_path)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda:' + str(args.gpu))
    if args.train_pattern == 'drae_train':
        fusion_model = FeatureFusion(drae_dim=256, output_channels=64).to(device)
        # 加载融合模型权重
        state_dict = torch.load("checkpoint/fusion_model.pth", map_location='cpu')
        missing_keys, unexpected_keys = fusion_model.load_state_dict(state_dict, strict=False)
    else:
        fusion_model = None


    segmodel = segmodel.to(device)
    clsmodel = clsmodel.to(device)
    if args.train_pattern == 'drae_train':
        drae = DRAE(input_channels=1, latent_dim=256).to(device)
        checkpoint = torch.load('checkpoint/drae/drae_checkpoint2.pth')
        drae.load_state_dict(checkpoint['model_state_dict'])
        drae.eval()
        # 提取编码器
        drae_encoder = drae.encoder
        drae_encoder.eval()
    else:
        drae_encoder = None
    criterion = criterion.to(device)

    results = test_cls(segmodel, clsmodel, test_loader, criterion, args)



if __name__ == '__main__':
    # set seed for reproduce
    args = parse_option()
    seeding(args.seed)
    # for fold_num in [0]:
    #     args.fold_num = fold_num
    #     worker(args)
    worker(args)