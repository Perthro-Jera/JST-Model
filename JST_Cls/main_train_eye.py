import os
import torch
import torch.nn as nn
import argparse
import timm
import yaml
from dataset import TransformerDataSet,EyeOCTClsDataset
from dataset import TransUnetDataSet
from dataset import TransformerLBPDataSet
import models.resnet as resnet

import models.convnext as convnext
from models.semi_MedNeXt import KnowSAM
from torch.utils.data import DataLoader
from models.MedNextV1 import get_MedNeXt_model
from models.seg_drae import FusionModel
from util import save_checkpoint, cosine_scheduler, seeding
from engine_train import train_ST_cls_epoch, train_cls_epoch, train_rec_cls_epoch, train_seg_drae_epoch
from torch.optim import AdamW
from criterion import CrossEntropy, OhemCrossEntropy
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard
import torchvision.transforms as T
from torch.utils.data import Sampler
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='seg_drae', type=str,
                        choices=['seg_drae','resnet18', 'convnext_pico', 'vit_base', 'vmamba_tiny', 'unet'],
                        help='the model we used.')
    parser.add_argument('--dataset', default='eye', choices=['eye', 'cervix'],
                        help='选择数据集类型，用于自动设置类别数等参数')
    parser.add_argument('--model_config', default='config/transUnet.yaml', type=str, help='model config files')
    # 分类/分割类别数（默认值会被 dataset 自动覆盖）
    parser.add_argument('--num_class', default=3, type=int, help="分类类别数")
    parser.add_argument('--seg_num_classes', default=6, type=int, help="分割类别数")

    # parameters for data
    parser.add_argument('--crop_frame_height', type=int, default=600, help='the frame height we used.')
    parser.add_argument('--frame_height', type=int, default=256, help='the frame height we used during training.')
    parser.add_argument('--frame_width', type=int, default=512, help='the frame width we used during training.')
    parser.add_argument('--ignore_label', type=int, default=-1, help='ignoring the label of pixels')

    # folder num，以便用于内部数据集的交叉验证
    parser.add_argument('--fold_num', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='the fold num we used for cross validation.')

    parser.add_argument('--train_pattern', type=str, default="seg_drae",
                        choices=["seg_drae", "train", "seg_train", "texture_train", "st_train", ])
    # model parameters initialization pattern
    # parser.add_argument('--pre_train', action='store_true', help="weight initialized by the weight pretrained from "
    #                                                              "imageNet")
    parser.add_argument('--pre_train', default=True, type=bool, help="weight initialized by the weight pretrained from "
                                                                     "imageNet")
    parser.add_argument('--finetune', default=False, type=bool, help="pretrained model for fine-tuning")

    # training configuration
    parser.add_argument('--epochs', type=int, default=30, help="training epoch")
    parser.add_argument('--start_epoch', type=int, default=0, help="start epoch")
    parser.add_argument('--batch_size', type=int, default=16, help="training batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="data loader thread")

    parser.add_argument('--gpu', type=int, default=0, help='the gpu number will be used')
    parser.add_argument('--checkpoint', type=str, default='checkpoint',
                        help='the directory to save the model weights.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--result_dir', default='result', type=str,
                        help='the directory of the model testing performance')

    # loss
    parser.add_argument('--use_ohem', default=False, type=bool, help='whether use ohem for cross entropy')
    parser.add_argument('--ohemthres', default=0.8, type=float, help='threshold for ohem')
    parser.add_argument('--ohemkeep', type=int, default=125000, help='minimal numbers of ohem')

    # optimizer
    parser.add_argument('--optimizer', default='AdamW', type=str, help='optimizer')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--warmup_lr', type=float, default=5e-7, help='warmup learning rate (default: 5e-7)')
    parser.add_argument('--min_lr', type=float, default=5e-7,
                        help='lower lr bound for cyclic schedulers that hit 0 (5e-7)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--weight_decay', default=0.6, type=float, help="weight_decay")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
           weight decay. We use a cosine schedule for WD.""")
    # resume
    parser.add_argument('--resume', action='store_true', help="if need to resume from the latest checkpoint")
    parser.add_argument('--seg_model', default='KnowSAM', choices=['MedNeXt', 'KnowSAM'], help='选择分割模型')
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
    print('==> loading seg model')
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_dict = checkpoint['state_dict']
    state = model.load_state_dict(model_dict, strict=False)
    print('loading checkpoint from {}'.format(weight_path))
    print(state)
    return model


def load_KnowSAM_weight(model):
    # weight_path = os.path.join('checkpoint/train/KnowSAM/results_OCT_resize256_nomixup30_mednext_unet1000_3/fold_0/SGDL_best_model.pth')
    weight_path = os.path.join('checkpoint/train/KnowSAM/eye/oct_full/SGDL_best_model.pth')  # 眼科oct分割权重
    # 加载模型权重
    print('==> loading seg model')
    state = model.load_state_dict(torch.load(weight_path), strict=True)
    print('loading checkpoint from {}'.format(weight_path))
    print(state)
    return model


def worker(args):
    # model configuration
    model_name = args.model_name
    num_class = args.num_class

    # model parameters initialization pattern
    pre_train = args.pre_train

    # training configuration
    # Trainer settings
    epochs = args.epochs
    start_epoch = args.start_epoch
    batch_size = args.batch_size
    num_workers = args.num_workers
    gpu = args.gpu

    # optimizer configuration
    lr = args.lr
    weight_decay = args.weight_decay
    weight_decay_end = args.weight_decay_end
    min_lr = args.min_lr
    warmup_epochs = args.warmup_epochs

    # resume configuration
    resume = args.resume

    # 初始化 TensorBoard 记录器
    log_dir = os.path.join(args.result_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)
    # 合并yaml文件配置参数
    # args = merge_config(args)
    # 定义训练所用的数据集
    train_set = EyeOCTClsDataset(args, pattern=args.train_pattern)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)
    # 调用模型
    # segmodel = get_MedNeXt_model()
    if args.seg_model == 'MedNeXt':
        segmodel = get_MedNeXt_model()
        segmodel = load_seg_weight(segmodel)
    else:
        segmodel = KnowSAM(in_channels=3, num_classes=args.seg_num_classes)
        segmodel = load_KnowSAM_weight(segmodel)

    if args.model_name == 'resnet18':
        clsmodel = getattr(resnet, model_name)(pretrained=pre_train, num_classes=num_class)
    elif args.model_name in ['convnext_pico', 'convnext_little', 'convnext_lite']:
        clsmodel = getattr(convnext, model_name)(pretrained=pre_train, num_classes=num_class)
    elif args.model_name == 'unet':
        clsmodel = UNetClassifier(in_channels=3, num_classes=num_class)
    elif args.model_name in ['vmamba_tiny']:
        clsmodel = getattr(vmamba, model_name)(pretrained=pre_train, num_classes=num_class)
    elif args.model_name in ['vit_tiny']:
        clsmodel = timm.create_model(model_name='vit_tiny_patch16_224', pretrained=False, num_classes=num_class,
                                     img_size=(args.frame_height, args.frame_width))
    elif args.model_name in ['vit_base']:
        clsmodel = timm.create_model(model_name='vit_base_patch16_224', pretrained=False, num_classes=num_class,
                                     img_size=(args.frame_height, args.frame_width))
    elif args.model_name == 'seg_drae':
        clsmodel = FusionModel(
            input_channels=1,  # 原来的设置
            latent_dim=256,  # 原来的设置
            class_num=args.num_class,  # ⚡关键：根据数据集自动设置类别数
            texture_extractor = 'fpn'   #重构或者特征金字塔
        )

    # 加载分割模型训练权重：
    # segmodel = load_seg_weight(segmodel)
    # fusion_model = FeatureFusion(drae_dim=256, output_channels=64)

    # 定义优化器
    optimizer = AdamW(clsmodel.parameters(), lr=lr, weight_decay=weight_decay)
    # fusion_optimizer = AdamW(fusion_model.parameters(), lr=lr, weight_decay=weight_decay)

    # 定义分类损失函数
    # criterion
    # loss

    # class_weight = torch.FloatTensor(
    #     [0.069, 0.286, 0.258, 0.25, 0.25])  # '宫颈炎': 0, '囊肿': 1, '外翻': 2, '高级别病变': 3, '宫颈癌': 4
    #class_weight = torch.FloatTensor([ 0.98, 0.62, 2.77 ]) # [NORMAL, AMD, DME ]
    class_weight = torch.FloatTensor([0.98, 2.77])  # [NORMAL, DME ]
    classification_criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.1)
    recon_criterion = nn.MSELoss()  # 你可以换成 nn.L1Loss()
    # 将模型和损失函数部署到GPU
    device = torch.device('cuda:' + str(args.gpu))
    # fusion_model = fusion_model.to(device)
    segmodel = segmodel.to(device)
    clsmodel = clsmodel.to(device)


    classification_criterion = classification_criterion.to(device)

    # 定义模型权重的存储路径
    param_pattern = 'pretrain' if args.pre_train else 'random_initial'
    dir_postfix = os.path.join(args.train_pattern, model_name, param_pattern, str(args.fold_num))
    checkpoint_dir = os.path.join('checkpoint', dir_postfix)
    checkpoint_dir = 'checkpoint/baseline'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 使用余弦学习率
    num_ite_per_epoch = len(train_set) // args.batch_size
    lr_schedule_values = cosine_scheduler(lr, min_lr, epochs, num_ite_per_epoch, warmup_epochs)
    # using the cosine weight decay scheduler
    if weight_decay_end is None:
        weight_decay_end = weight_decay
    wd_schedule_values = cosine_scheduler(weight_decay, weight_decay_end, epochs, num_ite_per_epoch)

    # 设置分割模型为评估模式，确保其参数在分类阶段不会更新
    segmodel.eval()

    # 训练分类模型
    for epoch in range(args.epochs):
        if args.train_pattern == 'train':
            cls_loss = train_cls_epoch(clsmodel, train_loader, optimizer, classification_criterion, epoch, args.epochs,
                                       num_ite_per_epoch, lr_schedule_values, wd_schedule_values, args.lr)

        elif args.train_pattern in ['seg_train', 'texture_train', 'st_train']:
            cls_loss = train_ST_cls_epoch(segmodel, clsmodel, train_loader, optimizer, classification_criterion, epoch,
                                          args.epochs, num_ite_per_epoch, lr_schedule_values,
                                          wd_schedule_values, args.lr)
        elif args.train_pattern == 'seg_drae':
            cls_loss, recon_loss, total_loss = train_seg_drae_epoch(
                segmodel, clsmodel, train_loader, optimizer, classification_criterion, recon_criterion,
                epoch, args.epochs, num_ite_per_epoch, lr_schedule_values,
                wd_schedule_values)

        # 将损失写入 TensorBoard
        writer.add_scalar('Loss/Classification_Loss', cls_loss, epoch)
        # writer.add_scalar('Loss/Reconstruction_loss', recon_loss, epoch)
        # writer.add_scalar('Loss/Total_Loss', total_loss, epoch)

        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': clsmodel.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_dir=checkpoint_dir, filename=f'checkpoint_eye_oct5k_2_30_01_fpn_cross.pth', is_best=False)

    writer.close()


if __name__ == '__main__':
    # set seed for reproduce
    args = parse_option()
    seeding(args.seed)
    worker(args)
