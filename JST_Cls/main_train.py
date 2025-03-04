import os
import torch
import torch.nn as nn
import argparse
import timm
import yaml
from dataset import TransformerDataSet
from dataset import TransUnetDataSet
from dataset import TransformerLBPDataSet
import models.resnet as resnet
import models.resnet1 as resnet1
import models.convnext as convnext
from models.Unet import UNetClassifier
#import models.vision_mamba as vim
#import models.vmamba as vmamba
import models.swin_transformer as swin_transformer
import models.cswin as cswin
from torch.utils.data import DataLoader
from models.MedNextV1 import get_MedNeXt_model
from models.drae import DRAE
from models.seg_drae import FusionModel
from models.fusion import FeatureFusion
from models.visiontransformer import get_vit_cls_model, DualViTClassifier, get_cls_token_model
from timm.models.vision_transformer import VisionTransformer
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
                        choices=['resnet18', 'convnext_pico',  'vit_base', 'vmamba_tiny', 'unet'],
                        help='the model we used.')
    parser.add_argument('--model_config', default='config/transUnet.yaml', type=str, help='model config files')
    parser.add_argument('--num_class', default=5, type=int, help="class num")

    # parameters for data
    parser.add_argument('--crop_frame_height', type=int, default=600, help='the frame height we used.')
    parser.add_argument('--frame_height', type=int, default=512, help='the frame height we used during training.')
    parser.add_argument('--frame_width', type=int, default=1024, help='the frame width we used during training.')
    parser.add_argument('--ignore_label', type=int, default=-1, help='ignoring the label of pixels')

    # folder num，以便用于内部数据集的交叉验证
    parser.add_argument('--fold_num', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='the fold num we used for cross validation.')

    parser.add_argument('--train_pattern', type=str, default="seg_drae",
                        choices=["train", "seg_train", "texture_train", "st_train", ])
    # model parameters initialization pattern
    # parser.add_argument('--pre_train', action='store_true', help="weight initialized by the weight pretrained from "
    #                                                              "imageNet")
    parser.add_argument('--pre_train', default=False, type=bool, help="weight initialized by the weight pretrained from "
                                                                 "imageNet")
    parser.add_argument('--finetune', default=False, type=bool,help="pretrained model for fine-tuning")

    # training configuration
    parser.add_argument('--epochs', type=int, default=60, help="training epoch")
    parser.add_argument('--start_epoch', type=int, default=0, help="start epoch")
    parser.add_argument('--batch_size', type=int, default=16, help="training batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="data loader thread")

    parser.add_argument('--gpu', type=int, default=1, help='the gpu number will be used')
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
    parser.add_argument('--min_lr', type=float, default=5e-7, help='lower lr bound for cyclic schedulers that hit 0 (5e-7)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--weight_decay', default=0.6, type=float, help="weight_decay")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
           weight decay. We use a cosine schedule for WD.""")
    # resume
    parser.add_argument('--resume', action='store_true', help="if need to resume from the latest checkpoint")
    args = parser.parse_args()
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
    print('==> loading seg model')
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_dict = checkpoint['state_dict']
    state = model.load_state_dict(model_dict, strict=False)
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
    #args = merge_config(args)
    # 定义训练所用的数据集
    train_set = TransformerDataSet(args, pattern=args.train_pattern)
    train_loader = DataLoader(dataset=train_set,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          pin_memory=True,
                          shuffle=True,  
                          drop_last=True)
    #调用模型
    segmodel = get_MedNeXt_model()
    if args.model_name == 'resnet18':
        clsmodel = getattr(resnet, model_name)(pretrained=pre_train, num_classes=num_class)
    elif args.model_name in ['convnext_pico', 'convnext_little', 'convnext_lite']:
        clsmodel = getattr(convnext, model_name)(pretrained=pre_train, num_classes=num_class)
    elif args.model_name == 'unet':
        clsmodel = UNetClassifier(in_channels=3, num_classes=num_class)
    elif args.model_name in ['vmamba_tiny']:
        clsmodel = getattr(vmamba, model_name)(pretrained=pre_train, num_classes=num_class)
    elif args.model_name in ['vit_tiny']:
        clsmodel = timm.create_model(model_name='vit_tiny_patch16_224', pretrained=False, num_classes=num_class, img_size=(args.frame_height,args.frame_width))
    elif args.model_name in ['vit_base']:
        clsmodel = timm.create_model(model_name='vit_base_patch16_224', pretrained=False, num_classes=num_class,
                                  img_size=(args.frame_height, args.frame_width))
    elif args.model_name in ['vim_tiny']:
        clsmodel = getattr(vim, model_name)(pretrained=pre_train, num_classes=num_class)
    elif args.model_name =='seg_drae':
        clsmodel = FusionModel()

    
    #加载分割模型训练权重：
    segmodel = load_seg_weight(segmodel)
    #fusion_model = FeatureFusion(drae_dim=256, output_channels=64)


    # 定义优化器
    optimizer = AdamW(clsmodel.parameters(), lr=lr, weight_decay=weight_decay)
    #fusion_optimizer = AdamW(fusion_model.parameters(), lr=lr, weight_decay=weight_decay)

    # 定义分类损失函数
    # criterion
    # loss
    #500张数据权重
    class_weight = torch.FloatTensor([0.069, 0.286, 0.258, 0.25, 0.25]) #'宫颈炎': 0, '囊肿': 1, '外翻': 2, '高级别病变': 3, '宫颈癌': 4
    #class_weight = torch.FloatTensor([0.069, 0.286, 0.258, 0.221, 0.166])
    #400张
    #class_weight = torch.FloatTensor([1.79, 7.41, 6.67, 10.00, 17.39])
    #3372张数据权重
    #class_weight = torch.FloatTensor([2.40, 16.13, 17.24, 7.94, 7.94])
    classification_criterion = nn.CrossEntropyLoss(weight=class_weight,label_smoothing=0.1)
    recon_criterion = nn.MSELoss()  # 你可以换成 nn.L1Loss()
    # 将模型和损失函数部署到GPU
    device = torch.device('cuda:' + str(args.gpu))
    #fusion_model = fusion_model.to(device)
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

    classification_criterion = classification_criterion.to(device)

    # 定义模型权重的存储路径
    param_pattern = 'pretrain' if args.pre_train else 'random_initial'
    dir_postfix = os.path.join(args.train_pattern, model_name, param_pattern, str(args.fold_num))
    checkpoint_dir = os.path.join('checkpoint', dir_postfix)
    checkpoint_dir = 'checkpoint/drae/stru_checkpoint1.pth'
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


    #训练分类模型
    for epoch in range(args.epochs):
        if args.train_pattern == 'train':
            cls_loss = train_cls_epoch(clsmodel, train_loader,  optimizer,  classification_criterion, epoch, args.epochs, num_ite_per_epoch, lr_schedule_values, wd_schedule_values, args.lr)

        # elif args.train_pattern == 'drae_train':
        #     cls_loss = train_rec_cls_epoch(
        #         segmodel, clsmodel, drae_encoder, fusion_model, train_loader, optimizer, fusion_optimizer, classification_criterion,
        #         epoch, args.epochs, num_ite_per_epoch, lr_schedule_values,
        #         wd_schedule_values)

        elif args.train_pattern in ['seg_train', 'texture_train', 'st_train']:
            cls_loss = train_ST_cls_epoch(segmodel, clsmodel, train_loader,  optimizer,  classification_criterion, epoch, args.epochs, num_ite_per_epoch, lr_schedule_values,
                        wd_schedule_values, args.lr)
        elif args.train_pattern == 'seg_drae':
            cls_loss, recon_loss, total_loss = train_seg_drae_epoch(
                segmodel, clsmodel, train_loader, optimizer, classification_criterion, recon_criterion,
                epoch, args.epochs, num_ite_per_epoch, lr_schedule_values,
                wd_schedule_values)

        # 将损失写入 TensorBoard
        writer.add_scalar('Loss/Classification_Loss', cls_loss, epoch)
        #writer.add_scalar('Loss/Reconstruction_loss', recon_loss, epoch)
        #writer.add_scalar('Loss/Total_Loss', total_loss, epoch)

        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            # 保存融合模型权重
            # if args.train_pattern == 'drae_train':
            #     torch.save(fusion_model.state_dict(), "checkpoint/fusion_model.pth")
            #     print("Saving fusion model weights:", fusion_model.state_dict()['drae_to_spatial.0.weight'][0][0][:5])

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': clsmodel.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_dir=checkpoint_dir, is_best=False)

    writer.close()

if __name__ == '__main__':
    # set seed for reproduce
    args = parse_option()
    seeding(args.seed)
    # for fold_num in [0, 1, 2, 3, 4]:
    # for fold_num in [0]:
    #     args.fold_num = fold_num
    #     worker(args)
    worker(args)
