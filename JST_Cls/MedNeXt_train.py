import os
import torch
import torch.nn as nn
import argparse
import timm
import yaml
from dataset import TransUnetDataSet
from torch.utils.data import DataLoader
from models.MedNextV1 import get_MedNeXt_model
from util import save_checkpoint, seeding
from engine_train import train_mednext_one_epoch
from criterion import CrossEntropy, OhemCrossEntropy
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='MedNeXt', type=str, help='model name')
    parser.add_argument('--model_config', default='config/transUnet.yaml', type=str, help='model config files')
    parser.add_argument('--num_class', default=8, type=int, help="class num")

    # parameters for data
    parser.add_argument('--frame_height', type=int, default=256, help='the frame height we used during training.')
    parser.add_argument('--frame_width', type=int, default=512, help='the frame width we used during training.')
    parser.add_argument('--ignore_label', type=int, default=-1, help='ignoring the label of pixels')

    # folder num，以便用于内部数据集的交叉验证
    parser.add_argument('--fold_num', type=int, default=2, choices=[0, 1, 2, 3, 4],
                        help='the fold num we used for cross validation.')

    parser.add_argument('--train_pattern', type=str, default="train",
                        choices=["train", "distill_label", "distill_unlabel"])
    # model parameters initialization pattern
    # parser.add_argument('--pre_train', action='store_true', help="weight initialized by the weight pretrained from "
    #                                                              "imageNet")
    parser.add_argument('--pre_train', default=False, type=bool, help="weight initialized by the weight pretrained from "
                                                                 "imageNet")

    # training configuration
    parser.add_argument('--epochs', type=int, default=100, help="training epoch")
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
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help="weight_decay")
    parser.add_argument('--nesrerov', default=True, type=bool, help='whether use nesrerov for sgd')

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


def worker(args):
    # 初始化 TensorBoard 记录器
    log_dir = os.path.join(args.result_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)
    # 合并yaml文件配置参数
    # args = merge_config(args)
    # 定义训练所用的数据集
    print(f"fold_num: {args.fold_num}")
    train_set = TransUnetDataSet(args, pattern=args.train_pattern)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)


    model = get_MedNeXt_model(args)


    class_weight = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1])
    # 定义分割损失函数
    # criterion
    # loss
    if args.use_ohem:
        segmentation_criterion = OhemCrossEntropy(ignore_label=args.ignore_label,
                                     thres=args.ohemthres,
                                     min_kept=args.ohemkeep,
                                     weight=class_weight)
    else:
        segmentation_criterion = CrossEntropy(ignore_label=args.ignore_label, weight=class_weight)

    # 定义优化器
    params_dict = dict(model.named_parameters())
    params = [{'params': list(params_dict.values()), 'lr': args.lr}]
    # optimizer = torch.optim.SGD(params,
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay,
    #                             nesterov=args.nesrerov,
    #                             )
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    #将模型和损失函数部署到GPU
    device = torch.device('cuda:' + str(args.gpu))
    model = model.to(device)
    segmentation_criterion = segmentation_criterion.to(device)


    # 定义模型权重的存储路径
    param_pattern = 'ade20k_pretrain' if args.pre_train else 'random_initial'
    #model_name = args.model_config.split('/')[-1].split('.')[0]
    model_name = args.model_name
    dir_postfix = os.path.join(args.train_pattern, model_name, param_pattern, str(args.fold_num))
    checkpoint_dir = os.path.join('checkpoint', dir_postfix)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # using the cosine learning rate scheduler
    num_ite_per_epoch = len(train_set) // args.batch_size

    for epoch in range(args.epochs):
        seg_loss = train_mednext_one_epoch(model, train_loader, optimizer, segmentation_criterion,  epoch, args.epochs, num_ite_per_epoch, args.lr)
        # 将损失写入 TensorBoard
        writer.add_scalar('Loss/Segmentation_Loss', seg_loss, epoch)
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_dir=checkpoint_dir, filename=f'MedNeXt_checkpoint_256_337.pth', is_best=False)
    writer.close()

if __name__ == '__main__':
    # set seed for reproduce
    args = parse_option()
    seeding(args.seed)
    # for fold_num in [0, 1, 2, 3, 4]:
    #for fold_num in [0]:
    #args.fold_num = fold_num
    worker(args)
    # args = merge_config(args)
    # model = get_seg_model(args)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")

