import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import cv2
from models.drae import DRAE
import models.resnet as resnet
#import models.resnet1 as resnet1
import models.convnext as convnext
from models.ae_densenet121 import ModifiedDenseNet121
#from models.Unet import UNetClassifier
#import models.vision_mamba as vim
#import models.vmamba as vmamba
import timm
#import models.swin_transformer as swin_transformer
#import models.cswin as cswin
from models.seg_drae import FusionModel
from models.MedNextV1 import get_MedNeXt_model
import albumentations as A
import time
import math
from albumentations.pytorch.transforms import ToTensorV2
import tifffile
import argparse



# === 1. 配置参数 ===
MODEL_NAME = 'ae_densenet121'
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MIN_LR = 5e-7
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 0.6
WEIGHT_DECAY_END = 0.4
LATENT_DIM = 256
HEIGHT = 256
WIDTH = 512
OCT_DEFAULT_MEAN = (0.3124)
OCT_DEFAULT_STD = (0.2206)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 使用 GPU 1
MODEL_SAVE_PATH = 'checkpoint/baseline/checkpoint_ae_densenet.pth'  # 保存模型的路径


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='glaucnet', type=str,
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

    parser.add_argument('--train_pattern', type=str, default="train",
                        choices=["seg_drae", "train", "seg_train", "texture_train", "st_train", ])
    # model parameters initialization pattern
    # parser.add_argument('--pre_train', action='store_true', help="weight initialized by the weight pretrained from "
    #                                                              "imageNet")
    parser.add_argument('--pre_train', default=True, type=bool, help="weight initialized by the weight pretrained from "
                                                                     "imageNet")
    parser.add_argument('--finetune', default=False, type=bool, help="pretrained model for fine-tuning")

    # training configuration
    parser.add_argument('--epochs', type=int, default=60, help="training epoch")
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


def verify_clsfile_paths(image_paths):
    invalid_image_paths = [path for path in image_paths if not os.path.isfile(path)]
    if invalid_image_paths:
        print(f"以下图像文件路径无效: {invalid_image_paths}")

    return len(invalid_image_paths) == 0

class EyeOCTClsDataset(Dataset):
    def __init__(self, args, pattern='train'):
        self.args = args
        self.pattern = pattern

        self.imgs, self.labels = self.__load_file(
            pattern=pattern,
            fold_num=args.fold_num
        )

        if not verify_clsfile_paths(self.imgs):
            raise ValueError("检测到无效的 image 路径")

        print(f"[EyeOCT-CLS] pattern={pattern}, samples={len(self.imgs)}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        imgPath = self.imgs[index]
        label = self.labels[index]

        # ========== 1. 读取图像 ==========
        if imgPath.lower().endswith((".tif", ".tiff")):
            img = tifffile.imread(imgPath)
            # TIFF可能返回 (H, W, 1)，统一降到 (H, W)
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]
        else:
            img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
            # 如果彩色图像，转灰度
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 如果单通道是 (H, W, 1)，降维
            elif img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]

        # ========== 2. Resize ==========
        H, W = self.args.frame_height, self.args.frame_width
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        # ========== 3. 归一化 ==========
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0

        # ========== 4. OCT Normalize ==========
        img = (img - img.mean()) / (img.std() + 1e-6)

        # ========== 5. HWC → CHW ==========
        img = torch.from_numpy(img[np.newaxis, :, :]).float()
        label = torch.tensor(label, dtype=torch.long)

        return img, label, imgPath


    def __load_file(self, pattern="train", fold_num=0):


        # 你之前已经统一：sheet0=train, sheet1=test
        if pattern == "test":
            if self.args.test_file == 'OCT2017':
                file_path = 'dataset/OCT2017_cls.xlsx'
            elif self.args.test_file == 'internal':
                file_path = 'dataset/oct5k_classication.xlsx'
            else:
                file_path = 'dataset/RetinalOCT_cls.xlsx'
            df = pd.read_excel(file_path, sheet_name=1)
        else:
            file_path = 'dataset/oct5k_classication.xlsx'
            df = pd.read_excel(file_path, sheet_name=0)

        # 眼科 label 映射（示例）
        label_map = {
            'Normal': 0,
            #'AMD': 1,
            'NORMAL': 0,
            'DME': 1
        }

        df = df[df['label'].isin(label_map.keys())]
        df['label'] = df['label'].map(label_map)
        df.reset_index(drop=True, inplace=True)

        image_paths = df['image_path'].tolist()
        labels = df['label'].tolist()

        return image_paths, labels
class BalancedBatchSampler:
    """
    Balanced sampler for Eye OCT binary classification
    label: 0 = NORMAL, 1 = DME
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        assert batch_size % 2 == 0, "batch_size must be even for balanced sampling"

        # 二分类：0 / 1
        self.neg_indices = [i for i, l in enumerate(dataset.labels) if l == 0]
        self.pos_indices = [i for i, l in enumerate(dataset.labels) if l == 1]

        if len(self.neg_indices) == 0 or len(self.pos_indices) == 0:
            raise ValueError("BalancedBatchSampler: empty class detected")

    def __iter__(self):
        np.random.shuffle(self.neg_indices)
        np.random.shuffle(self.pos_indices)

        neg_ptr, pos_ptr = 0, 0
        half = self.batch_size // 2

        while neg_ptr + half <= len(self.neg_indices):
            neg_batch = self.neg_indices[neg_ptr:neg_ptr + half]
            neg_ptr += half

            pos_batch = []
            while len(pos_batch) < half:
                remain = half - len(pos_batch)
                pos_batch.extend(self.pos_indices[pos_ptr:pos_ptr + remain])
                pos_ptr += remain

                if pos_ptr >= len(self.pos_indices):
                    np.random.shuffle(self.pos_indices)
                    pos_ptr = 0

            yield neg_batch + pos_batch

    def __len__(self):
        return len(self.neg_indices) // (self.batch_size // 2)




# === 4. 定义损失函数 ===
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()


def dice_loss(pred, target, num_classes):
    eps = 1e-6  # 防止除0错误
    dice = 0.0

    # 对每个类别计算Dice损失
    for c in range(num_classes):
        # 提取当前类别的预测和真实标签
        pred_c = pred[:, c, :, :]  # 预测的概率图，形状为 [batch_size, num_classes, height, width]
        target_c = target[:, c, :, :]  # 真实标签，形状为 [batch_size, num_classes, height, width]

        # 二值化预测结果（默认阈值为0.5）
        pred_c = pred_c > 0.5

        # 计算Dice系数
        intersection = torch.sum(pred_c * target_c)
        union = torch.sum(pred_c) + torch.sum(target_c)

        dice_c = (2. * intersection + eps) / (union + eps)

        dice += 1 - dice_c  # Dice损失是Dice系数的补数

    return dice / num_classes  # 对所有类别求平均


class_weight = torch.FloatTensor([0.98, 2.77])  # [NORMAL, DME ]

classification_criterion = nn.CrossEntropyLoss(weight=class_weight,label_smoothing=0.1).to(DEVICE)

def discriminative_loss(z, labels):
    """Fisher 判别损失"""
    z_neg = z[labels == 0]
    z_pos = z[labels == 1]

    mu_neg, sigma_neg = torch.mean(z_neg, dim=0), torch.var(z_neg, dim=0)
    mu_pos, sigma_pos = torch.mean(z_pos, dim=0), torch.var(z_pos, dim=0)

    inter_class_variance = torch.norm(mu_neg - mu_pos) ** 2
    intra_class_variance = sigma_neg.mean() + sigma_pos.mean()

    return intra_class_variance / (inter_class_variance + 1e-6)

def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    print("Set warmup steps = %d" % warmup_iters)

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

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


# === 5. 初始化模型 ===
num_class = 5
pre_train = True
if MODEL_NAME == 'resnet18':
    clsmodel = getattr(resnet, MODEL_NAME)(pretrained=pre_train, num_classes=num_class)
elif MODEL_NAME in ['convnext_pico', 'convnext_little', 'convnext_lite']:
    clsmodel = getattr(convnext, MODEL_NAME)(pretrained=pre_train, num_classes=num_class)
elif MODEL_NAME == 'unet':
    clsmodel = UNetClassifier(in_channels=3, num_classes=num_class)
elif MODEL_NAME in ['vmamba_tiny']:
    clsmodel = getattr(vmamba, MODEL_NAME)(pretrained=pre_train, num_classes=num_class)
elif MODEL_NAME in ['vit_tiny']:
    clsmodel = timm.create_model(model_name='vit_tiny_patch16_224', pretrained=False, num_classes=num_class,
                                 img_size=(HEIGHT, WIDTH))
elif MODEL_NAME in ['vit_base']:
    clsmodel = timm.create_model(model_name='vit_base_patch16_224', pretrained=False, num_classes=num_class,
                                 img_size=(HEIGHT, WIDTH))
elif MODEL_NAME in ['vim_tiny']:
    clsmodel = getattr(vim, MODEL_NAME)(pretrained=pre_train, num_classes=num_class)
elif MODEL_NAME == 'efficientnet_v2' :
    clsmodel = timm.create_model('efficientnetv2_rw_s', pretrained=False, num_classes=num_class)
elif MODEL_NAME == 'densenet121':
    clsmodel = timm.create_model('densenet121', pretrained=False, num_classes=num_class)
elif MODEL_NAME == 'swin_base':
    clsmodel = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_class, img_size=(HEIGHT, WIDTH))
elif MODEL_NAME == 'ae_densenet121':
    clsmodel = ModifiedDenseNet121(pretrained=True, num_classes=2)

clsmodel = clsmodel.to(DEVICE)
# AdamW 优化器
optimizer = optim.AdamW(clsmodel.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

writer = SummaryWriter(log_dir='drae_runs/drae_training')  # 用于存储 TensorBoard 的数据

# === 6. 训练 DRAE ===
def train(args):
    train_set = EyeOCTClsDataset(args, pattern=args.train_pattern)
    sampler = BalancedBatchSampler(train_set, batch_size=args.batch_size)

    train_loader = DataLoader(
        dataset=train_set,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # 使用余弦学习率
    num_ite_per_epoch = len(train_loader)
    lr_schedule_values = cosine_scheduler(LEARNING_RATE, MIN_LR, EPOCHS, num_ite_per_epoch, WARMUP_EPOCHS)
    # using the cosine weight decay scheduler
    wd_schedule_values = cosine_scheduler(WEIGHT_DECAY, WEIGHT_DECAY_END, EPOCHS, num_ite_per_epoch)

    for epoch in range(EPOCHS):
        clsmodel.train()
        epoch_start_time = time.time()

        total_cls_loss = 0.0


        for batch_idx, (batch_images, batch_labels,image_paths) in enumerate(train_loader):
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            batch_start_time = time.time()  # 记录开始时间
            batch_images = batch_images
            output = clsmodel(batch_images)


            cls_loss = classification_criterion(output,batch_labels)

            loss = cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time = time.time() - batch_start_time
            print(f"Batch Time: {batch_time:.2f} seconds.")

            total_cls_loss += cls_loss.item()



            # 更新学习率和权重衰减
            global_step = epoch * num_ite_per_epoch + batch_idx

            # 更新学习率和权重衰减
            global_step = epoch * num_ite_per_epoch + batch_idx
            lr = lr_schedule_values[min(global_step, len(lr_schedule_values) - 1)]
            wd = wd_schedule_values[min(global_step, len(wd_schedule_values) - 1)]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
                if param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd

            print(f"Epoch [{epoch + 1}/{EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}],LR: {lr:.6e}, Cls_loss: {cls_loss.item():.4f}")

            # 计算每个 epoch 的平均损失

        avg_cls_loss = total_cls_loss / len(train_loader)


        # 将每个 epoch 的平均损失值写入 TensorBoard

        writer.add_scalar('Epoch_Loss/Cls_loss', avg_cls_loss, epoch + 1)


        epoch_time = time.time() - epoch_start_time

        print(f"Epoch [{epoch + 1}/{EPOCHS}] completed in {epoch_time:.2f} seconds.")
        #print(f"Total Recon_loss: {total_recon_loss:.4f}, Total Disc_loss: {total_disc_loss:.4f}, Total Reg_loss: {total_reg_loss:.4f}")


        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': clsmodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, MODEL_SAVE_PATH)
            print(f"Checkpoint saved at epoch {epoch + 1}.")

        # 训练结束时关闭 TensorBoard writer
        writer.close()

if __name__ == "__main__":
    args = parse_option()
    train(args)