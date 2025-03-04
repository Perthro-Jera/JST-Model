import torch
import os
import torch.nn as nn
import random
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import cv2
from models.seg_drae import FusionModel
from models.MedNextV1 import get_MedNeXt_model
import albumentations as A
import time
import math
from albumentations.pytorch.transforms import ToTensorV2


# === 1. 配置参数 ===
MODEL_NAME = 'DRAE'
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
MIN_LR = 5e-7
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 0.6
WEIGHT_DECAY_END = 0.4
LATENT_DIM = 256
HEIGHT = 512
WIDTH = 1024
OCT_DEFAULT_MEAN = (0.3124)
OCT_DEFAULT_STD = (0.2206)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 使用 GPU 1
MODEL_SAVE_PATH = 'checkpoint/drae/checkpoint0222.pth'  # 保存模型的路径


# === 2. 数据增强 ===
train_transforms = A.Compose([
    A.RandomResizedCrop(size=(HEIGHT, WIDTH),
                        scale=(0.5, 1.0), ratio=(1.75, 2.25),
                        interpolation=cv2.INTER_CUBIC),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=OCT_DEFAULT_MEAN,
                std=OCT_DEFAULT_STD),
    ToTensorV2()
])

# === 3. 读取数据 ===
class OCTDataset(Dataset):
    def __init__(self, xlsx_file, transform=None):
        self.data = pd.read_excel(xlsx_file)
        self.image_paths = self.data["image_path"].tolist()
        self.labels = self.data["label"].map({
            "宫颈炎": 0, "囊肿": 1, "外翻": 2,
            "高级别病变": 3, "宫颈癌": 4
        }).tolist()  # 将标签映射为阴性（0）和阳性（1）
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]  # 读取标签
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Image not found or unable to read: {img_path}")
            img = np.expand_dims(img, axis=-1)  # 转为 (H, W, 1)
            if self.transform:
                img = self.transform(image=img)["image"]
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_paths))

class BalancedBatchSampler:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.neg_indices = [i for i, label in enumerate(dataset.labels) if label <= 2]
        self.pos_indices = [i for i, label in enumerate(dataset.labels) if label > 2]

    def shuffle_indices(self, indices):
        np.random.shuffle(indices)
        return indices

    def __iter__(self):
        # 每次迭代时重新打乱索引
        self.neg_indices = self.shuffle_indices(self.neg_indices)
        self.pos_indices = self.shuffle_indices(self.pos_indices)

        neg_pointer = 0
        pos_pointer = 0

        while neg_pointer < len(self.neg_indices):
            neg_batch = self.neg_indices[neg_pointer:neg_pointer + self.batch_size // 2]
            neg_pointer += len(neg_batch)

            pos_batch = []
            while len(pos_batch) < self.batch_size // 2:
                remaining = self.batch_size // 2 - len(pos_batch)
                pos_sample = self.pos_indices[pos_pointer:pos_pointer + remaining]
                pos_batch.extend(pos_sample)
                pos_pointer += len(pos_sample)

                if pos_pointer >= len(self.pos_indices):
                    self.pos_indices = self.shuffle_indices(self.pos_indices)
                    pos_pointer = 0

            # 确保最终每个批次的大小都匹配
            if len(neg_batch) + len(pos_batch) == self.batch_size:
                yield neg_batch + pos_batch
            else:
                break  # 如果剩余样本不足一个完整批次，则终止迭代

    def __len__(self):
        return len(self.neg_indices) // (self.batch_size // 2)




train_dataset = OCTDataset(
    xlsx_file="dataset/internal_A.xlsx",
    transform=train_transforms
)

# 创建采样器
batch_sampler = BalancedBatchSampler(train_dataset, batch_size=BATCH_SIZE)

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_sampler=batch_sampler,  # 使用自定义采样器
    num_workers=4
)

# === 4. 定义损失函数 ===
mse_loss = nn.MSELoss()

class_weight = torch.FloatTensor([0.069, 0.286, 0.258, 0.25, 0.25])
#class_weight = torch.FloatTensor([0.069, 0.286, 0.258, 0.221, 0.166])
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
clsmodel = FusionModel().to(DEVICE)
segmodel = get_MedNeXt_model().to(DEVICE)
segmodel = load_seg_weight(segmodel).to(DEVICE)
# AdamW 优化器
optimizer = optim.AdamW(clsmodel.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

writer = SummaryWriter(log_dir='drae_runs/drae_training')  # 用于存储 TensorBoard 的数据

# === 6. 训练 DRAE ===
def train():
    # 使用余弦学习率
    num_ite_per_epoch = len(train_loader)
    lr_schedule_values = cosine_scheduler(LEARNING_RATE, MIN_LR, EPOCHS, num_ite_per_epoch, WARMUP_EPOCHS)
    # using the cosine weight decay scheduler
    wd_schedule_values = cosine_scheduler(WEIGHT_DECAY, WEIGHT_DECAY_END, EPOCHS, num_ite_per_epoch)

    for epoch in range(EPOCHS):
        clsmodel.train()
        segmodel.eval()
        epoch_start_time = time.time()
        total_recon_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_loss = 0.0

        for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
            batch_images, batch_labels = batch_images.to(DEVICE), batch_labels.to(DEVICE)
            # **避免额外梯度计算**
            with torch.no_grad():
                images = batch_images.repeat(1, 3, 1, 1)
                seg_output, _, _, _, _ = segmodel(images)
                pred_mask = torch.argmax(seg_output, dim=1)  # [batch_size, height, width]
                pred_mask = pred_mask.unsqueeze(1).float()
                seg_image = pred_mask.repeat(1, 3, 1, 1) # 扩展通道，变为 [12, 3, 512, 1024]
                # 将类别 1 和 5 的像素值设置为 0
                pred_mask[(pred_mask == 1) | (pred_mask == 5)] = 0
                pred_mask[(pred_mask == 2) | (pred_mask == 3) | (pred_mask == 4) | (pred_mask == 6) | (pred_mask == 7)] = 1

                extracted_image = batch_images * pred_mask



            output, z, recon_x = clsmodel(extracted_image, seg_image)

            #recon_loss = mse_loss(recon_x, extracted_image)
            mask = (pred_mask == 1).float()
            recon_loss = torch.sum(mask * (extracted_image - recon_x) ** 2) / torch.sum(mask)

            #disc_loss = bce_loss(disc_output, batch_labels.unsqueeze(1).float())
            cls_loss = classification_criterion(output,batch_labels)
            bin_labels = ((batch_labels == 3) | (batch_labels == 4)).float()  # 3, 4 为正类，其他为负类
            reg_loss = discriminative_loss(z, bin_labels)

            loss = cls_loss + 0.5 * recon_loss + 0.3 * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            total_loss += loss.item()

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

            print(f"Epoch [{epoch + 1}/{EPOCHS}], Batch [{batch_idx + 1}/{len(train_loader)}],LR: {lr:.6e}, Recon_loss: {recon_loss.item():.4f}, Cls_loss: {cls_loss.item():.4f}, Reg_loss: {reg_loss.item():.4f}")

            # 计算每个 epoch 的平均损失
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)
        avg_total_loss =total_loss / len(train_loader)

        # 将每个 epoch 的平均损失值写入 TensorBoard
        writer.add_scalar('Epoch_Loss/Recon_loss', avg_recon_loss, epoch + 1)
        writer.add_scalar('Epoch_Loss/Cls_loss', avg_cls_loss, epoch + 1)
        writer.add_scalar('Epoch_Loss/Reg_loss', avg_reg_loss, epoch + 1)
        writer.add_scalar('Epoch_Loss/Total_loss', avg_total_loss, epoch + 1)
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
    train()