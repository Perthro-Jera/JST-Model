import cv2
import torch
import numpy as np
import math
import os
import sys
import shutil
import random
import albumentations as A
from skimage import feature
import torch.nn.functional as F
from constants import OCT_DEFAULT_MEAN, OCT_DEFAULT_STD
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from PIL import Image

np.random.seed(42)


# Class DataUpdater, used to update the statistic data such as loss, accuracy.
class DataUpdater(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# define the data augmentation for image train
class DAForTrain(object):
    def __init__(self, args):
        self.transform = A.Compose([
            A.RandomCrop(height=args.frame_height, width=args.frame_width, always_apply=True),
            A.HorizontalFlip(p=0.5),
        ])
        self.img_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.GaussianBlur(p=0.15),
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])

    def __call__(self, image, mask):
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']  # [H, W] 这个地方先对mask不做处理，就是int8的二维数组
        mask = torch.from_numpy(mask)
        image = self.img_transform(image=image)['image']  # [3, H, W] torch.Tensor
        return image, mask

class DAForMedNeXtTrain(object):
    def __init__(self, args):
        self.transform = A.Compose([
            A.RandomCrop(height=args.frame_height, width=args.frame_width, always_apply=True),
            A.HorizontalFlip(p=0.5),
        ])
        self.img_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.GaussianBlur(p=0.15),
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])

    def __call__(self, image, mask):
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']  # [H, W] 这个地方先对mask不做处理，就是int8的二维数组
        mask = torch.from_numpy(mask)
        image = self.img_transform(image=image)['image']  # [3, H, W] torch.Tensor
        return image, mask
# define the data augmentation for valid and test
class DAForTest(object):
    def __init__(self, args):
        self.img_transform = A.Compose([
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])

    def __call__(self, image, mask):
        image = self.img_transform(image=image)['image']  # [3, H, W] torch.Tensor
        # [H, W] 这个地方先对mask不做处理，就是np.int8的二维数组
        mask = torch.from_numpy(mask)
        return image, mask


# define the data augmentation for predict
class DAForPredict(object):
    def __init__(self, args):
        self.img_transform = A.Compose([
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])

    def __call__(self, image, mask=None):
        image = self.img_transform(image=image)['image']  # [3, H, W] torch.Tensor
        return image, mask


# define the data augmentation for image train
class DAForUnlabel(object):
    def __init__(self, args):
        self.transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomResizedCrop(height=256, width=512,
                                scale=(0.4, 1.0), ratio=(1.8, 2.2),
                                interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])

    def __call__(self, image, mask=None):
        image = self.transform(image=image)['image']
        return image, mask  # shape is (3, H, W)


# ====== 随机遮挡数据增强 ======
def random_mask(image, mask_size=(50, 50), num_masks=3):
    h, w, c = image.shape
    masked_image = image.copy()

    for _ in range(num_masks):
        x = random.randint(0, w - mask_size[1])
        y = random.randint(0, h - mask_size[0])
        masked_image[y:y + mask_size[0], x:x + mask_size[1]] = 0  # 黑色填充

    return masked_image
class RandomMask(object):
    def __init__(self, args, mask_ratio=0.3, mask_size_range=(10, 30)):
        """
        :param args: 训练参数
        :param mask_ratio: 遮挡比例（0~1），例如 0.1 代表遮挡 10% 的像素
        :param mask_size_range: 遮挡块大小范围 (最小尺寸, 最大尺寸)
        """
        self.mask_ratio = mask_ratio
        self.mask_size_range = mask_size_range  # 遮挡块的大小范围

        # 主要数据增强（空间变换）
        self.transform = A.Compose([
            A.RandomResizedCrop(height=args.frame_height, width=args.frame_width, always_apply=True),
            A.HorizontalFlip(p=0.5),
        ], additional_targets={'texture': 'image'})  # 纹理图像也做相同变换

        # 影像数据增强
        self.img_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.GaussianBlur(p=0.15),
            A.Normalize(mean=OCT_DEFAULT_MEAN, std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])

        # 纹理图转换
        self.texture_transform = ToTensorV2()

    def apply_random_mask(self, image):
        """对 image 进行随机遮挡"""
        H, W, _ = image.shape
        total_pixels = H * W  # 总像素数
        mask_pixels = int(total_pixels * self.mask_ratio)  # 需要遮挡的像素数

        masked_image = image.copy()
        covered_pixels = 0  # 记录已遮挡像素数

        while covered_pixels < mask_pixels:
            # 随机生成遮挡块大小
            mask_h = random.randint(*self.mask_size_range)
            mask_w = random.randint(*self.mask_size_range)

            # 随机选择一个位置
            top = random.randint(0, H - mask_h)
            left = random.randint(0, W - mask_w)

            # 用均值填充（打码）
            masked_image[top:top+mask_h, left:left+mask_w, :] = np.mean(masked_image, axis=(0, 1))

            # 更新已遮挡的像素数
            covered_pixels += mask_h * mask_w

        return masked_image

    def __call__(self, image, texture):
        """对输入的 image 和 texture 进行增强"""
        # 应用基础空间变换（裁剪、翻转）
        augmented = self.transform(image=image, texture=texture)
        image = augmented['image']
        texture = augmented['texture']  # [H, W] numpy

        # 在裁剪后的 image 上添加随机遮挡
        image = np.array(image)
        image = self.apply_random_mask(image)  # 遮挡部分像素

        # 进行图像增强（亮度、模糊、归一化等）
        image = self.img_transform(image=image)['image']  # [3, H, W] torch.Tensor
        texture = self.texture_transform(image=np.array(texture))['image'].float()  # [3, H, W] torch.Tensor

        return image, texture

#定义cls的训练数据集
class DAForClsTrain(object):
    def __init__(self, args):
        self.transform = A.Compose([
            A.RandomResizedCrop(height=args.frame_height, width=args.frame_width, always_apply=True),
            A.HorizontalFlip(p=0.5),
        ],additional_targets={'texture': 'image'})
        self.img_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.GaussianBlur(p=0.15),
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])
        self.texture_transform = ToTensorV2()

    def __call__(self, image, texture):
        augmented = self.transform(image=image, texture=texture)
        image = augmented['image']
        texture = augmented['texture']  # [H, W]
        texture = np.array(texture)
        image = self.img_transform(image=image)['image']  # [3, H, W] torch.Tensor
        texture = self.texture_transform(image=texture)['image'].float()  # [3, H, W] torch.Tensor
        return image, texture

class DAForVitTrain1(object):
    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(args.frame_height, args.frame_width), scale=(0.8, 1.0), ratio=(1.8, 2.2), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            transforms.ToTensor(),  # 转换为Tensor并归一化
            transforms.Normalize(mean=OCT_DEFAULT_MEAN, std=OCT_DEFAULT_STD)  # 假设你已定义好OCT_DEFAULT_MEAN和STD
        ])
        # 掩码不做处理，因为你没有进行分割任务，mask不需要像图像一样做归一化
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为Tensor
        ])

    def __call__(self, image, mask):
        # 对图像和掩码同时进行增强
        augmented_image = self.transform(image)  # 图像增强
        augmented_mask = self.mask_transform(mask)  # 掩码只转为Tensor

        return augmented_image, augmented_mask

#定义Vit测试的数据
class DAForClsTest(object):
    def __init__(self, args):
        self.transform = A.Compose([
            A.Resize(height=args.frame_height, width=args.frame_width, always_apply=True),
        ],additional_targets={'texture': 'image'})
        self.img_transform = A.Compose([
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])
        self.texture_transform = ToTensorV2()

    def __call__(self, image, texture):
        augmented = self.transform(image=image, texture=texture)
        image = augmented['image']
        texture = augmented['texture']
        texture = np.array(texture)
        image = self.img_transform(image=image)['image']  # [3, H, W] torch.Tensor
        texture = self.texture_transform(image=texture)['image'].float()
        return image, texture


class DAForTransUnetTest(object):
    def __init__(self, args):
        self.transform = A.Compose([
            A.Resize(height=args.frame_height, width=args.frame_width, always_apply=True),
        ])
        self.img_transform = A.Compose([
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])

    def __call__(self, image, mask):
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']  # [H, W] 这个地方先对mask不做处理，就是int8的二维数组
        mask = torch.from_numpy(mask)
        image = self.img_transform(image=image)['image']  # [3, H, W] torch.Tensor
        return image, mask

#定义seg_to_transformer数据集
class DAForSegVit(object):
    def __init__(self, args):
        self.transform = A.Compose([
            A.Resize(height=args.frame_height, width=args.frame_width, always_apply=True),
        ],additional_targets={'texture': 'image'})
        self.img_transform = A.Compose([
            A.Normalize(mean=OCT_DEFAULT_MEAN,
                        std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])
        self.texture_transform = ToTensorV2()

    def __call__(self, image, texture):
        texture = np.array(texture)
        augmented = self.transform(image=image, texture=texture)
        image = augmented['image']
        texture = augmented['texture']  #[H,W,3]

        image = self.img_transform(image=image)['image']  # [3, H, W] torch.Tensor
        texture = self.texture_transform(image=texture)['image'].float()  # [1, H, W] torch.Tensor

        return image, texture

#LBP
class LBPDAForSegVit:
    def __init__(self, args):
        self.transform = A.Compose([
            A.Resize(height=args.frame_height, width=args.frame_width, always_apply=True),
        ], additional_targets={'texture': 'image'})

        self.img_transform = A.Compose([
            A.Normalize(mean=OCT_DEFAULT_MEAN, std=OCT_DEFAULT_STD),
            ToTensorV2()
        ])

    def _normalize_lbp_to_255(self, texture):
        """将 LBP 图像归一化到 [0, 255]"""
        texture_min = texture.min()
        texture_max = texture.max()
        if texture_max > texture_min:
            texture = (texture - texture_min) / (texture_max - texture_min) * 255
        else:
            texture = np.zeros_like(texture)
        return texture.astype(np.uint8)

    def _normalize_lbp(self, texture):
        """将 LBP 图像归一化到 [0, 1] 范围"""
        texture_min = texture.min()
        texture_max = texture.max()
        if texture_max > texture_min:
            texture = (texture - texture_min) / (texture_max - texture_min)
        else:
            texture = np.zeros_like(texture)  # 避免除零情况
        return texture

    def ensure_gray(self, image):
        """确保图像为单通道灰度图"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image[:, :, 0]
        return image

    def _lbp_transform(self, image):
        """计算 LBP 特征并归一化"""
        image = self.ensure_gray(image)  # 确保单通道
        texture = feature.local_binary_pattern(image, P=24, R=3, method='ror').astype(float)
        texture = self._normalize_lbp_to_255(texture)
        return texture

    def __call__(self, image, texture=None):
        """处理输入图像和 LBP 特征"""
        texture = self._lbp_transform(image)

        augmented = self.transform(image=image, texture=texture)
        image = augmented['image']
        texture = augmented['texture']  # [H, W]

        # print(f"LBP 图像形状: {texture.shape}")
        # print(f"LBP 图像像素值范围: [{texture.min()}, {texture.max()}]")

        image = self.img_transform(image=image)['image']  # [3, H, W] torch.Tensor
        texture = torch.tensor(texture, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return image, texture
class ImageProcessor(object):
    def __init__(self, args, augmentation='train'):
        self.augmentation = augmentation
        self.args = args

    def __call__(self, img, mask):

        if self.augmentation in ["train", "distill_label"]:
            #hrnet的训练数据集预处理
            if self.args.model_name == 'hrnet':
                data_augmentation = DAForTrain(self.args)
            #Vit的训练数据集预处理
            elif self.args.model_name in ["resnet18", "convnext_pico", 'vit_base', 'vmamba_tiny', 'unet']:
                data_augmentation = DAForClsTrain(self.args)
                #data_augmentation = RandomMask(self.args)
            #transUnet的训练数据集预处理
            elif self.args.model_name == 'transUnet':
                data_augmentation = DAForTrain(self.args)
            #seg_to_transformer的训练数据集预处理
            elif self.args.model_name == 'seg_to_transformer':
                data_augmentation = DAForSegVit(self.args)
            #MedNeXt的训练数据集预处理
            elif self.args.model_name == 'MedNeXt':
                data_augmentation = DAForMedNeXtTrain(self.args)
        elif self.augmentation in ["drae_train", "seg_train", "texture_train", "st_train", "seg_drae"]:
            data_augmentation = DAForClsTrain(self.args)
            #data_augmentation = DAForSegVit(self.args)
            #data_augmentation = RandomMask(self.args)
        elif self.augmentation in ["test"]:
            #hrnet的测试数据集预处理
            if self.args.model_name == 'hrnet':
                data_augmentation = DAForTest(self.args)
            #测试数据集预处理
            elif self.args.model_name in ["resnet18", "convnext_pico", 'vit_base', 'vmamba_tiny', 'unet', 'seg_drae', 'efficientnet_v2', 'densenet121', 'swin_base', 'GlaucNet', 'ae_densenet121']:
                data_augmentation = DAForClsTest(self.args)
                #data_augmentation = DAForSegVit(self.args)
            #transUnet得测试数据集预处理
            elif self.args.model_name == 'transUnet':
                data_augmentation = DAForTransUnetTest(self.args)
            # seg_to_transformer的测试数据集预处理
            elif self.args.model_name == 'seg_to_transformer':
                data_augmentation = DAForSegVit(self.args)
                # MedNeXt的训练数据集预处理
            elif self.args.model_name == 'MedNeXt':
                data_augmentation = DAForTransUnetTest(self.args)
        elif self.augmentation in ["predict"]:
            data_augmentation = DAForSegVit(self.args)
        else:
            print('not support data augmentation type !')
            sys.exit(-1)
        img, mask = data_augmentation(img, mask)
        return img, mask


def seeding(seed):
    """
    Set the seed for randomness.
    :param seed: int; the seed for randomness.
    :return: None.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


def create_directory(directory):
    """
    Create the directory.
    :param directory: str; the directory path we want to create.
    :return: None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_checkpoint(state, checkpoint_dir, is_best=False, filename='checkpoint.pth'):
    """
    :param state:
    :param checkpoint_dir: 存储的检查点目录路径
    :param is_best: 自监督训练时is_best为False, 监督训练和微调时动态传入
    :param filename:
    :return:
    """
    filename = os.path.join(checkpoint_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth'))


if __name__ == '__main__':
    # color_transform = A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=1)
    # sol_transform = A.Solarize(p=1.0)
    # img = cv2.imread('2.png', cv2.COLOR_BGR2GRAY)
    # img = cv2.imread('2.png')
    # img = A.GaussianBlur(blur_limit=(9, 11))(image=img)['image']
    # print(img.shape)

    # col_img = color_transform(image=img)['image']
    # sol_img = sol_transform(image=img)['image']
    # print(img.shape)
    # cv2.imshow('img', img)
    # # cv2.imshow('col_img', col_img)
    # cv2.imshow('sol_img', sol_img)
    # cv2.waitKey(0)
    # print(col_img)

    # a = np.random.rand(224, 448, 3)
    #
    # a = np.clip(a, 0, 1)  # 将numpy数组约束在[0, 1]范围内
    # a = (a * 255).astype(np.uint8)
    #
    # im = Image.fromarray(a)
    # print(im.size)
    pass


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.int32)
    seg_gt = np.asarray(
        label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


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
