import os
import sys
import random
import cv2
import numpy as np
import torch
import itertools
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from util import ImageProcessor

def verify_file_paths(image_paths, mask_paths):
        invalid_image_paths = [path for path in image_paths if not os.path.isfile(path)]
        invalid_mask_paths = [path for path in mask_paths if not os.path.isfile(path)]

        if invalid_image_paths:
            print(f"以下图像文件路径无效: {invalid_image_paths}")
        if invalid_mask_paths:
            print(f"以下标签文件路径无效: {invalid_mask_paths}")

        return len(invalid_image_paths) == 0 and len(invalid_mask_paths) == 0


def verify_clsfile_paths(image_paths):
    invalid_image_paths = [path for path in image_paths if not os.path.isfile(path)]
    if invalid_image_paths:
        print(f"以下图像文件路径无效: {invalid_image_paths}")

    return len(invalid_image_paths) == 0


class SegDataSet(Dataset):

    def __init__(self, args, pattern='train'):
        self.args = args
        self.pattern = pattern
        self.imgs, self.masks, self.labels = self.__load_file(pattern=pattern, fold_num=args.fold_num)

        if not verify_file_paths(self.imgs, self.masks):
            raise ValueError("检测到无效的文件路径，请检查输入数据文件")
        
        print(f"num of samples {len(self.imgs)}")
        self.img_processor = ImageProcessor(args, augmentation=pattern)

    def __getitem__(self, index):
        if self.pattern in ["train", "test", "distill_label"]:
            imgPath = self.imgs[index]
            maskPath = self.masks[index]
            label = self.labels[index]

            # 根据路径读取图片和mask标签
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)  # ndarray [H, W]
            image, mask = self.img_processor(img, mask)  # Tensor [3, H, W], Tensor [H, W]

            return [image, mask, label, imgPath, maskPath]

        elif self.pattern in ["distill_unlabel", 'predict']:
            imgPath = self.imgs[index]
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            image, mask = self.img_processor(img, None)  # Tensor [3, H, W], None
            return [image, imgPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, pattern="train", fold_num=0):
        if pattern in ["train", "test", "distill_label"]:
            file_path = os.path.join("dataset", "data.xlsx")
            l_d = {0: '第一折', 1: '第二折', 2: '第三折', 3: '第四折', 4: '第五折'}
            df_frame = pd.read_excel(file_path, sheet_name=l_d[fold_num])
            # 将label转化为整形
            label_map = {'宫颈炎': 0, '囊肿': 1, '外翻': 2}  # Example mapping
            if pattern in ["train", "distill_label"]:
                # 这个地方先注释掉，后面真正进行5折时再修改回来
                # df_frame = df_frame.iloc[:int(len(df_frame) * 0.8), :]
                pass
            else:
                # 这个地方先注释掉，后面真正进行5折时再修改回来
                # df_frame = df_frame.iloc[int(len(df_frame) * 0.8):, :]
                df_frame = pd.read_excel(file_path, sheet_name=l_d[1])
                pass
            # Filter out rows where label is not in label_map
            df_frame = df_frame[df_frame['label'].isin(label_map.keys())]          
            # Map labels to integers
            df_frame['label'] = df_frame['label'].map(label_map)
            
            df_frame.reset_index(drop=True, inplace=True)
            # 接下来返回文件名列表和frame类别列表
            image_paths = df_frame['image_path'].tolist()
            mask_paths = df_frame['mask_path'].tolist()
            labels = df_frame['label'].tolist()
            return image_paths, mask_paths, labels
        elif pattern in ["distill_unlabel"]:
            file_path = os.path.join("dataset", "unlabeled_data.xlsx")
            df_frame = pd.read_excel(file_path, sheet_name="原始数据")
            image_paths = df_frame['image_path'].tolist()
            return image_paths, [], []
        elif pattern in ["predict"]:
            file_path = os.path.join("dataset", "data.xlsx")
            df_frame = pd.read_excel(file_path, sheet_name="第二折")
            # 我们只读取宫颈炎，囊肿，外翻的图像用来预测分割掩码
            #df_frame = df_frame[df_frame['类别'].isin([3, 4])]
            # df_frame = df_frame[df_frame['类别'].isin(['宫颈炎', '囊肿', '外翻'])]
            image_paths = df_frame['image_path'].tolist()
            print(f'预测的图片总数为{len(image_paths)}')
            return image_paths, [], []


class Vit_Cls_DataSet(Dataset):
    def __init__(self, args, pattern='train'):
        self.args = args
        self.pattern = pattern
        self.imgs, self.masks, self.labels = self.__load_file(pattern=pattern, fold_num=args.fold_num)

        if not verify_file_paths(self.imgs, self.masks):
            raise ValueError("检测到无效的文件路径，请检查输入数据文件")

        print(f"num of samples {len(self.imgs)}")
        self.img_processor = ImageProcessor(args, augmentation=pattern)

    def __getitem__(self, index):
        if self.pattern in ["train", "test", "distill_label"]:
            imgPath = self.imgs[index]
            maskPath = self.masks[index]
            label = self.labels[index]

            # 根据路径读取图片和mask标签
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)  # ndarray [H, W]
            image, mask = self.img_processor(img, mask)  # Tensor [3, H, W], Tensor [H, W]

            return [image, mask, label, imgPath, maskPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, pattern="train", fold_num=0):
        if pattern in ["train", "test"]:
            file_path = os.path.join("dataset", "data_C.xlsx")
            l_d = {0: '第一折', 1: '第二折', 2: '第三折', 3: '第四折', 4: '第五折'}
            # 将label转化为整形
            label_map = {'宫颈炎': 0, '囊肿': 1, '外翻': 2}  # Example mapping
            #label_map = {'宫颈炎': 0, '囊肿': 0, '外翻': 0, '宫颈癌': 1, '高级别病变': 1}  # Example mapping
            if pattern in ["train", "distill_label"]:
                df_frame = pd.read_excel(file_path, sheet_name=l_d[fold_num])
                pass
            else:
                df_frame = pd.read_excel(file_path, sheet_name=l_d[1])
                pass
            # Filter out rows where label is not in label_map
            df_frame = df_frame[df_frame['label'].isin(label_map.keys())]
            # Map labels to integers
            df_frame['label'] = df_frame['label'].map(label_map)

            df_frame.reset_index(drop=True, inplace=True)
            # 接下来返回文件名列表和frame类别列表
            image_paths = df_frame['image_path'].tolist()
            mask_paths = df_frame['mask_path'].tolist()
            labels = df_frame['label'].tolist()
            return image_paths, mask_paths, labels


class TransUnetDataSet(Dataset):

    def __init__(self, args, pattern='train'):
        self.args = args
        self.pattern = pattern
        self.imgs, self.masks, self.labels = self.__load_file(pattern=pattern, fold_num=args.fold_num)

        if not verify_file_paths(self.imgs, self.masks):
            raise ValueError("检测到无效的文件路径，请检查输入数据文件")
        print(f"fold_num: {self.args.fold_num}")
        print(f"num of samples {len(self.imgs)}")
        self.img_processor = ImageProcessor(args, augmentation=pattern)

    def __getitem__(self, index):
        if self.pattern in ["train", "test", "distill_label"]:
            imgPath = self.imgs[index]
            maskPath = self.masks[index]
            label = self.labels[index]

            # 根据路径读取图片和mask标签
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)  # ndarray [H, W]
            image, mask = self.img_processor(img, mask)  # Tensor [3, H, W], Tensor [H, W]

            return [image, mask, label, imgPath, maskPath]

        elif self.pattern in ["distill_unlabel", 'predict']:
            imgPath = self.imgs[index]
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            image, mask = self.img_processor(img, None)  # Tensor [3, H, W], None
            return [image, imgPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, pattern="train", fold_num=0):
        if pattern in ["train", "test", "distill_label"]:
            file_path = os.path.join("dataset", "data_huafen.xlsx")
            l_d = {0: '第一折', 1: '第二折', 2: '第三折', 3: '第四折', 4: '第五折'}
            df_frame = pd.read_excel(file_path, sheet_name=l_d[fold_num])
            # 将label转化为整形
            label_map = {'宫颈炎': 0, '囊肿': 1, '外翻': 2}  # Example mapping
            if pattern in ["train", "distill_label"]:
                # 这个地方先注释掉，后面真正进行5折时再修改回来
                # df_frame = df_frame.iloc[:int(len(df_frame) * 0.8), :]
                pass
            else:
                # 这个地方先注释掉，后面真正进行5折时再修改回来
                # df_frame = df_frame.iloc[int(len(df_frame) * 0.8):, :]
                df_frame = pd.read_excel(file_path, sheet_name=l_d[3])
                pass
            # Filter out rows where label is not in label_map
            df_frame = df_frame[df_frame['label'].isin(label_map.keys())]
            # Map labels to integers
            df_frame['label'] = df_frame['label'].map(label_map)

            df_frame.reset_index(drop=True, inplace=True)
            # 接下来返回文件名列表和frame类别列表
            image_paths = df_frame['image_path'].tolist()
            mask_paths = df_frame['mask_path'].tolist()
            labels = df_frame['label'].tolist()
            return image_paths, mask_paths, labels
        elif pattern in ["distill_unlabel"]:
            file_path = os.path.join("dataset", "unlabeled_data.xlsx")
            df_frame = pd.read_excel(file_path, sheet_name="原始数据")
            image_paths = df_frame['image_path'].tolist()
            return image_paths, [], []
        elif pattern in ["predict"]:
            file_path = os.path.join("dataset", "data_A5.xlsx")
            df_frame = pd.read_excel(file_path, sheet_name="第三折")
            # 我们只读取宫颈炎，囊肿，外翻的图像用来预测分割掩码
            # df_frame = df_frame[df_frame['类别'].isin([3, 4])]
            # df_frame = df_frame[df_frame['类别'].isin(['宫颈炎', '囊肿', '外翻'])]
            image_paths = df_frame['image_path'].tolist()
            print(f'预测的图片总数为{len(image_paths)}')
            return image_paths, [], []

class MedNextDataSet(Dataset):

    def __init__(self, args, pattern='train'):
        self.args = args
        self.pattern = pattern
        self.imgs, self.masks, self.labels = self.__load_file(pattern=pattern, fold_num=args.fold_num)

        if not verify_file_paths(self.imgs, self.masks):
            raise ValueError("检测到无效的文件路径，请检查输入数据文件")
        print(f"fold_num: {self.args.fold_num}")
        print(f"num of samples {len(self.imgs)}")
        self.img_processor = ImageProcessor(args, augmentation=pattern)

    def __getitem__(self, index):
        if self.pattern in ["train", "test", "distill_label"]:
            imgPath = self.imgs[index]
            maskPath = self.masks[index]
            label = self.labels[index]

            # 根据路径读取图片和mask标签
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)  # ndarray [H, W]
            image, mask = self.img_processor(img, mask)  # Tensor [3, H, W], Tensor [H, W]

            return [image, mask, label, imgPath, maskPath]

        elif self.pattern in ["distill_unlabel", 'predict']:
            imgPath = self.imgs[index]
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            image, mask = self.img_processor(img, None)  # Tensor [3, H, W], None
            return [image, imgPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, pattern="train", fold_num=0):
        if pattern in ["train", "test", "distill_label"]:
            file_path = os.path.join("dataset", "data.xlsx")
            l_d = {0: '第一折', 1: '第二折', 2: '第三折', 3: '第四折', 4: '第五折'}
            df_frame = pd.read_excel(file_path, sheet_name=l_d[fold_num])
            # 将label转化为整形
            label_map = {'宫颈炎': 0, '囊肿': 1, '外翻': 2}  # Example mapping
            if pattern in ["train", "distill_label"]:
                # 这个地方先注释掉，后面真正进行5折时再修改回来
                # df_frame = df_frame.iloc[:int(len(df_frame) * 0.8), :]
                pass
            else:
                # 这个地方先注释掉，后面真正进行5折时再修改回来
                # df_frame = df_frame.iloc[int(len(df_frame) * 0.8):, :]
                df_frame = pd.read_excel(file_path, sheet_name=l_d[1])
                pass
            # Filter out rows where label is not in label_map
            df_frame = df_frame[df_frame['label'].isin(label_map.keys())]
            # Map labels to integers
            df_frame['label'] = df_frame['label'].map(label_map)

            df_frame.reset_index(drop=True, inplace=True)
            # 接下来返回文件名列表和frame类别列表
            image_paths = df_frame['image_path'].tolist()
            mask_paths = df_frame['mask_path'].tolist()
            labels = df_frame['label'].tolist()
            return image_paths, mask_paths, labels
        elif pattern in ["distill_unlabel"]:
            file_path = os.path.join("dataset", "unlabeled_data.xlsx")
            df_frame = pd.read_excel(file_path, sheet_name="原始数据")
            image_paths = df_frame['image_path'].tolist()
            return image_paths, [], []
        elif pattern in ["predict"]:
            file_path = os.path.join("dataset", "data.xlsx")
            df_frame = pd.read_excel(file_path, sheet_name="第二折")
            # 我们只读取宫颈炎，囊肿，外翻的图像用来预测分割掩码
            # df_frame = df_frame[df_frame['类别'].isin([3, 4])]
            # df_frame = df_frame[df_frame['类别'].isin(['宫颈炎', '囊肿', '外翻'])]
            image_paths = df_frame['image_path'].tolist()
            print(f'预测的图片总数为{len(image_paths)}')
            return image_paths, [], []

#不需要mask
class TransformerDataSet(Dataset):

    def __init__(self, args, pattern='train'):
        self.args = args
        self.pattern = pattern
        self.imgs, self.textures, self.labels = self.__load_file(pattern=pattern, fold_num=args.fold_num)

        if not verify_clsfile_paths(self.imgs):
            raise ValueError("检测到无效的文件路径，请检查输入数据文件")

        print(f"num of samples {len(self.imgs)}")
        self.img_processor = ImageProcessor(args, augmentation=pattern)


    def __getitem__(self, index):
       
        if self.pattern in ["train", "seg_drae", "drae_train", "test", "seg_train", "texture_train" ]:
            imgPath = self.imgs[index]
            textPath = self.textures[index]
            label = self.labels[index]

            # 根据路径读取图片和标签
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            #img = crop_background(img, self.args.crop_frame_height)  # [H, W, 3]
            #texture = cv2.imread(textPath,cv2.IMREAD_GRAYSCALE)
            texture = cv2.imread(textPath)
            image, texture = self.img_processor(img, texture)  # Tensor [3, H, W], Tensor [3, H, W]

            return [image, texture, label, imgPath, textPath]

        elif self.pattern in ["distill_unlabel", 'predict']:
            imgPath = self.imgs[index]
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            image, mask = self.img_processor(img, None)  # Tensor [3, H, W], None
            return [image, imgPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, pattern="train", fold_num=0):
        l_d = {0: '第一折', 1: '第二折', 2: '第三折', 3: '第四折', 4: '第五折'}
        if pattern in ["train", "seg_drae", "drae_train", "seg_train", "texture_train", "st_train"]:
            file_name = 'internal_A.xlsx'
            #file_name = 'data_A5.xlsx'
            file_path = os.path.join("dataset", file_name)
            df_frame = pd.read_excel(file_path, sheet_name=l_d[fold_num])
        elif pattern == "test":
            if self.args.test_file == 'internal':
                file_name = 'internal_A.xlsx'
                #file_name = 'data_A5.xlsx'
            elif self.args.test_file == 'huaxi':
                file_name = 'data_Bttf.xlsx'
            elif self.args.test_file == 'xiangya':
                file_name = 'data_Cttf.xlsx'
            elif self.args.test_file == 'xiangyang':
                file_name = 'data_D5.xlsx'
            file_path = os.path.join("dataset", file_name)
            #第二折为测试集
            df_frame = pd.read_excel(file_path, sheet_name=l_d[1])
        # 将label转化为整形 5分类
        label_map = {'宫颈炎': 0, '囊肿': 1, '外翻': 2, '高级别病变': 3, '宫颈癌': 4}  # Example mapping
        # Filter out rows where label is not in label_map
        df_frame = df_frame[df_frame['label'].isin(label_map.keys())]
        # Map labels to integers
        df_frame['label'] = df_frame['label'].map(label_map)
        df_frame.reset_index(drop=True, inplace=True)
        # 接下来返回文件名列表和frame类别列表
        image_paths = df_frame['image_path'].tolist()
        texture_paths = df_frame['texture_path'].tolist()
        labels = df_frame['label'].tolist()
        return image_paths, texture_paths, labels
        

class FrameDataSet(Dataset):

    def __init__(self, args, pattern='train'):
        self.args = args
        self.pattern = pattern
        self.imgs, self.textures, self.labels = self.__load_file(pattern=pattern, fold_num=args.fold_num)

        if not verify_clsfile_paths(self.imgs):
            raise ValueError("检测到无效的文件路径，请检查输入数据文件")

        print(f"num of samples {len(self.imgs)}")
        self.img_processor = ImageProcessor(args, augmentation=pattern)


    def __getitem__(self, index):
        if self.pattern in ["train", "test", "distill_label"]:
            imgPath = self.imgs[index]
            textPath = self.textures[index]
            label = self.labels[index]

            # 根据路径读取图片和标签
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            texture = cv2.imread(textPath,cv2.IMREAD_GRAYSCALE)
            image, texture = self.img_processor(img, texture)  # Tensor [3, H, W], Tensor [3, H, W]

            return [image, texture, label, imgPath, textPath]

        elif self.pattern in ["distill_unlabel", 'predict']:
            imgPath = self.imgs[index]
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            image, mask = self.img_processor(img, None)  # Tensor [3, H, W], None
            return [image, imgPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, pattern="train", fold_num=0):
        if pattern in ["train", "test"]:
            file_path = os.path.join("dataset", "data_B5.xlsx")
            l_d = {0: '第一折', 1: '第二折', 2: '第三折', 3: '第四折', 4: '第五折'}
            df_frame = pd.read_excel(file_path, sheet_name=l_d[fold_num])
            # 将label转化为整形 2分类
            label_map = {'宫颈炎': 0, '囊肿': 1, '外翻': 2, '宫颈癌': 3, '高级别病变': 4}  # Example mapping
            # 将label转化为整形 3分类
            #label_map = {'宫颈炎': 0, '囊肿': 1, '外翻': 2}
            if pattern in ["test"]:
                label_map = {'宫颈炎': 0, '囊肿': 0, '外翻': 0, '宫颈癌': 1, '高级别病变': 1}  # Example mapping
                df_frame = pd.read_excel(file_path, sheet_name=l_d[1])

            # Filter out rows where label is not in label_map
            df_frame = df_frame[df_frame['label'].isin(label_map.keys())]
            # Map labels to integers
            df_frame['label'] = df_frame['label'].map(label_map)

            df_frame.reset_index(drop=True, inplace=True)
            # 接下来返回文件名列表和frame类别列表
            image_paths = df_frame['image_path'].tolist()
            texture_paths = df_frame['texture_path'].tolist()
            labels = df_frame['label'].tolist()
            return image_paths, texture_paths, labels
        
#LBP数据处理
class TransformerLBPDataSet(Dataset):

    def __init__(self, args, pattern='train'):
        self.args = args
        self.pattern = pattern
        self.imgs, self.labels = self.__load_file(pattern=pattern, fold_num=args.fold_num)

        if not verify_clsfile_paths(self.imgs):
            raise ValueError("检测到无效的文件路径，请检查输入数据文件")

        print(f"num of samples {len(self.imgs)}")
        self.img_processor = ImageProcessor(args, augmentation=pattern)


    def __getitem__(self, index):
        if self.pattern in ["train", "test", "distill_label"]:
            imgPath = self.imgs[index]
            label = self.labels[index]

            # 根据路径读取图片和标签
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            image, texture = self.img_processor(img, None)  # Tensor [3, H, W], Tensor [3, H, W]

            return [image,  texture, label, imgPath]

        elif self.pattern in ["distill_unlabel", 'predict']:
            imgPath = self.imgs[index]
            img = cv2.imread(imgPath)  # ndarray [H, W, 3]
            image, mask = self.img_processor(img, None)  # Tensor [3, H, W], None
            return [image, imgPath]

    def __len__(self):
        return len(self.imgs)

    def __load_file(self, pattern="train", fold_num=0):
        if pattern in ["train", "test"]:
            file_path = os.path.join("dataset", "data_Attf.xlsx")
            l_d = {0: '第一折', 1: '第二折', 2: '第三折', 3: '第四折', 4: '第五折'}
            df_frame = pd.read_excel(file_path, sheet_name=l_d[fold_num])
            # 将label转化为整形 5分类
            label_map = {'宫颈炎': 0, '囊肿': 1, '外翻': 2, '宫颈癌': 3, '高级别病变': 4}  # Example mapping
            # 将label转化为整形 3分类
            #label_map = {'宫颈炎': 0, '囊肿': 1, '外翻': 2}
            if pattern in ["test"]:
                # 将label转化为整形 2分类
                label_map = {'宫颈炎': 0, '囊肿': 0, '外翻': 0, '宫颈癌': 1, '高级别病变': 1}  # Example mapping
                df_frame = pd.read_excel(file_path, sheet_name=l_d[1])

            # Filter out rows where label is not in label_map
            df_frame = df_frame[df_frame['label'].isin(label_map.keys())]
            # Map labels to integers
            df_frame['label'] = df_frame['label'].map(label_map)

            df_frame.reset_index(drop=True, inplace=True)
            # 接下来返回文件名列表和frame类别列表
            image_paths = df_frame['image_path'].tolist()
            labels = df_frame['label'].tolist()
            return image_paths, labels

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def crop_background(frame_array, height):
    """
    对每一帧的背景进行裁剪
    :param frame_array: shape is [H, W, 3]
    :param image_path: 要裁剪的图片的路径
    :param height: 要裁剪的图片的图片的高度
    :return: 裁剪过后的数组
    """
    frame_array = frame_array[-height:, ...]

    return frame_array
if __name__ == '__main__':
    # file_name = 'train.txt'
    # print(file_name.replace('.txt', '_frame.txt'))
    import random

    a = list(range(20))
    c = random.sample(a, 10)
    print(c)
