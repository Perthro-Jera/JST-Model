import os
import random
import cv2
import openpyxl
import pandas as pd
import numpy as np
from tifffile import TiffFile
from shutil import copyfile

path_pre_fix = '/ssd_data/segmentation_dataset/'


def create_excel_file(file_dir):
    """
    给定保存图片的目录，生成我们所需要的对应的文件，文件的格式为：
    图片的绝对路径 \t 图片的label \t 病人ID
    :param file_dir: 保存图片的文件夹
    :return:
    """
    l2n_class_dict = {'0': '宫颈炎', '1': '囊肿', '2': '外翻', '3': '低级别病变', '4': '高级别病变', '5': '宫颈癌'}
    n2l_class_dict = {'宫颈炎': '0', '囊肿': '1', '外翻': '2', '低级别病变': '3', '高级别病变': '4', '宫颈癌': '5'}
    file_dir_norm = os.path.normpath(file_dir)
    file_dir_norm_list = file_dir_norm.split(os.sep)

    # 创建存储该数据集使用的excel文件名称
    file_name = file_dir_norm_list[-1] + '.xlsx'
    header = ['文件名', '类别', '病人id']
    # 定义所要输出的Excel文件
    # 创建新的工作簿
    workbook = openpyxl.Workbook()
    # 获取当前活跃的worksheet，默认就是第一个worksheet
    worksheet = workbook.active
    worksheet.title = 'file'
    # 添加结果excel文件的表头
    worksheet.append(header)
    for root, dirs, files in os.walk(os.path.join(path_pre_fix, file_dir), topdown=False):
        # if len(files) > 0 and ('宫颈炎' in root or '囊肿' in root or '外翻' in root):  这一行是用来挑出阴性的图片用来分割使用
        if len(files) > 0 and ('低级别病变' not in root):
            for file in files:
                image_path = os.path.join(root, file)
                path = os.path.normpath(image_path)
                label = path.split(os.sep)[-2]
                patient = '_'.join(file.split('_')[:3])
                worksheet.append([image_path, label, patient])
    file_name = os.path.join('dataset', file_name)
    workbook.save(filename=file_name)
    return file_name


# def create_excel_file():
#     """
#     给定保存图片的目录，生成我们所需要的对应的分割文件，文件的格式为：
#     图片的绝对路径 \t mask的绝对路径 \t 图片的分类类别
#     :return:
#     """
#     l2n_class_dict = {'0': '宫颈炎', '1': '囊肿', '2': '外翻', '3': '低级别病变', '4': '高级别病变', '5': '宫颈癌'}
#     n2l_class_dict = {'宫颈炎': '0', '囊肿': '1', '外翻': '2', '低级别病变': '3', '高级别病变': '4', '宫颈癌': '5'}
#     img_dir = os.path.join(path_pre_fix, 'images')
#     mask_dir = os.path.join(path_pre_fix, 'labels', 'Annotations')
#     img_paths = []
#     mask_paths = []
#     img_classes = []
#     # 遍历img_dir得到所有图片的路径
#     for img_type in os.listdir(img_dir):
#         img_type_dir = os.path.join(img_dir, img_type)
#         for img_name in os.listdir(img_type_dir):
#             img_path = os.path.join(img_type_dir, img_name)
#             img_paths.append(img_path)
#             mask_paths.append(os.path.join(mask_dir, img_type, img_name))
#             img_classes.append(img_type)
#
#     # 创建存储该数据集使用的excel文件名称
#     file_name = 'data.xlsx'
#     header = ['image_path', 'mask_path', 'label']
#     # 定义所要输出的Excel文件
#     # 创建新的工作簿
#     workbook = openpyxl.Workbook()
#     # 获取当前活跃的worksheet，默认就是第一个worksheet
#     worksheet = workbook.active
#     worksheet.title = '第一折'
#     # 添加结果excel文件的表头
#     worksheet.append(header)
#     for i in range(len(img_paths)):
#         worksheet.append([img_paths[i], mask_paths[i], img_classes[i]])
#         print([img_paths[i], mask_paths[i], img_classes[i]])
#     workbook.save(filename=os.path.join('dataset', file_name))


def cal_label_weight():
    """
    该函数用来计算每个分割标签的权重
    """
    df = pd.read_excel(io=os.path.join('dataset', 'data.xlsx'), sheet_name='第一折')
    mask_paths = df['mask_path'].tolist()
    label_count = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    scale = 1e6
    for path in mask_paths:
        print(path)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # ndarray [H, W]
        for i in range(8):
            label_count[i].append(np.sum(mask == i) / scale)
    for i in range(8):
        label_count[i] = np.sum(label_count[i])
    print(label_count)
    # [0: 0.7295, 1: 1012.5066, 2: 679.7656, 3: 94.9533, 4: 246.5242, 5: 369.7600, 6: 95.2811, 7: 1164.8790]
    # ['background', '背景', '上皮', '间隙', '保护套', '凸起', '囊肿', '基质']


def add_file():
    pre_fix = '/ssd_data/dataset/'
    hospitals = ['tiff_xiangyangP1-200_frame', 'tiff_jingzhouP1-168_frame', 'tiff_huaxi_modified_frame',
                 'tiff_xiangya_frame']
    # 创建一个空的DataFrame
    df = pd.DataFrame(columns=['文件名', '类别', '病人id'])
    for h_n in hospitals:
        file_dir = os.path.join(pre_fix, h_n)
        excel_file_name = create_excel_file(file_dir)
        # 按行合并DataFrame
        df_1 = pd.read_excel(io=excel_file_name, sheet_name='file')
        df = pd.concat([df, df_1])
    # 重置索引，使其变回默认的整数索引
    df = df.reset_index()
    # 将DataFrame保存为Excel文件
    df.to_excel(os.path.join('dataset', 'merge.xlsx'), index=False, sheet_name='file', engine='openpyxl')


if __name__ == '__main__':
    # create_excel_file()
    # cal_label_weight()
    add_file()
