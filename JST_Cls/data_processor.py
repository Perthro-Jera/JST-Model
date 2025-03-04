"""
这个模块用来将得到的分割数据路径写入到Excel表格中，并将数据划分为5折
"""
import os
import openpyxl
import pandas as pd

header = ['image_path', 'mask_path', 'label']
l_d = {0: '第一折', 1: '第二折', 2: '第三折', 3: '第四折', 4: '第五折'}


def write_to_excel(path_pre="/ssd_data/segmentation_dataset"):
    """
    将数据写入到Excel表格中
    :return:
    """
    # 打开存储数据路径的Excel表格
    workbook = openpyxl.load_workbook('dataset/data.xlsx')
    worksheet = workbook['原始数据']
    worksheet.append(header)
    # 遍历一个文件夹下的所有文件
    images, masks, labels = [], [], []
    for label_name in os.listdir(path_pre):
        for file_name in os.listdir(os.path.join(path_pre, label_name, 'images')):
            images.append(os.path.join(path_pre, label_name, 'images', file_name))
            masks.append(os.path.join(path_pre, label_name, 'masks', file_name))
            labels.append(label_name)
    for image, mask, label in zip(images, masks, labels):
        worksheet.append([image, mask, label])
    workbook.save('dataset/data.xlsx')


def split_data(file_path="dataset/data.xlsx", fold_nums=5):
    """
    将数据划分为5折
    :return:
    """
    # 打开存储数据路径的Excel表格
    df = pd.read_excel('dataset/data.xlsx', sheet_name='原始数据')
    yanzheng = df[df['label'] == "炎症"]
    nangzhong = df[df['label'] == "囊肿"]
    waifan = df[df['label'] == "外翻"]

    # 根据label标签划分五折，保证每一折中各个类别的比例相同
    yanzheng_split_num = int(yanzheng.shape[0] / fold_nums)
    nangzhong_split_num = int(nangzhong.shape[0] / fold_nums)
    waifan_split_num = int(waifan.shape[0] / fold_nums)
    fold_list = [[] for _ in range(fold_nums)]
    for i in range(fold_nums):
        ith_yanzheng_test = yanzheng.iloc[i * yanzheng_split_num:(i + 1) * yanzheng_split_num]
        ith_yanzheng_train = pd.concat([yanzheng, ith_yanzheng_test, ith_yanzheng_test]).drop_duplicates(keep=False)
        ith_nangzhong_test = nangzhong.iloc[i * nangzhong_split_num:(i + 1) * nangzhong_split_num]
        ith_nangzhong_train = pd.concat([nangzhong, ith_nangzhong_test, ith_nangzhong_test]).drop_duplicates(keep=False)
        ith_waifan_test = waifan.iloc[i * waifan_split_num:(i + 1) * waifan_split_num]
        ith_waifan_train = pd.concat([waifan, ith_waifan_test, ith_waifan_test]).drop_duplicates(keep=False)
        fold_list[i].extend(
            [ith_yanzheng_train, ith_nangzhong_train, ith_waifan_train, ith_yanzheng_test, ith_nangzhong_test,
             ith_waifan_test])

    with pd.ExcelWriter('dataset/data.xlsx') as writer:
        for i in range(fold_nums):
            pd.concat(fold_list[i]).to_excel(writer, sheet_name=l_d[i], index=False)
    print('数据集划分完成')


if __name__ == '__main__':
    # 现将路径写入文件
    write_to_excel()
    # 将文件路径划分为五折
    split_data()
