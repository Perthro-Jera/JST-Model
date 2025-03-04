import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# 完整的分类模型，使用分割图结果
class ClassifierWithSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        # 传入外部已经训练好的分割模型（MedNeXt）
        #self.segment_model = segment_model

        # 使用ResNet50作为分类器
        self.detection = resnet50(pretrained=True)
        self.detection.fc = nn.Linear(2048, 5)  # 5类分类任务

    def forward(self, x, segmentation_mask):
        # 使用外部分割模型预测分割图（如果你已经有分割结果，可以直接传入）
        # 这里的segmentation_mask是传入的分割结果图（1张图，8类掩膜图）
        masked_od = x * segmentation_mask  # 将分割图应用到输入图像上

        # 合并原图和掩膜图
        combined = torch.cat([x, masked_od], dim=1)  # 拼接图像和掩膜

        # 通过分类模型进行分类任务
        return self.detection(combined)

