import torch
import torch.nn as nn
import os
import torchvision.models as models
from .ae import AutoEncoder  # 假设DRAE包含编码器、解码器和判别器

class ModifiedDenseNet121(nn.Module):
    def __init__(self, pretrained=True, num_classes=5):
        super(ModifiedDenseNet121, self).__init__()

        # 加载预训练的DenseNet121
        densenet121 = models.densenet121(pretrained=pretrained)

        # 去除前两层
        self.features = densenet121.features[2:]  # 假设前两块是densenet121.features[:2]

        # 加载整个DRAE模型
        self.ae_model = AutoEncoder()  # 假设RAE模型包含编码器、解码器和判别器
        # 加载DRAE模型的权重
        self.load_ae_weights()

        # 获取DRAE中的编码器部分
        self.autoencoder_encoder = self.ae_model.encoder

        # 冻结编码器权重
        for param in self.autoencoder_encoder.parameters():
            param.requires_grad = False

        # 新的分类头部
        # 你需要根据自编码器输出特征的维度来修改这个线性层
        # 假设DenseNet121的输出特征图展平后为8192维
        self.fc = nn.Linear(8192, num_classes)  # 修改这里为8192维

    def load_ae_weights(self):
        """
        加载整个DRAE模型的权重到模型中
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(current_dir, '..', 'checkpoint', 'drae', 'rae_checkpoint.pth')
        # 加载保存的DRAE模型权重
        checkpoint = torch.load(weights_path)

        # 加载整个模型的state_dict
        self.ae_model.load_state_dict(checkpoint['model_state_dict'])
        print("RAE weights loaded successfully!")

    def forward(self, x):
        # 通过自编码器的编码器提取特征
        encoded_features = self.autoencoder_encoder(x)

        # 使用DenseNet121提取特征，encoded_features 应该与 DenseNet 剩余部分兼容
        # 这里假设自编码器输出的特征大小已经适配到 DenseNet121 剩余部分
        features = self.features(encoded_features)

        # 使用自适应池化使特征大小一致
        features = torch.flatten(features, 1)  # 展平特征图

        # 通过新的分类层
        out = self.fc(features)
        return out
