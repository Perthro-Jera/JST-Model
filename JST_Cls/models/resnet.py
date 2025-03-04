import os
import torch.nn as nn
import torch
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.functional as F

"""
这个脚本主要用来定义向Resnet模型中注入两个自定义的函数
"""


# ========== 1. 定义 Decoder 结构 ==========
# class SimpleDecoder(nn.Module):
#     def __init__(self, in_channels=512, out_channels=3):
#         super(SimpleDecoder, self).__init__()
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),  # 16x32 -> 32x64
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x64 -> 64x128
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x128 -> 128x256
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 128x256 -> 256x512
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),  # 256x512 -> 512x1024
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.deconv(x)


# hack: inject the forward_multi_stage function of `timm.models.convnext.ConvNeXt`
def forward_features(self: ResNet, x):
    """ this forward_features function is used to get hierarchical representation
    """
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)  # [B, C, H, W]

    # 通过 decoder 进行重构
    # x_recon = self.decoder(x)  # [B, 3, H, W]
    return x


def forward_head(self: ResNet, x):
    """ this forward_features function is used to get logits
    """
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x

# # ========== 4. 修改 ResNet 的 forward 方法 ==========
# def forward(self: ResNet, x):
#     """ 让模型返回分类结果和重构图像 """
#     features, x_recon = self.forward_features(x)  # 提取特征 & 重构
#     logits = self.forward_head(features)  # 进行分类
#     return logits, x_recon  # 返回 (分类, 重构)

ResNet.forward_features = forward_features
ResNet.forward_head = forward_head
#ResNet.forward = forward  # 这一步是关键！

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    # 在 ResNet 里添加 Decoder
    #model.decoder = SimpleDecoder(in_channels=512, out_channels=3)  # ResNet18 的最后通道数是 512

    if pretrained:
        weight_path = os.path.join('model_weight', 'resnet18.pth')
        print('loading weight from {}'.format(weight_path))
        pretrained_dict = torch.load(weight_path)
        if 'fc.weight' in pretrained_dict:
            pretrained_dict.pop('fc.weight')
        if 'fc.bias' in pretrained_dict:
            pretrained_dict.pop('fc.bias')
        state = model.load_state_dict(pretrained_dict, strict=False)
        print(state)
    return model


if __name__ == '__main__':
    model = resnet18(num_classes=5)

    # for name, param in model.named_parameters():
    #     print(name)
    for name, param in model.layer4.named_parameters():
        print(name)

    # a = torch.rand(2, 3, 224, 224)
    # out1 = model.forward_features(a)
    # print(out1.shape)
    # logits = model.forward_head(out1)
    # print(logits.shape)
