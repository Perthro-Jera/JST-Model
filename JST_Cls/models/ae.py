import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_channels=64):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 256, 512]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 128, 256]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 64, 128]
            nn.ReLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=2, padding=1),  # [B, latent_channels, 32, 64]
            nn.ReLU()
        )

    def forward(self, x):
        # 只返回编码后的特征
        return self.encoder(x)


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, output_channels=1, latent_channels=64):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # [B, 128, 64, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, 128, 256]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, 256, 512]
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # [B, 1, 512, 1024]
            nn.Sigmoid()  # 使输出在[0, 1]之间
        )

    def forward(self, x):
        # 返回解码后的重构图像
        return self.decoder(x)


# 定义完整的自动编码器模型（包含编码器和解码器）
class AutoEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_channels=64, output_channels=1):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(input_channels=input_channels, latent_channels=latent_channels)
        self.decoder = Decoder(output_channels=output_channels, latent_channels=latent_channels)

    def forward(self, x):
        # 编码器输出特征
        encoded = self.encoder(x)

        # 解码器重构图像
        decoded = self.decoder(encoded)

        return decoded


# 实例化模型
# encoder = Encoder(input_channels=1, latent_channels=64)
# decoder = Decoder(output_channels=1, latent_channels=64)
# autoencoder = AutoEncoder(input_channels=1, latent_channels=64)
#
# # 测试
# input_data = torch.randn(8, 1, 512, 1024)  # 假设 batch_size=8
#
# # 仅使用编码器获取特征
# encoded_features = encoder(input_data)
# print(f"Encoded features shape: {encoded_features.shape}")  # 应为 [8, 64, 32, 64]
#
# # 使用解码器重构图像
# reconstructed = decoder(encoded_features)
# print(f"Reconstructed shape: {reconstructed.shape}")  # 应为 [8, 1, 512, 1024]
#
# # 使用完整的自动编码器
# reconstructed_full, encoded_full = autoencoder(input_data)
# print(f"Reconstructed shape (full AE): {reconstructed_full.shape}")  # 应为 [8, 1, 512, 1024]
# print(f"Encoded features shape (full AE): {encoded_full.shape}")  # 应为 [8, 64, 32, 64]
