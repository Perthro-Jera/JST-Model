import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ResNet18结构用于提取分割图特征
class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Encoder, self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        # 去除原有的分类头部
        self.encoder = nn.Sequential(*list(resnet18.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return x.flatten(1)  # Flatten for the classifier

# DRAE编码器
class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=256):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # (512,1024) -> (256,512)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (256,512) -> (128,256)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (128,256) -> (64,128)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (64,128) -> (32,64)
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # (32,64) -> (16,32)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024 * 16 * 32, latent_dim),  # 映射到 256 维度
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

# DRAE解码器
class Decoder(nn.Module):
    def __init__(self, latent_dim=256, output_channels=1):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 16 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (1024, 16, 32)),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 归一化到 [0,1]
        )

    def forward(self, x):
        return self.decoder(x)

# 自注意力机制用于特征融合
class SelfAttentionFusion(nn.Module):
    def __init__(self, latent_dim, feature_dim, num_heads=8):
        super(SelfAttentionFusion, self).__init__()

        # 使用 Multi-Head Attention 来进行特征融合
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

        # 将 DRAE 的输出维度从 latent_dim 调整到 feature_dim
        self.linear = nn.Linear(latent_dim, feature_dim)

    def forward(self, z_segmentation, z_original):
        # 将 DRAE 输出的特征维度从 latent_dim 转换为 feature_dim
        z_original_aligned = self.linear(z_original)  # 调整后维度是 feature_dim (512)

        # 需要转换成 (batch_size, seq_len, feature_dim) 形式
        z_original_aligned = z_original_aligned.unsqueeze(1)  # 添加一个序列长度维度
        z_segmentation = z_segmentation.unsqueeze(1)  # 同样处理 segmentation 特征

        # 自注意力机制进行特征融合
        attn_output, _ = self.attn(z_original_aligned, z_segmentation, z_segmentation)

        # 输出是加权后的融合特征
        fused_features = attn_output.squeeze(1)  # 去除序列长度维度

        return fused_features


# ------------------------------
# 4. 交叉注意力融合模块（CrossAttentionFusion）
# ------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, in_dim_x, in_dim_y, embed_dim=256, num_heads=4, out_dim=512):
        """
        Args:
            in_dim_x: 结构特征的输入维度（例如512）
            in_dim_y: 纹理特征的输入维度（例如256）
            embed_dim: 统一投影后的嵌入维度（例如256）
            num_heads: 注意力头数
            out_dim: 最终融合特征的维度（例如512）
        """
        super(CrossAttentionFusion, self).__init__()
        # 将两路特征分别投影到 embed_dim
        self.fc_x = nn.Linear(in_dim_x, embed_dim)
        self.fc_y = nn.Linear(in_dim_y, embed_dim)

        # 双向交叉注意力：分别让 x 查询 y 和 y 查询 x
        self.mha_xy = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mha_yx = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 融合层，将两个方向的交叉注意力结果拼接后经过 MLP 处理
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, y):
        # x: [B, in_dim_x]（结构特征）， y: [B, in_dim_y]（纹理特征）
        # 分别投影后扩展维度为序列维度（B,1,embed_dim）
        x_proj = self.fc_x(x).unsqueeze(1)
        y_proj = self.fc_y(y).unsqueeze(1)

        # 交叉注意力：x 查询 y
        x_att, _ = self.mha_xy(x_proj, y_proj, y_proj)  # 输出形状：[B,1,embed_dim]
        # 交叉注意力：y 查询 x
        y_att, _ = self.mha_yx(y_proj, x_proj, x_proj)  # 输出形状：[B,1,embed_dim]

        # 残差连接：加上原始投影
        fused_x = x_proj + x_att  # [B,1,embed_dim]
        fused_y = y_proj + y_att  # [B,1,embed_dim]

        # 去除序列维度
        fused_x = fused_x.squeeze(1)  # [B,embed_dim]
        fused_y = fused_y.squeeze(1)  # [B,embed_dim]

        # 拼接两个方向的信息
        fused = torch.cat([fused_x, fused_y], dim=1)  # [B,2*embed_dim]
        # 经过 MLP 融合得到最终特征
        fused = self.mlp(fused)  # [B,out_dim]
        return fused

class ConcatFusion(nn.Module):
    def __init__(self):
        super(ConcatFusion, self).__init__()

    def forward(self, z_segmentation, z_original):
        # 直接拼接分割图和原图的特征
        fused_features = torch.cat([z_segmentation, z_original], dim=1)  # 拼接在特征维度
        return fused_features

# 主模型
class FusionModel(nn.Module):
    def __init__(self, input_channels=1, latent_dim=256, class_num=5):
        super(FusionModel, self).__init__()
        # 提取分割图的结构特征（512 维）
        self.resnet18 = ResNet18Encoder(pretrained=True)
        # 提取原图的纹理特征（latent_dim 维，例如256）
        self.drae_encoder = Encoder(input_channels, latent_dim)
        # 重构原图的解码器
        self.drae_decoder = Decoder(latent_dim, input_channels)

        # 交叉注意力融合模块：
        # 传入的结构特征维度为 512，纹理特征维度为 latent_dim（256）
        self.concat_fusion = ConcatFusion()
        self.self_attention_fusion = SelfAttentionFusion(latent_dim=latent_dim, feature_dim=512)
        self.cross_attention_fusion = CrossAttentionFusion(
            in_dim_x=512,
            in_dim_y=latent_dim,
            embed_dim=256,
            num_heads=4,
            out_dim=512
        )

        # 分类分支，使用融合后的 512 维特征
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, class_num)

        self.bn = nn.BatchNorm1d(latent_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, original_image, segmentation_image):
        # 分别提取纹理特征和结构特征
        z_original = self.drae_encoder(original_image)  # [B, latent_dim]，例如256维
        z_segmentation = self.resnet18(segmentation_image)  # [B, 512]

        # 利用交叉注意力融合模块进行特征交互和融合
        fused_features = self.cross_attention_fusion(z_segmentation, z_original)  # [B, 512]
        #fused_features = self.self_attention_fusion(z_segmentation, z_original)
        # 直接拼接特征
        #fused_features = self.concat_fusion(z_segmentation, z_original)  # [B, 768]
        # 分类分支
        x = F.relu(self.fc1(fused_features))
        #x = F.relu(self.fc1(z_original))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        # z = self.bn(z_original)
        # z = self.dropout(z)

        # 重构分支：利用DRAE解码器重构原图
        recon_x = self.drae_decoder(z_original)

        return output, z_original, recon_x

if __name__ == "__main__":
    # 输入数据大小
    input_channels = 1  # 灰度图
    latent_dim = 256
    class_num = 5  # 类别数修改为5
    model = FusionModel(input_channels, latent_dim, class_num)

    # 假设分割图和原图大小都是 512x1024
    original_image = torch.randn(8, 1, 512, 1024)  # Batch size 8
    segmentation_image = torch.randn(8, 1, 512, 1024)

    output, recon_x = model(original_image, segmentation_image)

    print("Output shape:", output.shape)  # (8, 5), 对应 5 个类别
    print("Reconstructed shape:", recon_x.shape)  # (8, 1, 512, 1024)，重构后的图像
