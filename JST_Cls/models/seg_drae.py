import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ResNet18结构用于提取分割图特征
class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Encoder, self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)

        # 修改第一个卷积层的卷积核大小和步幅
        # 将7x7的卷积核改为3x3，步幅改为1
        resnet18.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 修改maxpool层的步幅，原始是2，现在改为1以减缓尺寸缩小
        resnet18.maxpool = nn.Identity()  # 去掉maxpool层

        # 去除分类头部
        self.encoder = nn.Sequential(*list(resnet18.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return x.flatten(1)


# class Encoder(nn.Module):
#     def __init__(self, input_channels=1, latent_dim=256):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # (256,512) -> (128,256)
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128,256) -> (64,128)
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (64,128) -> (32,64)
#             nn.ReLU(),
#             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (32,64) -> (16,32)
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(512 * 16 * 32, latent_dim),  # 映射到 256 维度
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         return self.encoder(x)
#
#
# class Decoder(nn.Module):
#     def __init__(self, latent_dim=256, output_channels=1):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 512 * 16 * 32),
#             nn.ReLU(),
#             nn.Unflatten(1, (512, 16, 32)),
#             nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()  # 归一化到 [0,1]
#         )
#
#     def forward(self, x):
#         return self.decoder(x)

class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=256):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # 256x512 → 128x256
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128x256 → 64x128
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 64x128 → 32x64
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 32x64 → 16x32
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 输出变为 [B, 512, 1, 1]
            nn.Flatten(),  # 输出为 [B, 512]
            nn.Linear(512, latent_dim),  # 参数仅 512×256 = 131072
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=256, output_channels=1):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 16),  # 更小的展开目标
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 16)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (16,32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # -> (32,64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),    # -> (64,128)
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (128,256)
            nn.Upsample(size=(256, 512), mode='bilinear', align_corners=False),  # 最后补齐到原图大小
            nn.Sigmoid()
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



class ResNet18Backbone(nn.Module):
    """
    返回 ResNet18 的多层特征:
        c1: [B, 64, H/4,  W/4]
        c2: [B, 128, H/8, W/8]
        c3: [B, 256, H/16, W/16]
        c4: [B, 512, H/32, W/32]
    """
    def __init__(self, input_channels=1, pretrained=True):
        super(ResNet18Backbone, self).__init__()

        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)

        # 改第一层卷积，适配单通道 OCT
        if input_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )

            if pretrained:
                if input_channels == 1:
                    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

            resnet.conv1 = new_conv

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

    def forward(self, x):
        x = self.conv1(x)     # H/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # H/4

        c1 = self.layer1(x)   # [B, 64, H/4,  W/4]
        c2 = self.layer2(c1)  # [B, 128, H/8, W/8]
        c3 = self.layer3(c2)  # [B, 256, H/16, W/16]
        c4 = self.layer4(c3)  # [B, 512, H/32, W/32]

        return c1, c2, c3, c4


class ResNet18FPNTextureEncoder(nn.Module):
    """
    预训练 ResNet18 + FPN
    输入: 原始OCT图像 [B, 1, H, W]
    输出: 纹理特征 [B, latent_dim]
    """
    def __init__(self, input_channels=1, latent_dim=256, pretrained=True, fpn_dim=128):
        super(ResNet18FPNTextureEncoder, self).__init__()

        self.backbone = ResNet18Backbone(
            input_channels=input_channels,
            pretrained=pretrained
        )

        # lateral conv
        self.lateral_c1 = nn.Conv2d(64,  fpn_dim, kernel_size=1)
        self.lateral_c2 = nn.Conv2d(128, fpn_dim, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(256, fpn_dim, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(512, fpn_dim, kernel_size=1)

        # smooth conv
        self.smooth_p1 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
        self.smooth_p2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        # 4个尺度拼接后映射成 latent_dim
        self.fc = nn.Linear(fpn_dim * 4, latent_dim)

    def _upsample_add(self, x, y):
        x = F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False)
        return x + y

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x)

        # top-down pathway
        p4 = self.lateral_c4(c4)
        p3 = self._upsample_add(p4, self.lateral_c3(c3))
        p2 = self._upsample_add(p3, self.lateral_c2(c2))
        p1 = self._upsample_add(p2, self.lateral_c1(c1))

        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)
        p2 = self.smooth_p2(p2)
        p1 = self.smooth_p1(p1)

        # global pooling
        g1 = self.pool(p1).flatten(1)   # [B, fpn_dim]
        g2 = self.pool(p2).flatten(1)
        g3 = self.pool(p3).flatten(1)
        g4 = self.pool(p4).flatten(1)

        feat = torch.cat([g1, g2, g3, g4], dim=1)  # [B, fpn_dim*4]
        feat = self.fc(feat)                        # [B, latent_dim]

        return feat


# 主模型
class FusionModel(nn.Module):
    def __init__(self, input_channels=1, latent_dim=256, class_num=5,texture_extractor='drae'):
        """
        texture_extractor:
            - 'drae' : 使用原有DRAE编码器提取纹理特征
            - 'fpn'  : 使用特征金字塔提取纹理特征
        """
        super(FusionModel, self).__init__()
        assert texture_extractor in ['drae', 'fpn'], "texture_extractor must be 'drae' or 'fpn'"
        self.texture_extractor_type = texture_extractor
        self.latent_dim = latent_dim

        # 提取分割图的结构特征（512 维）
        self.resnet18 = ResNet18Encoder(pretrained=True)

        # 纹理分支：两种可选
        if self.texture_extractor_type == 'drae':
            self.texture_encoder = Encoder(input_channels, latent_dim) # 提取原图的纹理特征（latent_dim 维，例如256）
            self.texture_decoder = Decoder(latent_dim, input_channels)  # 仅DRAE需要重构
        elif self.texture_extractor_type == 'fpn':
            self.texture_encoder = ResNet18FPNTextureEncoder(
                input_channels=input_channels,
                latent_dim=latent_dim,
                pretrained=True,
                fpn_dim=128
            )
            self.texture_decoder = None

        # 交叉注意力融合模块：
        # 传入的结构特征维度为 512，纹理特征维度为 latent_dim（256）
        #self.concat_fusion = ConcatFusion()
        #self.self_attention_fusion = SelfAttentionFusion(latent_dim=latent_dim, feature_dim=512)
        self.cross_attention_fusion = CrossAttentionFusion(
             in_dim_x=512,
             in_dim_y=latent_dim,
             embed_dim=256,
             num_heads=4,
             out_dim=512
        )


        # 分类分支，使用融合后的 512 维特征
        self.fc1 = nn.Linear(512, 1024)
        #self.fc1 = nn.Linear(768, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, class_num)

        self.bn = nn.BatchNorm1d(latent_dim)
        self.dropout = nn.Dropout(0.2)

        self.texture_proj = nn.Linear(latent_dim, 512)

    def forward(self, original_image, segmentation_image):
        # 分别提取纹理特征和结构特征
        z_texture = self.texture_encoder(original_image)     # [B, latent_dim]
        z_segmentation = self.resnet18(segmentation_image)  # [B, 512]
        # print("z_texture", z_texture.size())
        # print("z_segmentation", z_segmentation.size())

        # 利用交叉注意力融合模块进行特征交互和融合
        fused_features = self.cross_attention_fusion(z_segmentation, z_texture)  # [B, 512]
        #fused_features = self.self_attention_fusion(z_segmentation, z_original)
        # 直接拼接特征
        #fused_features = self.concat_fusion(z_segmentation, z_texture)  # [B, 768]
        # 分类分支
        x = F.relu(self.fc1(fused_features))
        #z_original = self.drae_encoder(original_image)  # [B, 256]
        # z_texture_512 = self.texture_proj(z_original)  # [B, 512]
        #x = F.relu(self.fc1(z_texture_512))
        #x = F.relu(self.fc1(z_segmentation))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)


        # z = self.bn(z_original)
        # z = self.dropout(z)

        # 重构分支：利用DRAE解码器重构原图
        # 重构分支：只有 DRAE 模式下存在
        if self.texture_extractor_type == 'drae':
            recon_x = self.texture_decoder(z_texture)
        else:
            recon_x = None

        return output, z_texture, recon_x

if __name__ == "__main__":
    # 输入数据大小
    input_channels = 1  # 灰度图
    latent_dim = 256
    class_num = 5  # 类别数修改为5
    model = FusionModel(input_channels, latent_dim, class_num)

    # 假设分割图和原图大小都是 512x1024
    original_image = torch.randn(8, 1, 256, 512)  # Batch size 8
    segmentation_image = torch.randn(8, 3, 256, 512)

    output, _, recon_x = model(original_image, segmentation_image)

    print("Output shape:", output.shape)  # (8, 5), 对应 5 个类别
    print("Reconstructed shape:", recon_x.shape)  # (8, 1, 512, 1024)，重构后的图像
