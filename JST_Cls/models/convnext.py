import os

import torch
from timm.models.convnext import ConvNeXt, _create_convnext, DropPath

"""
这个脚本主要用来定义向ConvNeXt模型中注入两个自定义的函数
"""


# hack: inject the forward_multi_stage function of `timm.models.convnext.ConvNeXt`
def forward_features(self: ConvNeXt, x):
    """ this forward_features function is used to get hierarchical representation
    """
    x = self.stem(x)
    x = self.stages(x)
    return x


def forward_head(self: ConvNeXt, x):
    """ this forward_features function is used to get logits
    """
    x = self.norm_pre(x)
    x = self.head(x)
    return x


ConvNeXt.forward_features = forward_features
ConvNeXt.forward_head = forward_head


def convnext_pico(pretrained=False, **kwargs):
    depths = (2, 2, 6, 2)
    model_args = dict(
        depths=depths, dims=(64, 128, 256, 512), use_grn=False, ls_init_value=1.0, conv_mlp=True,
        drop_rate=0., drop_path_rate=0.1)
    model = _create_convnext('convnextv2_pico', pretrained=False, **dict(model_args, **kwargs))
    model.depths = list(depths)

    if pretrained:
        weight_path = os.path.join('model_weight', 'convnext_pico.pth')
        print('loading weight from {}'.format(weight_path))
        pretrained_dict = torch.load(weight_path)
        if 'head.fc.weight' in pretrained_dict:
            pretrained_dict.pop('head.fc.weight')
        if 'head.fc.bias' in pretrained_dict:
            pretrained_dict.pop('head.fc.bias')
        state = model.load_state_dict(pretrained_dict, strict=False)
        print(state)
    return model


def convnext_little(pretrained=False, **kwargs) -> ConvNeXt:
    depths = (2, 2, 4, 2)
    model_args = dict(depths=(2, 2, 4, 2), dims=(24, 48, 96, 192), use_grn=False, ls_init_value=1.0, conv_mlp=True)
    model = _create_convnext('convnext_little', pretrained=pretrained, **dict(model_args, **kwargs))
    model.depths = list(depths)
    return model


def convnext_lite(pretrained=False, **kwargs) -> ConvNeXt:
    depths = (2, 2, 2, 2)
    model_args = dict(depths=(2, 2, 2, 2), dims=(16, 32, 64, 128), use_grn=False, ls_init_value=1.0, conv_mlp=True)
    model = _create_convnext('convnext_lite', pretrained=pretrained, **dict(model_args, **kwargs))
    model.depths = list(depths)
    return model


if __name__ == '__main__':
    # model = convnext_tiny(num_classes=5)

    # for name, param in model.named_parameters():
    #     print(name)
    # for name, param in model.named_parameters():
    #     print(name)

    # a = torch.rand(2, 3, 224, 224)
    # out1 = model.forward_features(a)
    # print(out1.shape)
    # logits = model.forward_head(out1)
    # print(logits.shape)
    model = convnext_pico(pretrained=False, num_classes=5)
    # out1 = model.forward_features(a)
    # print(out1.shape)
    # logits = model.forward_head(out1)
    # print(logits.shape)
    # for name, param in model.named_parameters():
    #     print(name)
    # for name, module in model.named_modules():
    #     if isinstance(module, DropPath):
    #         print(name)
    # drop_path_rate = 0.1
    # depths = [2, 2, 6, 2]
    # dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths))]
    # print(len(dp_rates))
    # index = 0
    # for name, module in model.named_modules():
    #     if isinstance(module, DropPath):
    #         module.drop_prob = dp_rates[index]
    #         print(index)
    #         index += 1

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total number of parameters is: {total_params}")
