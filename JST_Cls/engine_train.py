import numpy as np
import torch
import torch.nn as nn
import time
import os
from util import DataUpdater, get_confusion_matrix, adjust_learning_rate
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms as T
from torch.cuda.amp import autocast, GradScaler
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray  # 导入灰度转换函数
import matplotlib.pyplot as plt



def train_one_epoch(model, train_loader, optimizer, segmentation_criterion, classification_criterion,
                    epoch, total_epoches, num_ite_per_epoch, base_lr):
    # Training
    model.train()
    total_iters = num_ite_per_epoch * total_epoches
    batch_time = DataUpdater()
    ave_total_loss = DataUpdater()
    ave_segmentation_loss = DataUpdater()
    ave_classification_loss = DataUpdater()
    tic = time.time()
    cur_iters = epoch * num_ite_per_epoch
    device = next(model.parameters()).device

    for i_iter, batch in enumerate(train_loader):
        images, masks, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)
        labels = torch.tensor(labels).clone().detach().long().to(device, non_blocking=True)

        # 获取模型的两个输出
        segmentation_output, classification_output = model(images)

        # 分别计算两个输出的损失
        segmentation_loss = segmentation_criterion(segmentation_output, masks)
        
        classification_loss = classification_criterion(classification_output, labels)


        # 总损失是两个损失的加权和，可以根据需要调整权重
        total_loss = segmentation_loss + classification_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average losses
        ave_total_loss.update(total_loss.item())
        ave_segmentation_loss.update(segmentation_loss.item())
        ave_classification_loss.update(classification_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  total_iters,
                                  i_iter + cur_iters)

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Total Loss: {:.6f}, Segmentation Loss: {:.6f}, Classification Loss: {:.6f}'.format(
                      epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                      batch_time.avg, [x['lr'] for x in optimizer.param_groups],
                      ave_total_loss.avg, ave_segmentation_loss.avg, ave_classification_loss.avg)
            print(msg)

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')



def train_distill_one_epoch(model_t, model_s, train_loader, optimizer, criterion, criterion_kd,
                            epoch, total_epoches, num_ite_per_epoch, base_lr, args):
    # Training
    model_t.eval()
    model_s.train()
    total_iters = num_ite_per_epoch * total_epoches
    batch_time = DataUpdater()
    ave_loss = DataUpdater()
    ave_cls_loss = DataUpdater()
    ave_kl_loss = DataUpdater()
    tic = time.time()
    cur_iters = epoch * num_ite_per_epoch
    device = next(model_s.parameters()).device

    for i_iter, batch in enumerate(train_loader):
        images, masks, _, _, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)

        outputs_student = model_s(images)
        with torch.no_grad():
            outputs_teacher = model_t(images)
            outputs_teacher = outputs_teacher.detach()

        loss_cls = criterion(outputs_student, masks)
        outputs_teacher = rearrange(outputs_teacher, 'b c h w -> (b h w) c')
        outputs_student = rearrange(outputs_student, 'b c h w -> (b h w) c')

        loss_kl = criterion_kd(outputs_student, outputs_teacher)

        loss = args.cls_weight * loss_cls + args.kl_weight * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_cls_loss.update(loss_cls.item())
        ave_kl_loss.update(loss_kl.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  total_iters,
                                  i_iter + cur_iters)

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Cls_Loss: {:.6f}, KL_Loss: {:.6f},'.format(epoch, total_epoches, i_iter,
                                                                                    num_ite_per_epoch,
                                                                                    batch_time.avg, [x['lr'] for x in
                                                                                                     optimizer.param_groups],
                                                                                    ave_loss.avg,
                                                                                    ave_cls_loss.avg, ave_kl_loss.avg)
            print(msg)


def train_distill_unlabel_one_epoch(model_t, model_s, train_loader, optimizer, criterion_kd,
                                    epoch, total_epoches, num_ite_per_epoch, base_lr):
    # Training
    model_t.eval()
    model_s.train()
    total_iters = num_ite_per_epoch * total_epoches
    batch_time = DataUpdater()
    ave_loss = DataUpdater()
    ave_kl_loss = DataUpdater()
    tic = time.time()
    cur_iters = epoch * num_ite_per_epoch
    device = next(model_s.parameters()).device

    for i_iter, batch in enumerate(train_loader):
        images, masks, _, _, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)

        outputs_student = model_s(images)  # [batch_size, num_class, h, w]
        with torch.no_grad():
            outputs_teacher = model_t(images)
            outputs_teacher = outputs_teacher.detach()

        outputs_teacher = rearrange(outputs_teacher, 'b c h w -> (b h w) c')
        outputs_student = rearrange(outputs_student, 'b c h w -> (b h w) c')

        loss_kl = criterion_kd(outputs_student, outputs_teacher)

        loss = loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_kl_loss.update(loss_kl.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  total_iters,
                                  i_iter + cur_iters)

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, KL_Loss: {:.6f},'.format(epoch, total_epoches, i_iter,
                                                                  num_ite_per_epoch,
                                                                  batch_time.avg, [x['lr'] for x in
                                                                                   optimizer.param_groups],
                                                                  ave_loss.avg,
                                                                  ave_kl_loss.avg)
            print(msg)


def train_vit_one_epoch(model, train_loader, optimizer,  classification_criterion,
                    epoch, total_epoches, num_ite_per_epoch, base_lr):
    # Training
    model.train()
    total_iters = num_ite_per_epoch * total_epoches
    batch_time = DataUpdater()
    ave_classification_loss = DataUpdater()
    tic = time.time()
    cur_iters = epoch * num_ite_per_epoch
    device = next(model.parameters()).device

    for i_iter, batch in enumerate(train_loader):
        images, _, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        labels = torch.tensor(labels).clone().detach().long().to(device, non_blocking=True)


        classification_output, _ = model(images)

        classification_loss = classification_criterion(classification_output, labels)

        optimizer.zero_grad()
        classification_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average losses
        ave_classification_loss.update(classification_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  total_iters,
                                  i_iter + cur_iters)

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {},  Classification Loss: {:.6f}'.format(
                epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                batch_time.avg, [x['lr'] for x in optimizer.param_groups],
                ave_classification_loss.avg)
            print(msg)

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')
    return ave_classification_loss.avg


def train_transUnet_one_epoch(model, train_loader, optimizer, segmentation_criterion,
                              epoch, total_epoches, num_ite_per_epoch, base_lr):
    # Training
    model.train()
    total_iters = num_ite_per_epoch * total_epoches
    batch_time = DataUpdater()
    ave_segmentation_loss = DataUpdater()
    tic = time.time()
    cur_iters = epoch * num_ite_per_epoch
    device = next(model.parameters()).device

    for i_iter, batch in enumerate(train_loader):
        images, masks, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)

        # 获取模型的输出
        segmentation_output, x1_out, x2_out, x3_out = model(images)

        # 计算分割的损失
        final_loss = segmentation_criterion(segmentation_output, masks)
        loss_1 = segmentation_criterion(x1_out, masks)
        loss_2 = segmentation_criterion(x2_out, masks)
        loss_3 = segmentation_criterion(x3_out, masks)
        segmentation_loss = final_loss + 0.5 * (loss_1 + loss_2 + loss_3)

        optimizer.zero_grad()
        segmentation_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average losses
        ave_segmentation_loss.update(segmentation_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  total_iters,
                                  i_iter + cur_iters)

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {},  Segmentation Loss: {:.6f}, '.format(
                epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                batch_time.avg, [x['lr'] for x in optimizer.param_groups],
                ave_segmentation_loss.avg,)
            print(msg)

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')

    return ave_segmentation_loss.avg


def compute_loss(segmentation_criterion, predictions, targets):
    """
    计算深度监督损失的总和，假设 predictions 是 [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
    """
    loss = 0.0
    # 假设 targets 是正确的标签，形状是 [batch_size, height, width]
    main_loss = segmentation_criterion(predictions[0], targets)  # 主输出的损失
    loss += main_loss

    # 为每个深度监督输出计算损失
    # 你可以根据需要调整每个输出的权重
    loss_weights = [0.1, 0.2, 0.3, 0.4]  # 例如，后面的层权重较大

    for i in range(1, len(predictions)):
        ds_loss = segmentation_criterion(predictions[i], targets)
        loss += loss_weights[i - 1] * ds_loss  # 根据权重调整各层的损失

    return loss

def compute_deep_supervision_loss(outputs, target, segmentation_criterion, deep_supervision=True, weight=0.5):
    """
    计算深度监督损失。
    outputs: 一个元组，包含 (segmentation_output, x1_out, x2_out, x3_out)。
    target: 真实标签。
    deep_supervision: 是否启用深度监督。
    weight: 深度监督损失的权重。
    """
    segmentation_output = outputs[0]  # 最终的分割输出
    total_loss = segmentation_criterion(segmentation_output, target)  # 对最终输出计算损失

    if deep_supervision:
        # 对每个阶段的输出进行深度监督损失计算
        loss_1 = segmentation_criterion(outputs[1], target)
        loss_2 = segmentation_criterion(outputs[2], target)
        loss_3 = segmentation_criterion(outputs[3], target)

        # 总深度监督损失是所有损失的加权和
        total_loss += weight * (loss_1 + loss_2 + loss_3)

    return total_loss

def compute_deep_supervision_loss(outputs, target, segmentation_criterion, deep_supervision=True, weight=0.5):
    """
    计算深度监督损失。
    outputs: 一个元组，包含 (segmentation_output, x1_out, x2_out, x3_out)。
    target: 真实标签。
    deep_supervision: 是否启用深度监督。
    weight: 深度监督损失的权重。
    """
    if deep_supervision:
        # 对每个阶段的输出进行深度监督损失计算
        print("Deep supervision  ")
        segmentation_output = outputs[0]  # 最终的分割输出
        total_loss = segmentation_criterion(segmentation_output, target)  # 对最终输出计算损失
        loss_1 = segmentation_criterion(outputs[1], target)
        loss_2 = segmentation_criterion(outputs[2], target)
        loss_3 = segmentation_criterion(outputs[3], target)

        # 总深度监督损失是所有损失的加权和
        total_loss += weight * (loss_1 + loss_2 + loss_3)
    else:
        total_loss = segmentation_criterion(outputs, target)  # 对最终输出计算损失

    return total_loss

def compute_consistency_loss(output1, output2, consistency_criterion, deep_supervision=True, weight=0.5):
    """
    计算深度监督一致性损失。
    Args:
        output1: 来自网络1的输出元组 (segmentation_output, x1_out, x2_out, x3_out)。
        output2: 来自网络2的输出元组 (segmentation_output, x1_out, x2_out, x3_out)。
        consistency_criterion: 一致性损失函数（如L2或KL散度）。
        deep_supervision: 是否启用深度监督。
        weight: 深度监督损失的权重。
    Returns:
        total_loss: 总的一致性损失。
    """
    # 对最终输出计算一致性损失
    # 在获取无标签数据的模型输出后
    output1_max = output1.max(dim=1, keepdim=True)[0]  # 取最大概率作为单通道
    output2_max = output2.max(dim=1, keepdim=True)[0]
    total_loss = consistency_criterion(output1_max, output2_max)  # 对最终分割结果计算一致性损失

    if deep_supervision:
        # 对每个阶段的输出计算一致性损失（同层之间）
        print("consistency Deep supervision")
        loss_1 = consistency_criterion(output1[1], output2[1])  # 第一层一致性损失
        loss_2 = consistency_criterion(output1[2], output2[2])  # 第二层一致性损失
        loss_3 = consistency_criterion(output1[3], output2[3])  # 第三层一致性损失

        # 总一致性损失加权累加
        total_loss += weight * (loss_1 + loss_2 + loss_3)

    return total_loss


def train_mednext_one_epoch(model, train_loader, optimizer, segmentation_criterion,
                              epoch, total_epoches, num_ite_per_epoch, base_lr, dice_loss_fn = None):
    # Training
    model.train()
    total_iters = num_ite_per_epoch * total_epoches
    batch_time = DataUpdater()
    ave_segmentation_loss = DataUpdater()
    tic = time.time()
    cur_iters = epoch * num_ite_per_epoch
    device = next(model.parameters()).device

    # 初始化 GradScaler
    scaler = GradScaler()

    for i_iter, batch in enumerate(train_loader):
        images, masks, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)

        # 获取模型的输出
        segmentation_output = model(images)


        # 计算分割的损失
        loss_ce = compute_loss(segmentation_criterion, segmentation_output, masks)
        probs = F.softmax(segmentation_output[0], dim=1)
        loss_dice = dice_loss_fn(probs, masks)
        segmentation_loss = loss_ce + loss_dice

        optimizer.zero_grad()
        # 反向传播，并在混合精度上下文中进行梯度缩放
        scaler.scale(segmentation_loss).backward()  # 使用 scaler 进行反向传播

        # 更新梯度并优化
        scaler.step(optimizer)  # 使用 scaler 进行参数更新
        scaler.update()  # 更新 scaler 的缩放因子

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average losses
        ave_segmentation_loss.update(segmentation_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  total_iters,
                                  i_iter + cur_iters)

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {},  Segmentation Loss: {:.6f}, '.format(
                epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                batch_time.avg, [x['lr'] for x in optimizer.param_groups],
                ave_segmentation_loss.avg,)
            print(msg)

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')

    return ave_segmentation_loss.avg

def train_ns_transUnet_one_epoch(model, train_loader, optimizer, segmentation_criterion,
                                 epoch, total_epochs, num_ite_per_epoch, base_lr, labeled_bs, consistency_weight):
    # Training
    model.train()
    total_iters = num_ite_per_epoch * total_epochs
    batch_time = DataUpdater()
    ave_segmentation_loss = DataUpdater()
    ave_consistency_loss = DataUpdater()
    ave_total_loss = DataUpdater()  # 用于记录总损失

    # 定义一致性损失函数
    consistency_criterion = nn.MSELoss()

    tic = time.time()
    cur_iters = epoch * num_ite_per_epoch
    device = next(model.parameters()).device

    for i_iter, batch in enumerate(train_loader):
        images, masks, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)

        # 处理有标签的数据
        labeled_images = images[:labeled_bs]  # 假设前 labeled_bs 是有标签的数据
        labeled_masks = masks[:labeled_bs]  # 对应有标签数据的 ground truth
        labeled_images = labeled_images.to(device, non_blocking=True)
        labeled_masks = labeled_masks.long().to(device, non_blocking=True)

        # 获取有标签数据的模型输出
        labeled_segmentation_output1, labeled_segmentation_output2 = model(labeled_images)

        # 计算分割的损失
        # loss1 = segmentation_criterion(labeled_segmentation_output1, labeled_masks)
        # loss2 = segmentation_criterion(labeled_segmentation_output2, labeled_masks)
        loss1 = compute_deep_supervision_loss(labeled_segmentation_output1, labeled_masks, segmentation_criterion, deep_supervision=False)
        loss2 = compute_deep_supervision_loss(labeled_segmentation_output2, labeled_masks, segmentation_criterion, deep_supervision=False)
        segmentation_loss = (loss1 + loss2) / 2

        # 处理无标签数据
        if labeled_bs < images.size(0):  # 确保有无标签数据时才计算一致性损失
            unlabeled_images = images[labeled_bs:]  # 后 labeled_bs 是无标签数据
            unlabeled_images = unlabeled_images.to(device, non_blocking=True)

            # 获取无标签数据的模型输出（两个输出用于一致性损失）
            unlabeled_output1, unlabeled_output2 = model(unlabeled_images)
            # 在获取无标签数据的模型输出后
            # unlabeled_output1_max = unlabeled_output1.max(dim=1, keepdim=True)[0]  # 取最大概率作为单通道
            # unlabeled_output2_max = unlabeled_output2.max(dim=1, keepdim=True)[0]

            # 计算无标签数据的一致性损失
            #consistency_loss = consistency_criterion(unlabeled_output1_max, unlabeled_output2_max)
            consistency_loss = compute_consistency_loss(unlabeled_output1, unlabeled_output2, consistency_criterion, deep_supervision=False)
        else:
            consistency_loss = torch.tensor(0.0, device=device)  # 没有无标签数据时一致性损失为0

        # 总损失
        total_loss = segmentation_loss + consistency_weight * consistency_loss  # 权重随轮数增加

        # 反向传播与优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 记录时间
        batch_time.update(time.time() - tic)
        tic = time.time()

        # 更新平均损失
        ave_segmentation_loss.update(segmentation_loss.item())
        ave_consistency_loss.update(consistency_loss.item())
        ave_total_loss.update(total_loss.item())  # 更新总损失

        lr = adjust_learning_rate(optimizer, base_lr, total_iters, i_iter + cur_iters)

        if i_iter % 5 == 0:
            msg = ('Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, lr: {}, '
                   'Segmentation Loss: {:.6f}, Consistency Loss: {:.6f}, Total Loss: {:.6f}').format(
                epoch + 1, total_epochs, i_iter, num_ite_per_epoch,
                batch_time.avg, [x['lr'] for x in optimizer.param_groups],
                ave_segmentation_loss.avg, ave_consistency_loss.avg, ave_total_loss.avg)  # 输出总损失
            print(msg)

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')

    return ave_total_loss.avg, ave_segmentation_loss.avg, ave_consistency_loss.avg


def extract_glcm_features(image, distances=[1], angles=[0]):
    # 将RGB图像转换为灰度图像
    image_gray = rgb2gray(image)

    # 确保灰度图像为二维数组
    glcm = graycomatrix(image_gray.astype(np.uint8), distances=distances, angles=angles, symmetric=True, normed=True)

    # 提取纹理特征，确保处理的是每个角度和距离的结果
    contrast = graycoprops(glcm, 'contrast')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    homogeneity = graycoprops(glcm, 'homogeneity')

    # 返回提取的特征
    return contrast, energy, correlation, homogeneity


def process_image_glcm(image):
    # 提取纹理特征，并返回扩展为适合模型输入的形式
    contrast, energy, correlation, homogeneity = extract_glcm_features(image)

    # 处理提取的特征
    texture_features = np.stack([contrast, energy, correlation, homogeneity], axis=0)  # 合并所有特征
    texture_features = torch.tensor(texture_features, dtype=torch.float32)  # 转换为tensor
    return texture_features

def train_transformer_seg_cls_epoch(segmodel, clsmodel, train_loader, cls_optimizer, classification_criterion,
                    epoch, total_epoches, num_ite_per_epoch,  base_cls_lr):

    # 冻结分割模型
    segmodel.eval()
    # Training
    clsmodel.train()

    total_iters = num_ite_per_epoch * total_epoches
    batch_time = DataUpdater()
    ave_classification_loss = DataUpdater()

    tic = time.time()
    cur_iters = epoch * num_ite_per_epoch
    device = next(segmodel.parameters()).device

    # 定义随机裁剪操作
    random_resized_crop = T.RandomResizedCrop(size=(256, 512), scale=(0.8, 1.0), ratio=(1.8, 2.2),
                                              interpolation=T.InterpolationMode.BICUBIC)

    for i_iter, batch in enumerate(train_loader):
        images, texture, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        texture = texture.to(device, non_blocking=True)
        labels = torch.tensor(labels).clone().detach().long().to(device, non_blocking=True)

        with torch.no_grad():
            # 使用分割模型预测mask
            segmentation_output, _, _, _, _ = segmodel(images)
            pred_mask = torch.argmax(segmentation_output, dim=1)  # [batch_size, height, width]
            pred_mask = pred_mask.unsqueeze(1).float()
            #pred_mask * (255.0 / 7.0)   # 归一化到0-255
            #pred_mask = pred_mask.repeat(1, 3, 1, 1) # 扩展通道，变为 [12, 3, 512, 1024]
            texture = torch.stack([random_resized_crop(text) for text in texture])  # 对每个texture进行随机裁剪
            pred_mask = torch.stack([random_resized_crop(mask) for mask in pred_mask])  # 对每个mask进行随机裁剪
            pred_mask = pred_mask.to(device)

        # 训练分类模型
        cls_optimizer.zero_grad()
        # classification_output, _ = clsmodel(pred_mask_3c_resized)
        # 前向传播
        outputs = clsmodel(pred_mask, texture)
        classification_loss = classification_criterion(outputs, labels)
        classification_loss.backward()
        cls_optimizer.step()

        # 更新损失
        ave_classification_loss.update(classification_loss.item())

        lr = adjust_learning_rate(cls_optimizer, base_cls_lr, total_iters, i_iter + cur_iters)

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, Cls LR: {}, Classification Loss: {:.6f}'.format(
                epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                batch_time.avg, [x['lr'] for x in cls_optimizer.param_groups],
                ave_classification_loss.avg)
            print(msg)

        # 计算时间
        batch_time.update(time.time() - tic)
        tic = time.time()

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')


    return ave_classification_loss.avg

def train_ST_cls_epoch(segmodel, clsmodel, train_loader, optimizer, classification_criterion,
                    epoch, total_epoches, num_ite_per_epoch, lr_schedule_values,
                    wd_schedule_values, base_cls_lr):

    # 冻结分割模型
    segmodel.eval()
    # Training
    clsmodel.train()

    batch_time = DataUpdater()
    ave_classification_loss = DataUpdater()

    tic = time.time()
    device = next(segmodel.parameters()).device

    # 定义随机裁剪操作
    random_resized_crop = T.RandomResizedCrop(size=(256, 512), scale=(0.8, 1.0), ratio=(1.8, 2.2),
                                              interpolation=T.InterpolationMode.BICUBIC)

    for i_iter, batch in enumerate(train_loader):
        images, texture, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        texture = texture.to(device, non_blocking=True)
        labels = torch.tensor(labels).clone().detach().long().to(device, non_blocking=True)

        with torch.no_grad():
            # 使用分割模型预测mask
            segmentation_output, _, _, _, _ = segmodel(images)
            pred_mask = torch.argmax(segmentation_output, dim=1)  # [batch_size, height, width]
            pred_mask = pred_mask.unsqueeze(1).float()
            pred_mask = pred_mask.repeat(1, 3, 1, 1) # 扩展通道，变为 [12, 3, 512, 1024]
            #texture = torch.stack([random_resized_crop(text) for text in texture])  # 对每个texture进行随机裁剪
            #pred_mask = torch.stack([random_resized_crop(mask) for mask in pred_mask])  # 对每个mask进行随机裁剪
            pred_mask = pred_mask.to(device)

        # 训练分类模型
        optimizer.zero_grad()

        # 前向传播
        outputs = clsmodel(pred_mask)
        classification_loss = classification_criterion(outputs, labels)
        classification_loss.backward()
        optimizer.step()

        # 更新损失
        ave_classification_loss.update(classification_loss.item())

        # 根据cosine scheduler 修改optimizer的learning rate 和 weight decay
        global_step = epoch * num_ite_per_epoch + i_iter
        for _, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[global_step]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[global_step]

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, Cls LR: {}, Classification Loss: {:.6f}'.format(
                epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                batch_time.avg, [x['lr'] for x in optimizer.param_groups],
                ave_classification_loss.avg)
            print(msg)

        # 计算时间
        batch_time.update(time.time() - tic)
        tic = time.time()

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')


    return ave_classification_loss.avg


def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)  # 确保是概率
    target = torch.sigmoid(target)  # 确保是概率
    intersection = (pred * target).sum(dim=(2,3))  # 计算交集
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))  # 计算并集
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()  # 取负，因为Dice越大表示相似度越高

def train_rec_cls_epoch(segmodel, clsmodel, drae_encoder, fusion_model, train_loader, optimizer, fusion_optimizer, criterion,
                    epoch, total_epoches, num_ite_per_epoch, lr_schedule_values,
                    wd_schedule_values):
    # 冻结分割模型
    segmodel.eval()
    # Training
    clsmodel.train()
    fusion_model.train()
    for name, param in fusion_model.named_parameters():
        print(f"{name} requires_grad: {param.requires_grad}")

    batch_time = DataUpdater()
    ave_classification_loss = DataUpdater()

    device = next(clsmodel.parameters()).device


    tic = time.time()

    for i_iter, batch in enumerate(train_loader):
        images, _, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        #texture = texture.to(device, non_blocking=True)
        labels = torch.tensor(labels).clone().detach().long().to(device, non_blocking=True)

        # **避免额外梯度计算**
        with torch.no_grad():
            seg_output, _, _, _, _ = segmodel(images)
            #print(f"Image shape: {images.shape}")  # 输出图像形状
            # 转为灰度图
            gray_image = images[:, 0:1, :, :]  # 取 B 通道
            # DRAE 编码器的纹理特征
            drae_features = drae_encoder(gray_image)  # [batch_size, latent_dim]
            
            
        # **特征融合**
        fused_features = fusion_model(drae_features, seg_output)
        # print("drae_features:", drae_features.shape)
        # print("drae_features:", drae_features)
        # print("fused_features:", fused_features.shape)
        # print("fused_features:", fused_features)
        print("Fused features mean:", fused_features.mean().item())

            # 训练分类模型 
        optimizer.zero_grad()
        fusion_optimizer.zero_grad()
        outputs = clsmodel(fused_features)
        classification_loss = criterion(outputs, labels)
        classification_loss.backward()
         # 检查梯度
        for name, param in fusion_model.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
            else:
                print(f"No gradient for {name}")
        optimizer.step()
        fusion_optimizer.step()

        # 更新损失
        ave_classification_loss.update(classification_loss.item())

        #ave_rec_seg_loss.update(rec_seg_loss.item())
        # 根据cosine scheduler 修改 optimizer 的 learning rate 和 weight decay
        global_step = epoch * num_ite_per_epoch + i_iter
        for _, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[global_step]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[global_step]

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, Cls LR: {}, Cls Loss: {:.6f} '.format(
                epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                batch_time.avg, [x['lr'] for x in optimizer.param_groups],
                ave_classification_loss.avg)
            print(msg)

        # 计算时间
        batch_time.update(time.time() - tic)
        tic = time.time()

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')

    return ave_classification_loss.avg

def discriminative_loss(z, labels):
    """Fisher 判别损失（稳定版）"""
    z_neg = z[labels == 0]
    z_pos = z[labels == 1]

    # 如果某一类不存在，跳过判别损失
    if z_neg.shape[0] < 2 or z_pos.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)

    mu_neg = torch.mean(z_neg, dim=0)
    sigma_neg = torch.var(z_neg, dim=0, unbiased=False)

    mu_pos = torch.mean(z_pos, dim=0)
    sigma_pos = torch.var(z_pos, dim=0, unbiased=False)

    inter_class_variance = torch.norm(mu_neg - mu_pos) ** 2
    intra_class_variance = sigma_neg.mean() + sigma_pos.mean()

    loss = intra_class_variance / (inter_class_variance + 1e-6)

    return loss



def positive_discriminative_loss(z, labels):
    """
    判别损失：仅对阳性类别（label 3 和 4）之间进行类间区分
    z: 特征向量 (B, D)
    labels: 对应标签 (B,)
    """
    # 只提取阳性样本（label == 3 或 4）
    mask_pos = (labels == 3) | (labels == 4)
    z_pos = z[mask_pos]
    pos_labels = labels[mask_pos]

    # 如果阳性样本不足两个类别，跳过
    if (pos_labels == 3).sum() == 0 or (pos_labels == 4).sum() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    z_3 = z_pos[pos_labels == 3]
    z_4 = z_pos[pos_labels == 4]

    mu_3 = torch.mean(z_3, dim=0)
    mu_4 = torch.mean(z_4, dim=0)

    sigma_3 = torch.var(z_3, dim=0)
    sigma_4 = torch.var(z_4, dim=0)

    inter_class_variance = torch.norm(mu_3 - mu_4) ** 2
    intra_class_variance = sigma_3.mean() + sigma_4.mean()

    loss = intra_class_variance / (inter_class_variance + 1e-6)
    return loss

def show_seg_pred(pred_mask):
    label_colors = np.array([
        [0, 0, 0],  # 类别0，背景
        [128, 0, 0],  # 类别1
        [0, 128, 0],  # 类别2
        [128, 128, 0],  # 类别3
        [0, 0, 128],  # 类别4
        [128, 0, 128],  # 类别5
        [0, 128, 128],  # 类别6
        [128, 128, 128],  # 类别7
    ])
    # 彩色化预测结果
    pred_mask_for_save = pred_mask.squeeze(1).cpu().numpy()  # [B, H, W]


    for i in range(pred_mask_for_save.shape[0]):
        color_mask = label_colors[pred_mask_for_save[i]]  # 使用颜色映射将类别索引转换为RGB
        color_mask = np.asarray(color_mask, dtype=np.uint8)  # 转换为uint8类型数组

        # 使用matplotlib显示结果
        plt.figure(figsize=(6, 6))
        plt.imshow(color_mask)
        plt.title(f"Predicted Mask for knowsam Image {i}")
        plt.axis('off')  # 去掉坐标轴
        plt.show()




def save_segmentation_vis(
    gray_image,      # [1, H, W]
    seg_logits,      # [C, H, W]
    save_path
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    gray = gray_image.squeeze().cpu().numpy()
    pred = torch.argmax(seg_logits, dim=0).cpu().numpy()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original OCT")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Segmentation")
    plt.imshow(pred, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(gray, cmap="gray")
    plt.imshow(pred, cmap="jet", alpha=0.4)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


EYE_CLASS_NAMES = {
    0: "NORMAL",
    1: "AMD",
    2: "DME"
}

CEV_CLASS_NAMES = {
    0:'宫颈炎',
    1:'囊肿',
    2:'外翻' ,
    3:'高级别病变',
    4:'宫颈癌'
}

def train_seg_drae_epoch(segmodel, clsmodel, train_loader, optimizer, criterion, recon_criterion,
                         epoch, total_epoches, num_ite_per_epoch, lr_schedule_values,
                         wd_schedule_values):
    # 冻结分割模型
    segmodel.eval()
    # Training
    clsmodel.train()

    batch_time = DataUpdater()
    ave_classification_loss = DataUpdater()
    ave_reconstruction_loss = DataUpdater()
    ave_discriminative_loss = DataUpdater()
    ave_total_loss = DataUpdater()

    device = next(clsmodel.parameters()).device
    tic = time.time()

    for i_iter, batch in enumerate(train_loader):
        images, _, labels, img_paths, _ = batch
        images = images.to(device, non_blocking=True)
        labels = torch.tensor(labels).clone().detach().long().to(device, non_blocking=True)

        # **避免额外梯度计算**
        with torch.no_grad():
            seg_output, _, _, _, _ = segmodel(images)
            gray_image = images[:, 0:1, :, :]

            # ================== 保存 segmentation 可视化 ==================
            # 每个 epoch 都存，但限制每类最多保存 K 张
            # 每个 epoch 单独统计一次
            saved_counter = {}
            MAX_SAVE_PER_CLASS = 10

            # ================== 保存 segmentation 可视化 ==================
            if i_iter % 3 == 0:
                save_root = f"./debug_seg_vis_cervical/epoch_{epoch + 1:02d}"

                for b in range(images.size(0)):
                    cls = int(labels[b].item())

                    if cls not in saved_counter:
                        saved_counter[cls] = 0
                    if saved_counter[cls] >= MAX_SAVE_PER_CLASS:
                        continue

                    class_name = CEV_CLASS_NAMES.get(cls, f"class_{cls}")

                    save_dir = os.path.join(
                        save_root,
                        f"class_{cls}_{class_name}"
                    )
                    os.makedirs(save_dir, exist_ok=True)

                    save_name = f"iter{i_iter}_b{b}.png"
                    save_path = os.path.join(save_dir, save_name)

                    # === 保存 segmentation 可视化 ===
                    save_segmentation_vis(
                        gray_image[b],
                        seg_output[b],
                        save_path
                    )

                    # === 新增：保存原图路径信息 ===
                    meta_path = os.path.join(save_dir, "meta.txt")
                    with open(meta_path, "a") as f:
                        f.write(
                            f"{save_name} | label={cls} | img_path={img_paths[b]}\n"
                        )

                    saved_counter[cls] += 1

            pred_mask = torch.argmax(seg_output, dim=1)  # [batch_size, height, width]
            #show_seg_pred(pred_mask)
            pred_mask = pred_mask.unsqueeze(1).float()
            seg_image = pred_mask.repeat(1, 3, 1, 1) # 扩展通道，变为 [12, 3, 512, 1024]

            # 将类别 1 和 5 的像素值设置为 0

            pred_mask[(pred_mask == 1) | (pred_mask == 5)] = 0
            pred_mask[(pred_mask == 2) | (pred_mask == 3) | (pred_mask == 4) | (pred_mask == 6) | (pred_mask == 7)] = 1
            extracted_image = gray_image * pred_mask
            # 将类别 5 的像素值设置为 0
            # 眼科OCT数据
            # pred_mask[(pred_mask == 5)] = 0
            # pred_mask[(pred_mask == 1) | (pred_mask == 2) | (pred_mask == 3) | (pred_mask == 4) ] = 1
            # extracted_image = gray_image * pred_mask
            #extracted_image = gray_image

        # 训练分类模型
        optimizer.zero_grad()

        # 前向传播
        outputs, z, recon = clsmodel(extracted_image, seg_image)

        # 分类损失
        classification_loss = criterion(outputs, labels)

        # 重构损失
        if recon is not None:
            reconstruction_loss = recon_criterion(recon, gray_image)

        disc_loss = discriminative_loss(z,labels)

        # 总损失（加权组合）
        #total_loss = classification_loss + 0.5 * reconstruction_loss  # 宫颈OCT的权重
        #total_loss = classification_loss + 0.1 * reconstruction_loss + 0.1 * disc_loss # 眼科OCT的权重
        total_loss = classification_loss +  0.1 * disc_loss # 眼科OCT的权重，特征金字塔没有重构损失
        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 更新损失
        ave_classification_loss.update(classification_loss.item())
        #ave_reconstruction_loss.update(reconstruction_loss.item())
        ave_discriminative_loss.update(disc_loss.item())
        ave_total_loss.update(total_loss.item())

        # 根据 cosine scheduler 修改 optimizer 的 learning rate 和 weight decay
        global_step = epoch * num_ite_per_epoch + i_iter
        for _, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[global_step]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[global_step]

        if i_iter % 5 == 0:
            msg = ('Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, Cls LR: {}, '
                   'Cls Loss: {:.6f},Disc Loss: {:.6f}, Total Loss: {:.6f}').format(
                epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                batch_time.avg, [x['lr'] for x in optimizer.param_groups],
                ave_classification_loss.avg,ave_discriminative_loss.avg, ave_total_loss.avg)
            print(msg)

        # 计算时间
        batch_time.update(time.time() - tic)
        tic = time.time()

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')

    return ave_classification_loss.avg, ave_reconstruction_loss.avg, ave_total_loss.avg





def train_cls_epoch( clsmodel, train_loader, optimizer, criterion,
                    epoch, total_epoches, num_ite_per_epoch, lr_schedule_values,
                    wd_schedule_values, base_cls_lr):

    # Training
    clsmodel.train()

    batch_time = DataUpdater()
    ave_classification_loss = DataUpdater()
    device = next(clsmodel.parameters()).device

    tic = time.time()

    for i_iter, batch in enumerate(train_loader):
        images, texture, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        texture = texture.to(device, non_blocking=True)
        labels = torch.tensor(labels).clone().detach().long().to(device, non_blocking=True)

        # 训练分类模型
        optimizer.zero_grad()

        # 前向传播
        outputs = clsmodel(images)
        classification_loss = criterion(outputs, labels)
        classification_loss.backward()
        optimizer.step()

        # 更新损失
        ave_classification_loss.update(classification_loss.item())

        # 根据cosine scheduler 修改optimizer的learning rate 和 weight decay
        global_step = epoch * num_ite_per_epoch + i_iter
        for _, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[global_step]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[global_step]

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, Cls LR: {}, Classification Loss: {:.6f}'.format(
                epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                batch_time.avg, [x['lr'] for x in optimizer.param_groups],
                ave_classification_loss.avg)
            print(msg)

        # 计算时间
        batch_time.update(time.time() - tic)
        tic = time.time()

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')

    return ave_classification_loss.avg


    #分割分类分阶段训练
def train_transUnet_seg_cls_epoch(segmodel, clsmodel, train_loader, seg_optimizer, cls_optimizer, segmentation_criterion, classification_criterion,
                    epoch, total_epoches, num_ite_per_epoch, base_seg_lr, base_cls_lr, phase='segmentation'):
    # Training
    segmodel.train()
    clsmodel.train()

    total_iters = num_ite_per_epoch * total_epoches
    batch_time = DataUpdater()
    ave_total_loss = DataUpdater()
    ave_segmentation_loss = DataUpdater()
    ave_classification_loss = DataUpdater()
    tic = time.time()
    cur_iters = epoch * num_ite_per_epoch
    device = next(segmodel.parameters()).device

    for i_iter, batch in enumerate(train_loader):
        images, masks, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)
        labels = torch.tensor(labels).clone().detach().long().to(device, non_blocking=True)

        if phase == 'segmentation':  # 第一阶段：只训练分割模型
            seg_optimizer.zero_grad()
            segmentation_output, _ = segmodel(images)
            segmentation_loss = segmentation_criterion(segmentation_output, masks)
            segmentation_loss.backward()
            seg_optimizer.step()

            # 更新损失
            ave_segmentation_loss.update(segmentation_loss.item())

            if i_iter % 5 == 0:
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, Seg LR: {}, Segmentation Loss: {:.6f}'.format(
                    epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                    batch_time.avg, [x['lr'] for x in seg_optimizer.param_groups],
                    ave_segmentation_loss.avg)
                print(msg)

        elif phase == 'classification':  # 第二阶段：冻结分割模型，训练分类模型
            # 冻结分割模型
            segmodel.eval()
            with torch.no_grad():
                segmentation_output, _ = segmodel(images)
                # segmentation_output shape: (batch_size, 8, height, width)
                pred_mask = torch.argmax(segmentation_output, dim=1)    # 只使用分割模型的输出mask作为输入
                #pred_mask数据处理，扩展成3通道
                pred_mask = pred_mask.unsqueeze(1).float()  # [1, H, W]，确保类型为 float
                pred_mask_3c = pred_mask.repeat(1, 3, 1, 1)  # 将单通道扩展为三通道，形状为 [3, H, W]
            # 训练分类模型
            cls_optimizer.zero_grad()
            classification_output = clsmodel(pred_mask_3c)
            classification_loss = classification_criterion(classification_output, labels)
            classification_loss.backward()
            cls_optimizer.step()

            # 更新损失
            ave_classification_loss.update(classification_loss.item())

            if i_iter % 5 == 0:
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, Cls LR: {}, Classification Loss: {:.6f}'.format(
                    epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                    batch_time.avg, [x['lr'] for x in cls_optimizer.param_groups],
                    ave_classification_loss.avg)
                print(msg)

        # 计算时间
        batch_time.update(time.time() - tic)
        tic = time.time()

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')

    if phase == 'segmentation':
        return ave_segmentation_loss.avg
    else:
        return ave_classification_loss.avg



def train_transUnet_seg_cls_epoch2(segmodel, clsmodel, train_loader, seg_optimizer, cls_optimizer, segmentation_criterion, classification_criterion,
                    epoch, total_epoches, num_ite_per_epoch, base_seg_lr, base_cls_lr):
    # Training
    segmodel.train()
    clsmodel.train()
    total_iters = num_ite_per_epoch * total_epoches
    batch_time = DataUpdater()
    ave_total_loss = DataUpdater()
    ave_segmentation_loss = DataUpdater()
    ave_classification_loss = DataUpdater()
    tic = time.time()
    cur_iters = epoch * num_ite_per_epoch
    device = next(segmodel.parameters()).device

    for i_iter, batch in enumerate(train_loader):
        images, masks, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)
        labels = torch.tensor(labels).clone().detach().long().to(device, non_blocking=True)

        # 1. 训练分割模型
        seg_optimizer.zero_grad()
        segmentation_output, features = segmodel(images)
        segmentation_loss = segmentation_criterion(segmentation_output, masks)
        segmentation_loss.backward()
        seg_optimizer.step()

        # 2. 冻结分割模型的参数以避免分类训练时修改
        for param in segmodel.parameters():
            param.requires_grad = False

        # 3. 用分割模型的输出作为分类模型的输入
        classification_output = clsmodel(features.detach())  # 使用detach避免梯度传播到分割模型
        cls_optimizer.zero_grad()
        classification_loss = classification_criterion(classification_output, labels)
        classification_loss.backward()
        cls_optimizer.step()

        # 解冻分割模型以便下一次迭代继续训练
        for param in segmodel.parameters():
            param.requires_grad = True

        # 计算耗时
        batch_time.update(time.time() - tic)
        tic = time.time()

        # 更新平均损失
        ave_total_loss.update(segmentation_loss.item() + classification_loss.item())
        ave_segmentation_loss.update(segmentation_loss.item())
        ave_classification_loss.update(classification_loss.item())

        # 调整学习率
        lr_seg = adjust_learning_rate(seg_optimizer, base_seg_lr, total_iters, i_iter + cur_iters)
        lr_cls = adjust_learning_rate(cls_optimizer, base_cls_lr, total_iters, i_iter + cur_iters)

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'Seg LR: {}, Cls LR: {}, Total Loss: {:.6f}, Segmentation Loss: {:.6f}, Classification Loss: {:.6f}'.format(
                epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                batch_time.avg, [x['lr'] for x in seg_optimizer.param_groups], [x['lr'] for x in cls_optimizer.param_groups],
                ave_total_loss.avg, ave_segmentation_loss.avg, ave_classification_loss.avg)
            print(msg)

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')

    return ave_total_loss.avg, ave_segmentation_loss.avg, ave_classification_loss.avg


def train_transUnet_seg_cls_epoch1(segmodel, clsmodel, train_loader, seg_optimizer, cls_optimizer, segmentation_criterion, classification_criterion,
                    epoch, total_epoches, num_ite_per_epoch, base_seg_lr, base_cls_lr):
    # Training
    segmodel.train()
    clsmodel.train()
    total_iters = num_ite_per_epoch * total_epoches
    batch_time = DataUpdater()
    ave_total_loss = DataUpdater()
    ave_segmentation_loss = DataUpdater()
    ave_classification_loss = DataUpdater()
    tic = time.time()
    cur_iters = epoch * num_ite_per_epoch
    device = next(segmodel.parameters()).device

    for i_iter, batch in enumerate(train_loader):
        images, masks, labels, _, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)
        labels = torch.tensor(labels).clone().detach().long().to(device, non_blocking=True)

        # 获取分割模型的输出
        segmentation_output = segmodel(images)
        # segmentation_output shape: (batch_size, 8, height, width)
        pred_mask = torch.argmax(segmentation_output, dim=1)  # 只使用分割模型的输出mask作为输入
        # pred_mask数据处理，扩展成3通道
        pred_mask = pred_mask.unsqueeze(1).float()  # [1, H, W]，确保类型为 float
        pred_mask_3c = pred_mask.repeat(1, 3, 1, 1)  # 将单通道扩展为三通道，形状为 [3, H, W]

        # 获取分类模型的输出
        classification_output = clsmodel(pred_mask_3c)

        # 分别计算两个输出的损失
        segmentation_loss = segmentation_criterion(segmentation_output, masks)

        classification_loss = classification_criterion(classification_output, labels)

        # 总损失是两个损失的加权和，可以根据需要调整权重
        total_loss = segmentation_loss + 0.1*classification_loss

        # 清空梯度
        seg_optimizer.zero_grad()
        cls_optimizer.zero_grad()

        # 反向传播
        total_loss.backward()

        # 分别更新优化器
        seg_optimizer.step()
        cls_optimizer.step()

        # 计算耗时
        batch_time.update(time.time() - tic)
        tic = time.time()

        # 更新平均损失
        ave_total_loss.update(total_loss.item())
        ave_segmentation_loss.update(segmentation_loss.item())
        ave_classification_loss.update(classification_loss.item())

        # 分别调整学习率
        lr_seg = adjust_learning_rate(seg_optimizer, base_seg_lr, total_iters, i_iter + cur_iters)
        lr_cls = adjust_learning_rate(cls_optimizer, base_cls_lr, total_iters, i_iter + cur_iters)

        if i_iter % 5 == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'Seg LR: {}, Cls LR: {}, Total Loss: {:.6f}, Segmentation Loss: {:.6f}, Classification Loss: {:.6f}'.format(
                epoch + 1, total_epoches, i_iter, num_ite_per_epoch,
                batch_time.avg, [x['lr'] for x in seg_optimizer.param_groups], [x['lr'] for x in cls_optimizer.param_groups],
                ave_total_loss.avg, ave_segmentation_loss.avg, ave_classification_loss.avg)
            print(msg)

    print(f'The {epoch + 1} epoch training time {batch_time.sum:.3f}\n')

    return ave_total_loss.avg, ave_segmentation_loss.avg, ave_classification_loss.avg