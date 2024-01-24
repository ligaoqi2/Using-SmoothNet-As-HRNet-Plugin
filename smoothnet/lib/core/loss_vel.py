import torch
import torch.nn as nn
import torch.nn.functional as F
from smoothnet.lib.utils.geometry_utils import *


class SmoothNetLoss(nn.Module):

    def __init__(self, w_accel, w_vel, w_pos):
        super().__init__()
        self.w_accel = w_accel                                                              # weights of accel_loss
        self.w_pos = w_pos                                                                  # weights of pos_loss
        self.w_vel = w_vel

    def mask_lr1_loss(self, inputs, mask, targets):
        Bs, C, L = inputs.shape

        not_mask = 1 - mask.int()
        not_mask = not_mask.unsqueeze(1).repeat(1, C, 1).float()

        N = not_mask.sum(dtype=torch.float32)
        loss = F.l1_loss(
            inputs * not_mask, targets * not_mask, reduction='sum') / N
        return loss

    def forward(self, denoise, gt):
        denoise = denoise.permute(0, 2, 1)
        gt = gt.permute(0, 2, 1)

        loss_pos = F.l1_loss(denoise, gt, reduction='mean')

        vel_gt = gt[:, :, :2] - gt[:, :, 1:-1]
        vel_denoise = denoise[:, :, :2] - denoise[:, :, 1:-1]

        accel_gt = gt[:, :, :-2] - 2 * gt[:, :, 1:-1] + gt[:, :, 2:]
        accel_denoise = denoise[:, :, :-2] - 2 * denoise[:, :, 1:-1] + denoise[:, :, 2:]

        loss_vel = F.l1_loss(vel_denoise, vel_gt, reduction='mean')
        loss_accel = F.l1_loss(accel_denoise, accel_gt, reduction='mean')

        weighted_loss = self.w_accel * loss_accel + self.w_pos * loss_pos + self.w_vel * loss_vel

        return weighted_loss
