import torch
from smoothnet.lib.utils.eval_metrics import *
from smoothnet.lib.utils.geometry_utils import *
from smoothnet.lib.utils.utils import slide_window_to_sequence
import os
import cv2
from smoothnet.lib.visualize.visualize_3d import visualize_3d
from smoothnet.lib.visualize.visualize_2d import visualize_2d
import sys
from tqdm import tqdm
import numpy as np


class Visualize():

    def __init__(self, cfg):

        self.cfg = cfg
        self.device = cfg.DEVICE

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.estimator = self.cfg.ESTIMATOR
        self.dataset_name = self.cfg.DATASET_NAME
        self.body_representation = self.cfg.BODY_REPRESENTATION

        self.slide_window_size = self.cfg.MODEL.SLIDE_WINDOW_SIZE
        self.slide_window_step = self.cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE

        self.base_data_path = self.cfg.DATASET.BASE_DIR

        self.device = self.cfg.DEVICE
        self.input_dimension = 32

    def pose_inference_2d(self, model, data_pred, img_shape):
        keypoint_number = self.input_dimension // 2

        data_imageshape = img_shape         # array(2, )
        data_gt = torch.tensor(data_pred).to(self.device)

        data_pred_norm = torch.tensor(data_pred.reshape(-1, 2) / data_imageshape - 0.5).to(self.device).reshape_as(data_gt)
        data_pred = torch.tensor(data_pred).to(self.device)

        data_len = data_pred.shape[0]

        data_pred_window = torch.as_strided(
            data_pred_norm, ((data_len - self.slide_window_size) // self.slide_window_step + 1, self.slide_window_size, keypoint_number, 2),
            (self.slide_window_step * keypoint_number * 2, keypoint_number * 2, 2, 1), storage_offset=0).reshape(-1, self.slide_window_size, self.input_dimension)

        with torch.no_grad():
            data_pred_window = data_pred_window.permute(0, 2, 1)
            predicted_pos = model(data_pred_window)

            data_pred_window = data_pred_window.permute(0, 2, 1)
            predicted_pos = predicted_pos.permute(0, 2, 1)

        predicted_pos = slide_window_to_sequence(predicted_pos, self.slide_window_step, self.slide_window_size).reshape(-1, keypoint_number, 2)

        data_len = predicted_pos.shape[0]
        data_pred = data_pred[:data_len, :].reshape(-1, keypoint_number, 2)

        predicted_pos = (predicted_pos.reshape(-1, keypoint_number, 2) + 0.5) * torch.tensor(data_imageshape).to(predicted_pos.device)
        data_pred = np.array(data_pred.cpu())
        predicted_pos = np.array(predicted_pos.cpu())
        return predicted_pos

    def visualize(self, model, un_smooth_data, img_shape):
        model.eval()
        return self.pose_inference_2d(model, data_pred=un_smooth_data, img_shape=img_shape)
