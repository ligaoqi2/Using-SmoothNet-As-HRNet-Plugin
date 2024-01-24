import os
import torch
from smoothnet.lib.models.smoothnet import SmoothNet
from smoothnet.lib.core.visualize_config import config_smoothnet_parameters
from smoothnet.lib.visualize.visualize import Visualize


def SmoothNetPredict(Un_Smooth_data, img_shape):
    cfg, cfg_file = config_smoothnet_parameters()

    model = SmoothNet(window_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                      output_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                      hidden_size=cfg.MODEL.HIDDEN_SIZE,
                      res_hidden_size=cfg.MODEL.RES_HIDDEN_SIZE,
                      num_blocks=cfg.MODEL.NUM_BLOCK,
                      dropout=cfg.MODEL.DROPOUT).to(cfg.DEVICE)

    visualizer = Visualize(cfg)

    if cfg.EVALUATE.PRETRAINED != '' and os.path.isfile(cfg.EVALUATE.PRETRAINED):
        checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'==> Loaded pretrained model from {cfg.EVALUATE.PRETRAINED}...')
    else:
        print(f'{cfg.EVALUATE.PRETRAINED} is not a pretrained model!!!!')
        exit()

    smoothnet_result = visualizer.visualize(model, Un_Smooth_data, img_shape)
    return smoothnet_result
