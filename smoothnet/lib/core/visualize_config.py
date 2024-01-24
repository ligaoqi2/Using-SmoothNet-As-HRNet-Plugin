import argparse
from yacs.config import CfgNode as CN

# CONSTANTS
# You may modify them at will
BASE_DATA_DIR = 'data/poses'  # data dir

# Configuration variables
cfg = CN()
cfg.DEVICE = 'cuda'  # training device 'cuda' | 'cpu'
cfg.SEED_VALUE = 4321  # random seed
cfg.LOGDIR = ''  # log dir
cfg.EXP_NAME = 'default'  # experiment name
cfg.DEBUG = True  # debug
cfg.OUTPUT_DIR = 'results'  # output folder

cfg.DATASET_NAME = 'h36m'  # dataset name
cfg.ESTIMATOR = 'hrnet'  # backbone estimator name
cfg.BODY_REPRESENTATION = '2D'  # 3D | 2D | smpl

# CUDNN config
cfg.CUDNN = CN()  # cudnn config
cfg.CUDNN.BENCHMARK = True  # cudnn config
cfg.CUDNN.DETERMINISTIC = False  # cudnn config
cfg.CUDNN.ENABLED = True  # cudnn config

# dataset config
cfg.DATASET = CN()
cfg.DATASET.BASE_DIR = BASE_DATA_DIR
cfg.DATASET.ROOT_H36M_FCN_3D = [0]
cfg.DATASET.ROOT_H36M_RLE_3D = [0]
cfg.DATASET.ROOT_H36M_TCMR_3D = [2, 3]
cfg.DATASET.ROOT_H36M_VIBE_3D = [2, 3]
cfg.DATASET.ROOT_H36M_VIDEOPOSET27_3D = [0]
cfg.DATASET.ROOT_H36M_VIDEOPOSET81_3D = [0]
cfg.DATASET.ROOT_H36M_VIDEOPOSET243_3D = [0]
cfg.DATASET.ROOT_H36M_MIX_3D = [0]

# model config
cfg.MODEL = CN()
cfg.MODEL.SLIDE_WINDOW_SIZE = 32  # slide window size

cfg.MODEL.HIDDEN_SIZE = 512  # hidden size
cfg.MODEL.RES_HIDDEN_SIZE = 256  # res hidden size
cfg.MODEL.NUM_BLOCK = 3  # block number
cfg.MODEL.DROPOUT = 0.5  # dropout

# training config
cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE = 1024  # batch size
cfg.TRAIN.WORKERS_NUM = 0  # workers number
cfg.TRAIN.EPOCH = 70  # epoch number
cfg.TRAIN.LR = 0.001  # learning rate
cfg.TRAIN.LRDECAY = 0.95  # learning rate decay rate
cfg.TRAIN.RESUME = None  # resume training checkpoint path
cfg.TRAIN.VALIDATE = True  # validate while training

# test config
cfg.EVALUATE = CN()
cfg.EVALUATE.PRETRAINED = ''  # evaluation checkpoint
cfg.EVALUATE.ROOT_RELATIVE = True  # root relative represntation in error caculation
cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE = 1  # slide window step size
cfg.EVALUATE.TRADITION = ''  # traditional filter for comparison
cfg.EVALUATE.TRADITION_SAVGOL = CN()
cfg.EVALUATE.TRADITION_SAVGOL.WINDOW_SIZE = 31
cfg.EVALUATE.TRADITION_SAVGOL.POLYORDER = 2
cfg.EVALUATE.TRADITION_GAUS1D = CN()
cfg.EVALUATE.TRADITION_GAUS1D.WINDOW_SIZE = 31
cfg.EVALUATE.TRADITION_GAUS1D.SIGMA = 3
cfg.EVALUATE.TRADITION_ONEEURO = CN()
cfg.EVALUATE.TRADITION_ONEEURO.MIN_CUTOFF = 0.04
cfg.EVALUATE.TRADITION_ONEEURO.BETA = 0.7

# loss config
cfg.LOSS = CN()
cfg.LOSS.W_ACCEL = 1.0  # loss w accel
cfg.LOSS.W_POS = 1.0  # loss w position
# cfg.LOSS.W_VEL = 0.1  # loss w vel

# log config
cfg.LOG = CN()
cfg.LOG.NAME = ''  # log name

# visualization config
cfg.VIS = CN()
cfg.VIS.START = 0
cfg.VIS.END = 1000


def get_cfg_defaults():
    """Get yacs CfgNode object with default values"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def config_smoothnet_parameters():
    cfg_file = "../smoothnet/configs/h36m_fcn_3D.yaml"
    cfg = update_cfg(cfg_file)
    cfg.DATASET_NAME = "h36m"
    cfg.ESTIMATOR = "hrnet"
    cfg.BODY_REPRESENTATION = "2D"
    cfg.MODEL.SLIDE_WINDOW_SIZE = 32
    cfg.EVALUATE.PRETRAINED = "../smoothnet/data/checkpoints/h36m_fcn_3D/checkpoint_32.pth.tar"
    return cfg, cfg_file
