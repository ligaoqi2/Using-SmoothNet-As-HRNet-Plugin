from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Basic
import argparse
import csv
import os
import shutil
import cv2
from PIL import Image
import numpy as np
import time
import json
from scipy import signal
from tqdm import tqdm

# torch
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

# HRNet
import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform
from utils.video_preprocess import video_preprocessing

# Prepare SmoothNet
from tools.coco_h36m import coco_h36m, coco_h36m_smoothNet

# SmoothNet
from smoothnet.visualize_smoothnet import SmoothNetPredict

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
    17: 'pelvis'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

SKELETON = [[1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]

SKELETON_h36m = [[10, 9], [9, 8], [8, 11], [8, 14], [8, 7], [11, 12], [12, 13], [14, 15], [15, 16], [7, 0], [0, 1], [0, 4], [4, 5], [5, 6], [1, 2], [2, 3]]

SKELETON_h36m_smoothNet = [[9, 8], [8, 10], [8, 13], [8, 7], [13, 14], [14, 15], [10, 11], [11, 12], [7, 0], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

# cuda
CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def draw_pose(keypoints, img):
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        if kpt_a == 17:
            x_a, y_a = (keypoints[kpt_a - 5][0] + keypoints[kpt_a - 6][0]) / 2, (keypoints[kpt_a - 5][1] + keypoints[kpt_a - 6][1]) / 2
            x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        elif kpt_b == 17:
            x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
            x_b, y_b = (keypoints[kpt_b - 5][0] + keypoints[kpt_b - 6][0]) / 2, (keypoints[kpt_b - 5][1] + keypoints[kpt_b - 6][1]) / 2
        else:
            x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
            x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)


def draw_pose_h36m(keypoints, img):
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON_h36m)):
        kpt_a, kpt_b = SKELETON_h36m[i][0], SKELETON_h36m[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)


def draw_pose_h36m_smoothNet(keypoints, img):
    assert keypoints.shape == (NUM_KPTS - 1, 2)
    for i in range(len(SKELETON_h36m_smoothNet)):
        kpt_a, kpt_b = SKELETON_h36m_smoothNet[i][0], SKELETON_h36m_smoothNet[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)


def draw_smoothNet(smoothNet_result, vidcap, fourcc, smoothnet_video_path):
    count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(smoothnet_video_path, fourcc, 30.0, (int(vidcap.get(3)), int(vidcap.get(4))))
    for frame in tqdm(range(count), desc="Drawing: "):
        ret, image_bgr = vidcap.read()
        if ret:
            draw_pose_h36m_smoothNet(smoothNet_result[frame], image_bgr)
            out.write(image_bgr)

            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
        else:
            print('cannot load the video.')
            break
    vidcap.release()
    out.release()


def draw_bbox(box, img):
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_classes = pred_classes[:pred_t + 1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def get_pose_estimation_prediction_with_confidence(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, confidence = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds, confidence


def box_to_center_scale(box, model_image_width, model_image_height):
    center = np.zeros(2, dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', type=str, default='demo/inference-config.yaml')
    parser.add_argument('--video', type=str)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()

    update_config(cfg, args)

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load("../" + cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video or an image or webcam
    if args.video:
        video_pre_path = video_preprocessing(args.video)
        # video_pre_path = args.video
        video_name = args.video.split('/')[-1]
        vidcap = cv2.VideoCapture(video_pre_path)
        vidcap2 = cv2.VideoCapture(video_pre_path)
        img_shape = np.array([int(vidcap.get(3)), int(vidcap.get(4))])
    else:
        print('please use --video define the input.')
        return

    if args.video:
        "*********************************"
        smooth_joint_data = []
        no_detected_frames = []
        hrnet_data = []
        "*********************************"

        if args.write:
            hrnet_save_path = '../output/' + 'HRNet/' + video_name
            smoothnet_save_path = '../output/' + 'SmoothNet/' + video_name
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(hrnet_save_path, fourcc, fps, (int(vidcap.get(3)), int(vidcap.get(4))))
            count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in tqdm(range(count), desc="Processing: "):
            ret, image_bgr = vidcap.read()
            if ret:
                last_time = time.time()
                image = image_bgr[:, :, [2, 1, 0]]

                input = []
                img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)
                input.append(img_tensor)

                # object detection box
                pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

                # deal with multi peoples
                if len(pred_boxes) > 1:
                    highest = 0
                    square = [[0]]
                    for submit in range(len(pred_boxes)):
                        height = abs(pred_boxes[submit][1][1] - pred_boxes[submit][0][1])
                        if height >= highest:
                            square[0] = pred_boxes[submit]
                            highest = height
                    pred_boxes = square

                # pose estimation
                assert len(pred_boxes) == 1

                # pose estimation
                if len(pred_boxes) >= 1:
                    for box in pred_boxes:
                        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                        image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)

                        if len(pose_preds) >= 1:
                            for kpt in pose_preds:      # kpt shape is (17, 2)

                                kpt_h36m = coco_h36m(kpt)
                                kpt_h36m = coco_h36m_smoothNet(kpt_h36m)
                                draw_pose_h36m_smoothNet(kpt_h36m, image_bgr)
                                smooth_joint_data.append(kpt_h36m.flatten())

                        hrnet_data.append(pose_preds)
                else:
                    print("\nframe {} can't detected people".format(i))
                    no_detected_frames.append(i)

                if args.write:
                    out.write(image_bgr)
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break

            else:
                print('cannot load the video.')
                break
        cv2.destroyAllWindows()
        vidcap.release()

        if args.write:
            print('video has been saved as {}'.format(hrnet_save_path))
            out.release()

        # SmoothNet
        smooth_joint_data = np.array(smooth_joint_data)
        smooth_joint_data = smooth_joint_data.reshape((1, smooth_joint_data.shape[0], smooth_joint_data.shape[1]))
        SmoothNet_result = SmoothNetPredict(smooth_joint_data[0], img_shape)

        draw_smoothNet(SmoothNet_result, vidcap2, fourcc, smoothnet_save_path)
        print('smoothNet video has been saved as {}'.format(smoothnet_save_path))


if __name__ == '__main__':
    main()
