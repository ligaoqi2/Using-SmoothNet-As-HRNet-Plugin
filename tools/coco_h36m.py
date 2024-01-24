'''
Project: https://github.com/fabro66/GAST-Net-3DPoseEstimation
'''
import numpy as np

h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]

scores_h36m_toe_oeder = [1, 2, 3, 5, 6, 7, 11, 13, 14, 15, 16, 17, 18]
kpts_h36m_toe_order = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
scores_coco_order = [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]

h36m_mpii_order = [3, 2, 1, 4, 5, 6, 0, 8, 9, 10, 16, 15, 14, 11, 12, 13]
mpii_order = [i for i in range(16)]
lr_hip_shouler = [2, 3, 12, 13]

h36m_order = [10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8]
h36m_order_smooth = [9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8]


def coco_h36m(keypoints):

    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps_keypoints = np.zeros((4, 2), dtype=np.float32)

    htps_keypoints[0, 0] = np.mean(keypoints[1:5, 0], axis=0, dtype=np.float32)
    htps_keypoints[0, 1] = np.sum(keypoints[1:3, 1], axis=0, dtype=np.float32) - keypoints[0, 1]

    htps_keypoints[1, :] = np.mean(keypoints[5:7, :], axis=0, dtype=np.float32)
    htps_keypoints[1, :] += (keypoints[0, :] - htps_keypoints[1, :]) / 3

    htps_keypoints[2, :] = np.mean(keypoints[11:13, :], axis=0, dtype=np.float32)

    htps_keypoints[3, :] = np.mean(keypoints[[5, 6, 11, 12], :], axis=0, dtype=np.float32)

    keypoints_h36m[spple_keypoints, :] = htps_keypoints

    keypoints_h36m[h36m_coco_order, :] = keypoints[coco_order, :]

    keypoints_h36m[9, :] -= (keypoints_h36m[9, :] - np.mean(keypoints[5:7, :], axis=0, dtype=np.float32)) / 4
    keypoints_h36m[7, 0] += 2 * (keypoints_h36m[7, 0] - np.mean(keypoints_h36m[[0, 8], 0], axis=0, dtype=np.float32))
    keypoints_h36m[8, 1] -= (np.mean(keypoints[1:3, 1], axis=0, dtype=np.float32) - keypoints[0, 1]) * 2 / 3

    return keypoints_h36m


def coco_h36m_smoothNet(keypoints):
    keypoints_h36m_smooth = np.zeros((16, 2), dtype=np.float32)
    keypoints_h36m_smooth[h36m_order_smooth, :] = keypoints[h36m_order, :]
    return keypoints_h36m_smooth


def coco_h36m_score(scores):
    scores_h36m = np.zeros_like(scores, dtype=np.float32)
    htps_scores = np.zeros((4, 1), dtype=np.float32)

    htps_scores[0] = np.mean(scores[1:5], dtype=np.float32)
    htps_scores[1] = np.mean(scores[5:7])
    htps_scores[2] = np.mean(scores[11:13], dtype=np.float32)
    htps_scores[3] = np.mean(scores[[5, 6, 11, 12]], dtype=np.float32)

    scores_h36m[spple_keypoints] = htps_scores
    scores_h36m[h36m_coco_order] = scores[coco_order]

    return scores_h36m


def coco_h36m_smoothNet_score(scores):
    scores_h36m_smooth = np.zeros((16, 1), dtype=np.float32)
    scores_h36m_smooth[h36m_order_smooth, :] = scores[h36m_order, :]
    return scores_h36m_smooth


if __name__ == "__main__":
    print(len(h36m_order))
    print(len(h36m_order_smooth))
