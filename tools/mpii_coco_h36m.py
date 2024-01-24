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


def coco_h36m(keypoints):
    temporal = keypoints.shape[0]
    # 视频帧数temporal
    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)     # (153, 17, 2)
    htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)   # (153, 4, 2)

    # htps_keypoints: 头部， 胸部，骨盆，脊柱
    # HRNet kpts: [鼻子0，左眼1，右眼2，左耳3，右耳4，左肩5，右肩6，左肘7，右肘8，左手腕9，右手腕10，左髋11，右髋12，左膝13，右膝14，左脚踝15，右脚踝16]
    htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
    htps_keypoints[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
    # 第0维:头部
    htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3
    # 第1维:胸部
    htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
    # 第2维:骨盆
    htps_keypoints[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)
    # 第3维:脊柱

    # np.mean()求平均值:axis = 0->对各列求均值，axis = 1->对各行求均值(二维的情况)
    # axis等于几，就是在第几维度进行操作，也就是将第几维度压缩至一维。
    # np切片和list相同:[a:b]取得是list[a]~list[b-1]

    # htps_keypoints[头，胸，骨盆，脊柱]
    keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
    # keypoints_h36m
    # [骨盆，左眼，右眼，左耳，右耳，左肩，右肩，脊柱，胸，左手腕，头，左髋，右髋，左膝，右膝，左脚踝，右脚踝]
    # keypoints
    # [鼻子0，左眼1，右眼2，左耳3，右耳4，左肩5，右肩6，左肘7，右肘8，左手腕9，右手腕10，左髋11，右髋12，左膝13，右膝14，左脚踝15，右脚踝16]
    keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]
    # keypoints_h36m
    # [骨盆0，右髋1，右膝2，右脚踝3，左髋4，左膝5，左脚踝6，脊柱7，胸8，鼻子9，头10，左肩11，左肘12，左手腕13，右肩14，右肘15，右手腕16]

    # spple_keypoints = [10, 8, 0, 7]
    # h36m_coco_order = [9, 11, 14, 12, 15, 13, 16,  4,  1,  5,  2,  6,  3]
    # coco_order =      [0,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]

    keypoints_h36m[:, 9, :] -= (keypoints_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    keypoints_h36m[:, 7, 0] += 2 * (
                keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]) * 2 / 3

    # half body: the joint of ankle and knee equal to hip
    # keypoints_h36m[:, [2, 3]] = keypoints_h36m[:, [1, 1]]
    # keypoints_h36m[:, [5, 6]] = keypoints_h36m[:, [4, 4]]

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]
    # valid_frames:关节点坐标加起来不为0的帧
    # np.where()
    return keypoints_h36m, valid_frames


def mpii_h36m(keypoints):
    temporal = keypoints.shape[0]
    keypoints_h36m = np.zeros((temporal, 17, 2), dtype=np.float32)
    keypoints_h36m[:, h36m_mpii_order] = keypoints
    # keypoints_h36m[:, 7] = np.mean(keypoints[:, 6:8], axis=1, dtype=np.float32)
    keypoints_h36m[:, 7] = np.mean(keypoints[:, lr_hip_shouler], axis=1, dtype=np.float32)

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]
    return keypoints_h36m, valid_frames


def coco_h36m_toe_format(keypoints):
    assert len(keypoints.shape) == 3
    temporal = keypoints.shape[0]

    new_kpts = np.zeros((temporal, 19, 2), dtype=np.float32)

    # convert body+foot keypoints
    coco_body_kpts = keypoints[:, :17].copy()
    h36m_body_kpts, _ = coco_h36m(coco_body_kpts)
    new_kpts[:, kpts_h36m_toe_order] = h36m_body_kpts
    new_kpts[:, 4] = np.mean(keypoints[:, [20, 21]], axis=1, dtype=np.float32)
    new_kpts[:, 8] = np.mean(keypoints[:, [17, 18]], axis=1, dtype=np.float32)

    valid_frames = np.where(np.sum(new_kpts.reshape(-1, 38), axis=-1) != 0)[0]

    return new_kpts, valid_frames
