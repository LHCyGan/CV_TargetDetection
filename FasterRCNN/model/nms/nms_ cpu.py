import torch
import numpy as np


def nms_cpu(dets, thresh):
    dets = dets.numpy()
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 3]
    x2 = dets[:, 2]
    scores = dets[:, 4]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # 获取第0个值
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # (array([2, 3], dtype=int64), array([0, 0], dtype=int64))
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return torch.IntTensor(keep)