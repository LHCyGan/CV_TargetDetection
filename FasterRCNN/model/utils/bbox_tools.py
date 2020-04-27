import numpy as np
import six
from six import __init__



def loc2bbox(src_bbox, loc):
    """已知源bbox 和位置偏差dx，dy，dh，dw，求目标框G"""
    if src_bbox.shape[0] == 0:  # src_bbox：（R，4），R为bbox个数，4为左下角和右上角四个坐标(这里有误，按照标准坐标系中y轴向下，应该为左上和右下角坐标)
        return np.zeros((0, 4), dtype=loc.dtype)
    src_bbox = src_bbox.astype(loc.dtype, copy=False)
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    # 计算出中心点坐标
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width
    # src_height为Ph,src_width为Pw，src_ctr_y为Py，src_ctr_x为Px
    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]
    # RCNN中提出的边框回归：寻找原始proposal与近似目标框G之间的映射关系
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]  #ctr_y为Gy
    ctr_x = dx * src_ctr_x[:, np.newaxis] + src_ctr_x[:, np.newaxis]  # ctr_x为Gx
    h = src_height[:, np.newaxis] * np.exp(dh)  #h为Gh
    w = src_width[:, np.newaxis] * np.exp(dw)  #w为Gw
    # 上面四行得到了回归后的目标框（Gx,Gy,Gh,Gw）

    # 由中心点转换为左上角和右下角坐标
    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w
    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """已知源框和目标框求出其位置偏差"""
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = height * 0.5 + src_bbox[:, 0]
    ctr_x = width * 0.5 + src_bbox[:, 1]  #计算出源框中心点坐标

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = base_height * 0.5 + dst_bbox[:, 0]
    base_ctr_x = base_width * 0.5 + dst_bbox[:, 1]  # 计算出源框中心点坐标
    #Machine limits for floating point types
    eps = np.finfo(height.dtype).eps  #求出最小的正数
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)  #将height,width与其比较保证全部是非负

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()  # np.vstack按照行的顺序把数组给堆叠起来
    return loc

def bbox_iou(bbox_a, bbox_b):
    """求两个bbox的相交的交并比"""
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError   #确保bbox第二维为bbox的四个坐标（ymin，xmin，ymax，xmax）
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])   # tl为交叉部分框左上角坐标最大值，为了利用numpy的广播性质，bbox_a[:, None, :2]的shape是(N,1,2)，bbox_b[:, :2]shape是(K,2),由numpy的广播性质，两个数组shape都变成(N,K,2)，也就是对a里每个bbox都分别和b里的每个bbox求左上角点坐标最大值
    br = np.maximum(bbox_a[:, None, 2:], bbox_b[:, 2:])   #br为交叉部分框右下角坐标最小值
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)  #所有坐标轴上tl<br时，返回数组元素的乘积(y1max-yimin)X(x1max-x1min)，bboxa与bboxb相交区域的面积
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)  #计算bbox_a的面积
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)  #计算bbox_b的面积
    return area_i / (area_a[:, None] + area_b - area_i)


def generate_base_anchor(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    """ #对特征图features以基准长度为16、选择合适的ratios和scales取基准锚点anchor_base。（选择长度为16的原因是图片大小为600*800左右，基准长度16对应的原图区域是256*256，考虑放缩后的大小有128*128，512*512比较合适）"""
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            # 生成9种不同比例的h和w
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base

