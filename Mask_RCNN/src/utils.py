import os
import sys
import tensorflow as tf
import math
import random
import scipy
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
import skimage
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion


# 训练好的coco数据集权重
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


def extract_bboxex(mask):
    """
    # 根据mask计算bounding boxes
    # mask: [height, width, num_instances]， Mask的值非0即1。
    # Returns: bbox数组[num_instances, (y1, x1, y2, x2)]
    """
    boxes = np.zeros((mask.shape[-1], 4), dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # 返回mask值为1的横、纵坐标
        horizontal_indices = np.where(np.any(m, axis=0))[0] #[0]为indeces,[1]为values
        vertical_indices = np.where(np.any(m, axis=1))[0]
        if horizontal_indices.shape[0]:
            # 获取第一个和最后一个索引
            # 多个索引, 必须使用两个[]
            x1, x2 = horizontal_indices[[0, -1]]
            y1, y2 = vertical_indices[[0, -1]]
            # x2和y2不应该作为box的一部分. 所以各自加1.
            x2 += 1
            y2 += 1
        else:
            # 该instance没有mask，可能是由于缩放和裁剪导致的，将bbox设为0
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def compute_iou(box, boxes, box_area, boxes_area):
    # 计算给定box与boxes数组中的每个box的IoU.
    # box: 一维数组[y1, x1, y2, x2]
    # boxes: [boxes_count, (y1, x1, y2, x2)]
    # box_area: float型. 'box'的面积
    # boxes_area: 数组，长度是boxes的数量
    # 注意:出于效率的考虑，areas参数这里是传入进来的而不是计算得来的
    # 在调用的时候只计算一次以避免重复的工作
    # Calculate intersection areas

    # 利用numpy的广播特性
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + box_area[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_overlaps(boxes1, boxes2):
    # 计算两个boxes集合的IoU重叠
    # boxes1, boxes2: [N, (y1, x1, y2, x2)]。
    # 为了获得更好的性能，先传大的集合再传小的。
    # Areas of anchors and GT boxes
    # anchors和GT boxes的面积

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    # 计算重叠生成矩阵[boxes1 count, boxes2 count]
    # 每个cell都有IoU值
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        # box2:相当于gt_box
        box2 = boxes2[i]
        # boxes1: anchor
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_overlaps_masks(masks1, masks2):
    # 计算两个masks集合的IoU重叠
    # masks1, masks2: [Height, Width, instances]
    # If either set of masks is empty return empty result
    # 如果任意一个masks集合为空，则返回空的结果
    if masks1.shape[0] == 0 or masks2.shape[0] == 0:
        return np.zeros((masks1.shape[0], masks2.shape[-1]))
    # 扁平化masks并计算它们的面积：通过计算mask所包围的像素值和来计算
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def non_max_suppression(boxes, scores, threshold):
    # 执行non-maximum suppression并返回保留的boxes的索引
    # boxes: [N, (y1, x1, y2, x2)].注意(y2, x2)可以会超过box的边界
    # scores: box的分数的一维数组
    # threshold: Float型. 用于过滤IoU的阈值
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != 'f':
        boxes = boxes.astype(np.float32)

    # 计算box的面积
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # 获取根据分数排序的boxes的索引
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # 选择排在最前的box，并将其索引加到列表中
        i = ixs[0]
        pick.append(i)
        # 计算选择的box与剩下的box的IoU
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # 确定IoU大于阈值的boxes. 这里返回的是ix[1:]之后的索引
        # 所以为了与ixs保持一致，将结果加1
        remove_ixs = np.where(iou > threshold)[0] + 1
        # 将选择的box和重叠的boxes的索引删除
        ixs = np.delete(ixs, remove_ixs)
        # 删除自身
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)

def apply_box_deltas(boxes, deltas):
    # 将给出的deltas应用到指定的boxes
    # boxes: [N, (y1, x1, y2, x2)]. 注意(y2, x2)可能超出box的边界
    # deltas: [N, (dy, dx, log(dh), log(dw))]
    boxes = boxes.astype(np.float32)
    # 将box转化成y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # 应用deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.xp(deltas[:, 3])
    # 将y，x，h，w转回y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2= x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


# tensorflow实现
# 修正的是与gt_box的差异
def box_refinement_graph(box, gt_box):
    # 将box向gt_box优化
    # box和gt_box都是[N, (y1, x1, y2, x2)]
    box = tf.cast(box, dtype=tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result

# numpy实现
def box_refinement(box, gt_box):
    # 将box向gt_box优化。
    # box和gt_box都是[N, (y1, x1, y2, x2)]，(y2, x2)可能超出box的边界。

    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


class Dataset:
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # 背景类通常作为第一类
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):

        assert "." not in source, "Source name cannot contain a dot"

        # Does the class exist already?

        # 该类是否已经存在?

        for info in self.class_info:

            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip

                # source.class_id组合已经可用, 跳过

                return

        # Add the class
        # 添加类别

        self.class_info.append({

            "source": source,

            "id": class_id,

            "name": class_name,

        })

    def add_image(self, source, image_id, path, **kwargs):

        image_info = {

            "id": image_id,

            "source": source,

            "path": path,

        }

        image_info.update(kwargs)

        self.image_info.append(image_info)

    def image_reference(self, image_id):

        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.
        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """

        """返回图像的链接， 或者查找和调试的信息.
        在你自己的数据集类中重载本函数, 但是如果图像不属于你的数据集，
        则仍然使用本函数.
        """

        return ""

    def prepare(self, class_map=None):

        """Prepares the Dataset class for use.
        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.

        """

        """

        数据集准备.
        TODO: 类别映射还未实现.一旦实现, 则可以将来自不同数据集的类别映射到相同的class ID.

        """

        def clean_name(name):

            """Returns a shorter version of object names for cleaner display."""
            # 为方便显示，返回一个简短的物体名称

            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        # 通过class_info和image_info初始化

        self.num_classes = len(self.class_info)

        self.class_ids = np.arange(self.num_classes)

        self.class_names = [clean_name(c["name"]) for c in self.class_info]

        self.num_images = len(self.image_info)

        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        # 将class和image IDs映射到内部IDs

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id

                                      for info, id in zip(self.class_info, self.class_ids)}

        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id

                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        # 将sources映射到支持的class_ids

        self.sources = list(set([i['source'] for i in self.class_info]))

        self.source_class_ids = {}

        # Loop over datasets
        # 循环遍历数据集

        for source in self.sources:

            self.source_class_ids[source] = []

            # Find classes that belong to this dataset
            # 找到属于数据集的类别

            for i, info in enumerate(self.class_info):

                # Include BG class in all datasets
                # 所有的数据集都包含背景类

                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):

        """Takes a source class ID and returns the int class ID assigned to it.
        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """

        # 取出class ID并返回赋予它的整型class ID
        # 例如：
        # dataset.map_source_class_id("coco.12") -> 23

        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):

        """Map an internal class ID to the corresponding class ID in the source dataset."""
        # 将class ID映射到它在数据集中对应的class ID。
        info = self.class_info[class_id]
        assert info['source'] == source

        return info['id']

    @property
    def image_ids(self):

        return self._image_ids

    def source_image_link(self, image_id):

        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.

        """

        # 返回图像的路径或者URL链接，
        # 如果图像是在线的话，重载本函数返回其URL链接.

        return self.image_info[image_id]["path"]

    def load_image(self, image_id):

        """Load the specified image and return a [H,W,3] Numpy array.
        """

        # 加载指定图像并返回[H,W,3]的Numpy数组。
        # Load image

        image = skimage.io.imread(self.image_info[image_id]['path'])

        # If grayscale. Convert to RGB for consistency.
        # 如果是灰度图像，则转化成RGB图像以保持一致性。

        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        # If has an alpha channel, remove it for consistency
        # 如果图像有alpha通道，则去掉
        if image.shape[-1] == 4:
            image = image[..., :3]

        return image

    def load_mask(self, image_id):

        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        """

        加载给定图像的instance masks.
        不同的数据集使用不同的方式存储masks。本函数将不同格式的mask转化
        成同一格式的bitmap，维度是[height, width, instances].
        返回:
        masks:一个bool数组，尺寸是[height, width, instance count]，
            一个instance有一个mask.
        class_ids:一个一维数组，元素是instance masks的class IDs。
        """

        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        # 重载本函数以适应你的数据集，        #否则返回空的mask。
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)

        return mask, class_ids

def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode='square'):
    """缩放图像同时保持宽高比不变。
    min_dim: 如果给出了该值，缩放图像时要保持短边 == min_dim。
    max_dim: 如果给出了该值，缩放图像时要保持长边不超过它。
    min_scale: 如果给出了该值，则使用它来缩放图像，而不管是否满足min_dim。
    mode: 缩放模式.
        none:   无缩放或填充，返回原图。
        square: 缩放或填充0，返回[max_dim, max_dim]大小的图像。
        pad64:  宽和高填充0，使他们成为64的倍数。
                如果IMAGE_MIN_DIM 或 IMAGE_MIN_SCALE不为None, 则在填充之前先
                缩放。IMAGE_MAX_DIM在该模式中被忽略。
                要求为64的倍数是因为在对FPN金字塔的6个levels进行上/下采样时保证平滑(2**6=64)。
        crop:   对图像进行随机裁剪. 首先, 基于IMAGE_MIN_DIM和IMAGE_MIN_SCALE。
                对图像进行缩放, 然后随机裁剪IMAGE_MIN_DIM x IMAGE_MIN_DIM大小。
                仅在训练时使用。IMAGE_MAX_DIM在该模式中未使用。
    Return:
    image: 缩放后的图像
    window: (y1, x1, y2, x2). 如果给出了max_dim, 可能会对返回图像进行填充。
            如果是这样的，则窗口是全图的部分图像坐标 (不包括填充的部分)。 x2, y2 不包括。
    scale: 图像缩放因子
    padding: 图像填充部分[(top, bottom), (left, right), (0, 0)]
    """
    # 保持输入域输出类型一致
    image_dtype = image.dtype
    # 默认窗口（y1, x1, y2, x2）, scale==1
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == 'none':
        return image, window, scale, padding, crop
    # 缩放
    if min_dim:
        # 放大而非缩小
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # 是否超过max_dim
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # 使用双线性差值缩放图像
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                                         preserve_range=True)
    # 需要裁减还是需要填充
    if mode == "square":
        # 获取高和宽
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = np.pad(image, ((top_pad, bottom_pad),
                                 (left_pad, right_pad),
                                 (0, 0)), mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # 长和宽都能被64整除
        assert min_dim % 64 == 0
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0

        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - top_pad
        else:
            left_pad = right_pad = 0

        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # 随机裁减
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception(f"Mode {mode} not supported")

    return image.astype(image_dtype), window, scale, padding, crop

def resize_mask(mask, scale, padding, crop=None):
    """缩放mask"""
    # 用指定的scale和padding缩放mask。
    # 一般来说, 为了保持图像和mask的一致性，scale和padding是通过resize_image()
    # 获取的。
    # scale: mask缩放因子
    # padding: 填充[(top, bottom), (left, right), (0, 0)]

    # 去除warnging请使用scipy 0.13.0，zoom()输出的形状大小是用round()而非int()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, w, h = crop
        mask = mask[y:y+h, x:x+w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def minimize_mask(bbox, mask, mini_shape):
    """将masks缩放到一个更小的尺寸以减小内存的负载.
    使用expand_masks()可以将Mini-masks放大回图像的尺寸"""

    mini_mask = np.zeros(mini_shape + (mask.shape[-1], ), dtype=np.bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.round(m).astype(bool)
    return mini_mask

def expand_mask(bbox, mini_mask, image_shape):
    """将mini masks放大回图像尺寸. minimize_mask()的逆变换."""
    mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1]), dtype=bool)
    for i in range(mini_mask.shape[-1]):
        m = mini_mask[:, : , i]
        y1, x1, y2, x2 = bbox[i][:4]
        h = y2 - y1; w = x2 - x1
        # 使用双线性差值缩放
        m = skimage.resize(m, (h, w))
        mask[y1:y2, x1:x2] = np.round(m).astype(bool)
    return mask


def mold_mask(mask, config):
    pass

def unmold_mask(mask, bbox, image_shape):
    """将神经网络产生的mask转化成其原来的形状。
    mask: [height, width]，float型. 一个更小的尺寸是28x28。
    bbox: [y1, x1, y2, x2]，包围mask的box。
    返回一个与原始图像尺寸一样的二值mask。"""
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = skimage.resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(bool)

    # 将mask放到正确位置上
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """scales: anchor大小的一维数组，单位是像素。例如: [32, 64, 128]。
    ratios: anchor宽/高比的一维数组。 例如: [0.5, 1, 2]。
    shape: [height, width] feature map的形状。
    feature_stride: feature map与图像关联的stride，单位是像素。
    anchor_stride: feature map与anchors关联的stride。 例如, anchor_stride=2
    则每隔一个feature map生成anchors。"""

    # 对scales进行纵向扩展，扩展的次数以ratios为依据
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # 枚举宽和高
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # 枚举feature空间的偏移
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # 枚举shifts, widths, 和heights结合
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # 变形以获得(y, x)的(h, w)列表
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # 转成角点坐标(y1, x1, y2, x2)
    boxes = np.concatenate(
        [box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes],
        axis=1
    )
    return boxes

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """在特征金字塔的不同层级生成anchors. 每个scale
    关联一个金字塔层级, 但是每个ratio应用于金字塔的所有层级.
    返回:
    anchors: [N, (y1, x1, y2, x2)]. 所有生成的anchors放在一个数组中.
            以给出的scales顺序排序. 所以, scale[0]对应的anchors排在
            最前面，接着是scale[1]对应的anchors，以此类推."""
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i], feature_strides[i],
                                        anchor_stride))
    return np.concatenate(anchors, axis=0)

def trim_zeros(x):
    """ tensors比实际有效数据大并且填充0是很常见的。
    本函数就是去掉那些全是0的行。
    x: [rows, columns]。"""
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

def compute_matches(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids,
                    pred_scores, pred_masks, iou_threshold=0.5, score_threshold=0.0):
    """计算预测和ground truth instances的匹配情况.
    返回:
    gt_match: 一维数组。每一个GT box都有与其匹配的box的索引。
    pred_match: 一维数组。 每一个预测的box，都有与其匹配的ground truth box的索引。
    overlaps: [pred_boxes, gt_boxes] IoU重叠。"""
    # 去除填充的0
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # 根据分数从高到低的顺序对预测的结果排序
    indices = np.argsort(pred_boxes)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]
    # 计算IoU重叠[pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # 循环预测值并找出匹配的ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # 找出最佳匹配的ground truth box
        # 1. 根据分数排序匹配结果
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. 去掉分数低的
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. 寻找匹配
        for j in sorted_ixs:
            # 如果ground truth box已经匹配上了, 则继续下一个
            if gt_match[j] > 0:
                continue
            # 如果IoU小于阈值, 则结束循环
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # 是否找到匹配?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):

    """
    计算相同阈值的IoU集合的平均准确率 (默认阈值是0.5)。
    返回:
    mAP: 平均准确率
    precisions: 不同阈值的准确率列表。
    recalls: 不同阈值的召回率列表。
    overlaps: [pred_boxes, gt_boxes] IoU重叠。
    """
    # 获取匹配和重叠

    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # 计算每个预测box步骤的准确率和召回率

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
    # 在首尾进行填充以便于计算
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # 确保准确率的值是减小而不是增大，详细说明请参考VOC论文

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # 计算平均AP
    # 求出recall变化的点坐标
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    # 求PR曲线下的面积即AP
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):

    # 计算给定阈值范围的AP. 默认的范围是0.5-0.95.
    # 默认是0.5到0.95，步长是0.05

    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    # 计算AP
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = \
            compute_ap(gt_box, gt_class_id, gt_mask,
                       pred_box, pred_class_id, pred_score, pred_mask,
                       iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()

    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))

    return AP


def compute_recall(pred_boxes, gt_boxes, iou):

    # 计算给定IoU阈值的召回率
    # pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    # gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    # Measure overlaps
    # 计算重叠

    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    iou_max = np.max(overlaps, axis=1)

    iou_argmax = np.argmax(overlaps, axis=1)

    positive_ids = np.where(iou_max >= iou)[0]

    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]

    return recall, positive_ids


def batch_slice(inputs, graph_fn, batch_size, names=None):
    """
    将输入分解成slices并将每个slice喂给给定的计算graph的副本，
    最后将结果组合起来。仅允许在一个batch的输入上允许graph，尽管graph只支持一个instance。
    inputs: tensors列表. 它们的第一个维度必须相同。
    graph_fn: 函数，返回graph的一个 TF tensor。
    batch_size: 要分割的slices数量。
    names: 如果给出, 将names赋给结果tensors。

    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    outputs = []
    # 将batch拆分了，逐个处理， 以batch=3为例，[3,261888,4]
    for i in range(batch_size):

        # 抽出batch[i],inputs_slice为[261888,4]
        inputs_slice = [x[i] for x in inputs]
        # output_slice为[6000,4]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]

        outputs.append(output_slice)
    # 下面示意中假设每次graph_fn返回两个tensor
    # [[tensor11, tensor12], [tensor21, tensor22], ……]
    # ——> [(tensor11, tensor21, ……), (tensor12, tensor22, ……)]  zip返回的是多个tuple
    outputs = list(zip(*outputs))
    if names is None:
        names = [None] * len(outputs)
    # 将batch维度合并回去
    # tf.stack沿着新轴堆叠数据,axis=0时意味着会增加一个newaxis=0
    # zip打包时，o=(b0,b1,b2)，并且沿着axis=0进行stack堆叠，还原为原始的inputs.shape
    # result结构为tf.Tensor->[b1,b2,b3],shape为[3,6000,4]
    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]
    return result


def download_trained_weights(coco_model_path, verbose=1):
    # 下载发布的COCO训练权重文件。
    # coco_model_path: 保存COCO训练权重文件的路径

    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")

    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)

    if verbose > 0:
        print("... done downloading pretrained model!")


def norm_boxes(boxes, shape):
    """
    将boxes从像素坐标转化到归一化坐标。
    boxes: [N, (y1, x1, y2, x2)]像素坐标
    shape: [..., (height, width)]，单位是像素
    注意: 在像素坐标中(y2, x2)超过了box. 但是在归一化坐标中是在box内部.
    返回:
        [N, (y1, x1, y2, x2)]归一化坐标
    """

    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """
    将boxes从归一化坐标转化到像素坐标。
    boxes: [N, (y1, x1, y2, x2)]归一化坐标
    shape: [..., (height, width)]，单位是像素
    注意: 在像素坐标中(y2, x2)超过了box。 但是在归一化坐标中是在box内部。
    返回:
        [N, (y1, x1, y2, x2)]像素坐标
    """

    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,

           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    # 对Scikit-Image resize()函数的封装
    # Scikit-Image在每次调用Resize（）时都会产生警告，如果没有接收正确的参数。
    # 正确的参数取决于Scikit-Image的版本。这里通过使用不同的参数来解决版本问题。
    # 提供中心位置来控制缩放。

    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)