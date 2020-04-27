import numpy as np
from PIL import Image
import random


def read_image(path: "图片路径", dtype: "图片数据类型" = np.float32, color=True):
    """images ： 3×H×W ，BGR三通道，宽W，高H"""
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.array(img, dtype)
    except:
        if hasattr(f, 'close'):
            f.close()
    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return np.transpose(img, (2, 0, 1))


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.
        预计边界框将打包成二维
         形状张量：math：`（R，4）`，其中：math：`R`是
         图片中的边框。 第二个轴代表
         边界框。 它们是：（y_ {min}，x_ {min}，y_ {max}，x_ {max}），
         其中四个属性是左上角和
         右下角的顶点。
        bboxes： 4×K , K个bounding boxes，每个bounding box的左上角和右下角的座标，形如（Y_min,X_min, Y_max,X_max）,第Y行，第X列。
        Args:
              bbox（〜numpy.ndarray）：一个形状为（R，4）`的数组。
                 R是边界框的数量。
             in_size（元组）：一个长度为2的元组。高度和宽度
                 调整大小之前的图像。
             out_size（元组）：长度为2的元组。高度和宽度
                 调整大小后的图像。

         返回值：
             〜numpy.ndarray：
             边界框根据给定的图像形状重新缩放。
        """
    bbox = bbox.copy()
    # 对相应的bounding boxes 也也进行同等尺度的缩放
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """
    相应地翻转边界框。

     预计边界框将打包成二维
     形状张量：math：`（R，4）`，其中：math：`R`是
     图片中的边框。 第二个轴代表
     边界框。 它们是：（y_ {min}，x_ {min}，y_ {max}，x_ {max}）`，
     其中四个属性是左上角和
     右下角的顶点。

     Args：
         bbox（〜numpy.ndarray）：一个形状为（R，4）`的数组。
             R是边界框的数量。
         size（tuple）：长度为2的元组。高度和宽度
             调整大小之前的图像。
         y_flip（bool）：根据的垂直翻转来翻转边界框
             一个图像。
         x_flip（bool）：根据水平翻转的方向翻转边界框
             一个图像。

     返回值：
         〜numpy.ndarray：
         边界框根据给定的翻转进行翻转。
    """
    bbox = bbox.copy()
    H, W = size
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_max
        bbox[:, 2] = y_min
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_max
        bbox[:, 3] = x_min
    return bbox


def crop_bbox(bbox, y_slice=None,
              x_slice=None, allow_outside_center=True, return_param=False):
    """平移边界框以适合图像的裁剪区域。
    该方法主要与图像裁剪一起使用。
    此方法转换边界框的坐标，例如
    ：func：`data.util.translate_bbox`。此外，
    此功能会截断边界框以适合裁剪区域。
    如果边框与裁切区域不重叠，此边界框将被删除。
    预计边界框将打包成二维
    形状张量：math：`（R，4）`，其中：math：`R`是
    图片中的边框。第二个轴代表
    边界框。它们是：（y_ {min}，x_ {min}，y_ {max}，x_ {max}）`，
    其中四个属性是左上角和右下角的顶点。
    Args：
        bbox（〜numpy.ndarray）：要转换的边界框。形状是
            （R，4）`。 R是边界框的数量。
        y_slice（切片）：y轴的切片。
        x_slice（切片）：x轴的切片。
        allow_outside_center（bool）：如果此参数为False，
            中心在裁剪区域之外的边界框
            被删除。默认值为True。
        return_param（bool）：如果为True，则此函数返回
            保持边界框的索引。

    返回值：
        〜numpy.ndarray或（〜numpy.ndarray，dict）：

        如果：obj：`return_param = False`，则返回数组：obj：`bbox`。

        如果：obj：`return_param = True`，
        返回一个元组，其元素为：obj：`bbox，param`。
        ：obj：`param`是中间参数的字典，其中间参数
        下面列出了内容，包括键，值类型和说明
        价值。

        * ** index **（* numpy.ndarray *）：包含已使用\索引的数组
            边界框。
    """
    t, b = _slice_to_bound(y_slice)
    l, r = _slice_to_bound(x_slice)
    crop_bb = np.array((t, l, b, r))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2.
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb).all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    # 去除多余部分
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:])).all(axis=1)
    bbox = bbox[mask]

    if return_param:
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox


def _slice_to_bound(slice_):
    if slice_ is None:
        return 0, np.inf

    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def translate_bbox(bbox, y_offset=0, x_offset=0):
    """
    平移边界框。

    此方法主要与图像转换（例如填充）一起使用
    和裁剪，将图像的左上角
    坐标：（0，0）
    （y，x）=（y_ {offset}，x_ {offset}）`。

    预计边界框将打包成二维
    形状张量：math：`（R，4）`，其中：math：`R`是
    图片中的边框。第二个轴代表
    边界框。它们是：（y_ {min}，x_ {min}，y_ {max}，x_ {max}）`，
    其中四个属性是左上角和
    右下角的顶点。

    Args：
        bbox（〜numpy.ndarray）：要转换的边界框。形状是
            （R，4）`。 R是边界框的数量。
        y_offset（整数或浮点数）：沿y轴的偏移量。
        x_offset（整数或浮点数）：沿x轴的偏移量。

    返回值：
        〜numpy.ndarray：
        根据给定的偏移量翻译边界框。
    """
    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)
    return out_bbox

def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """
   沿垂直或水平方向随机翻转图像。

    Args：
        img（〜numpy.ndarray）：被翻转的数组。这是在
            CHW格式。
        y_random（布尔）：沿垂直方向随机翻转。
        x_random（布尔）：在水平方向上随机翻转。
        return_param（bool）：返回翻转信息。
        复制（布尔）：如果为False，将返回：obj：`img`的视图。

    返回值：
        〜numpy.ndarray或（〜numpy.ndarray，dict）：

        如果：obj：`return_param = False`，
        返回一个数组：obj：`out_img`，它是翻转的结果。

        如果：obj：`return_param = True`，
        返回一个元组，其元素是：obj：`out_img，param`。
        ：obj：`param`是中间参数的字典，其中间参数
        下面列出了内容，包括键，值类型和说明
        价值。

        * ** y_flip **（* bool *）：图像是否在\
            垂直方向与否。
        * ** x_flip **（* bool *）：图像是否在\
            水平方向与否。
    """
    y_filp, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()
    if return_param:
        return img, {'y_flip': y_filp, 'x_flip': x_flip}
    else:
        return img
