import os
import pathlib
import numpy as np
import xml.etree.ElementTree as ET
from .util import read_image


class VOCBboxDataset:
    """
    PASCAL`VOC`_的边界框数据集。

    .. _`VOC`：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    索引对应于每个图像。

    当通过索引查询时，如果：obj：`return_difficult == False`，
    该数据集返回一个对应的
    img，bbox，label，图像的元组，边界框和标签。
    这是默认行为。
    如果：obj：`return_difficult == True`，则此数据集返回对应的
    img，bbox，label，困难。 ：obj：`difficult`是一个布尔数组
    指示边界框是否标记为困难。

    包围盒包装成二维张量形状
    ：math：`（R，4）`，其中：math：`R`是其中的边界框的数量
    图片。第二个轴表示边界框的属性。
    它们是：（y_ {min}，x_ {min}，y_ {max}，x_ {max}）`，其中
    四个属性是左上角和右下角的坐标
    顶点。

    标签被打包成形状为（R，）`的一维张量。
    R是图像中边界框的数量。
    标签：math：`l的类名称是：math：`l`的元素
    ：obj：`VOC_BBOX_LABEL_NAMES`。

    数组：obj：`difficult`是形状的一维布尔数组
    （R，）`。 R是图像中边界框的数量。
    如果use_difficult是obj False，则此数组为
    具有所有False的布尔数组。

    图像的类型，边框和标签如下。

    *：obj：`img.dtype == numpy.float32`
    *：obj：`bbox.dtype == numpy.float32`
    *：obj：`label.dtype == numpy.int32`
    *：obj：`difficult.dtype == numpy.bool`

    Args：
        data_dir（字符串）：训练数据根的路径。
            即“ / data / image / voc / VOCdevkit / VOC2007 /”
        split（{'train'，'val'，'trainval'，'test'}）：选择
            数据集。 ：obj：`test`分割仅适用于
            2007数据集。
        年（{'2007'，'2012'}）：使用为挑战准备的数据集
            举办于：obj：`year`。
        use_difficult（bool）：如果为True，则使用标记为
            在原始注释中很难。
        return_difficult（bool）：如果为True，则此数据集返回
            布尔数组
            指示边界框是否标记为困难
            或不。默认值为False。
    """
    def __init__(self, data_dir, split='trainval', use_difficult=False, return_difficult=False):
        id_list_file = os.path.join(data_dir, '/{}.txt'.format(split))  # id_list_file为split.txt，split为'trainval'或者'test'
        self.ids = [id_.strip() for id_ in open(id_list_file)]  # id_为每个样本文件名
        self.data_dir = data_dir  # #写到/VOC2007/的路径
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES   #20类

    def __len__(self):
        return self.ids.__len__()  # #trainval.txt有5011个，test.txt有210个

    def get_example(self, i):
        """
        返回第i个示例。

         返回彩色图像和边框。 图像为CHW格式。
         返回的图像是RGB。

         Args：
             i（整数）：示例的索引。

         返回值：
             图像和边界框的元组
        """
        id_ = self.ids[i]
        # 解析xml文档
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + ".xml"))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1: #标为difficult的目标在测试评估中一般会被忽略
                continue
            # xml文件中包含object name和difficult(0或者1,0代表容易检测)
            difficult.append(int(obj.find('difficult').text))
            # bndbox（xmin,ymin,xmax,ymax),表示框左下角和右上角坐标
            bndbox_anno = obj.find('bndbox')
            # 让坐标基于（0,0）
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            # 框中object name
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        # 所有object的bbox坐标存在列表里
        bbox = np.stack(bbox).astype(np.float32)
        # 所有object的label存在列表里
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch 不支持 np.bool，所以这里转换为uint8

        #根据图片编号在/JPEGImages/取图片
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    # 一般如果想使用索引访问元素时，就可以在类中定义这个方法（__getitem__(self, key) )
    __getitem__ = get_example




VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

