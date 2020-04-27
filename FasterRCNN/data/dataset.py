import torch
from data.voc_dataset import VOC_BBOX_LABEL_NAMES
from skimage import transform as skts
from torchvision import transforms as thts
from data import util
from utils_.config import Config as opt
import numpy as np

def inverse_normalize(img):
    """函数首先读取opt.caffe_pretrain判断是否使用caffe_pretrain进行预训练如果是的话，对图片进行逆正则化处理，
    就是将图片处理成caffe模型需要的格式"""
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801])).reshape(3, 1, 1)
        # 转为BGR
        return img[::-1, :, :]
    else:
        return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

def pytorch_normalize(img):
    """函数首先设置归一化参数normalize=tvtsf.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    然后对图片进行归一化处理img=normalize(t.from_numpy(img))"""
    normalize = thts.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.tensor(img))

    return img.numpy()

def caffe_normalize(img):
    """caffe的图片格式是BGR，所以需要img[[2,1,0],:,:]将RGB转换成BGR的格式，然后图片img = img*255 ,
    mean = np.array([122.7717,115.9465,102.9801]).reshape(3,1,1)设置图片均值"""
    img = img[[2, 0, 1], :, :]
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    # 用图片减去均值完成caffe形式的归一化处理
    img = (img - mean).astype(np.float32, copy=True)
    return img

def preprocess(img, min_size=600, max_size=1000):
    """
    预处理图像以进行特征提取。
     较短边的长度缩放为：obj.`self.min_size`。
     缩放后，如果长边的长度大于
     ：param min_size：
     ：obj：`self.max_size`，图像按比例缩放以适应较长的边缘
     到：obj：`self.max_size`。

     调整图像大小后，将图像减去平均图像值
     ：obj：`self.mean`。

     Args：
         img（〜numpy.ndarray）：图片。 这是CHW和RGB格式。
             其值的范围是[math：`[0，255]`。

     返回值：
         〜numpy.ndarray：预处理的图像。
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)  # 设置放缩比，这个过程很直觉，选小的方便大的和小的都能够放缩到合适的位置
    img = img / 255.
    img = skts.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    # 长短都应小于 max_size 和 min_size
    """将图片调整到合适的大小位于(min_size,max_size)之间、
然后根据opt.caffe_pretrain是否存在选择调用前面的pytorch正则化还是caffe_pretrain正则化"""
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalize
    return normalize(img)


class Transform:
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)  # 将图片进行最小最大化放缩然后进行归一化
        _, o_H, o_W = img.shape
        scale = o_H / H  # 放缩前后相除，得出放缩比因子
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))  # 重新调整bboxes框的大小

        # 水平翻转
        # 进行图片的随机反转，图片旋转不变性，增强网络的鲁棒性！
        img, params = util.random_flip(img,
                                       x_random=True, return_param=True)
        # 同样的对bboxes进行随机反转
        bbox = util.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])
        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOC_BBOX_LABEL_NAMES(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        """从数据集存储路径中将例子一个个的获取出来，然后调用前面的Transform函数将图片,label进行最小值最大值放缩归一化，
        重新调整bboxes的大小，然后随机反转，最后将数据集返回！"""
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label, difficult))
        # TODO：检查其步幅为负数以解决此问题，而不是全部复制
        # 给定numpy数组的某些步幅为负。
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    """从Voc_data_dir中获取数据的时候使用了split='test'也就是从test往后分割的部分数据送入到TestDataset的self.db中，
    然后在进行图片处理的时候，并没有调用transform函数，因为测试图片集没有bboxes需要考虑，同时测试图片集也不需要随机反转，
    反转无疑为测试准确率设置了阻碍！所以直接调用preposses()函数进行最大值最小值裁剪然后归一化就完成了测试数据集的处理！
    最后将整个self.db返回"""
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = opt.VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)




