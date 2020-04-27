import numpy as np
import cupy as np

from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox
from model.nms import non_maximum_suppression


# 下面是ProposalCreator的代码： 这部分的操作不需要进行反向传播，因此可以利用numpy/tensor实现
class ProposalCreator:  # 对于每张图片，利用它的feature map，计算（H/16）x(W/16)x9(大概20000)个anchor属于前景的概率，然后从中选取概率较大的12000张，利用位置回归参数，修正这12000个anchor的位置， 利用非极大值抑制，选出2000个ROIS以及对应的位置参数。

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size,
                 scale=1.):  # 这里的loc和score是经过region_proposal_network中经过1x1卷积分类和回归得到的
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms  # 12000
            n_post_nms = self.n_train_post_nms  # 经过NMS后有2000个
        else:
            n_pre_nms = self.n_test_pre_nms  # 6000
            n_post_nms = self.n_test_post_nms  # 经过NMS后有300个


        roi = loc2bbox(anchor, loc)  # 将bbox转换为近似groudtruth的anchor(即rois)
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])  # 裁剪将rois的ymin,ymax限定在[0,H]
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])  # 裁剪将rois的xmin,xmax限定在[0,W]

        min_size = self.min_size * scale  # 16
        hs = roi[:, 2] - roi[:, 0]  # rois的宽
        ws = roi[:, 3] - roi[:, 1]  # rois的长
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]  # 确保rois的长宽大于最小阈值
        roi = roi[keep, :]
        score = score[keep]  # 对剩下的ROIs进行打分（根据region_proposal_network中rois的预测前景概率）

        order = score.ravel().argsort()[::-1]  # 将score拉伸并逆序（从高到低）排序
        if n_pre_nms > 0:
            order = order[:n_pre_nms]  # train时从20000中取前12000个rois，test取前6000个
        roi = roi[order, :]

        keep = non_maximum_suppression(
            cp.ascontiguousarray(cp.asarray(roi)),
            thresh=self.nms_thresh)  # （具体需要看NMS的原理以及输入参数的作用）调用非极大值抑制函数，将重复的抑制掉，就可以将筛选后ROIS进行返回。经过NMS处理后Train数据集得到2000个框，Test数据集得到300个框
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside

class AnchorTargetCreator:
    """
    为Faster-RCNN专有的RPN网络提供自我训练的样本，RPN网络正是利用AnchorTargetCreator产生的样本作为数据进行网络的训练和学习的，这样产生的预测anchor的类别和位置才更加精确，anchor变成真正的ROIS需要进行位置修正，而AnchorTargetCreator产生的带标签的样本就是给RPN网络进行训练学习用哒！自我修正自我提高！
    那么AnchorTargetCreator选取样本的标准是什么呢？
    _enumerate_shifted_anchor函数在一幅图上产生了20000多个anchor，而AnchorTargetCreator就是要从20000多个Anchor选出256个用于二分类和所有的位置回归！为预测值提供对应的真实值，选取的规则是：
    1.对于每一个Ground_truth bounding_box 从anchor中选取和它重叠度最高的一个anchor作为样本！
    2 从剩下的anchor中选取和Ground_truth bounding_box重叠度超过0.7的anchor作为样本，注意正样本的数目不能超过128
    3随机的从剩下的样本中选取和gt_bbox重叠度小于0.3的anchor作为负样本，正负样本之和为256
    PS:需要注意的是对于每一个anchor，gt_label要么为1,要么为0,所以这样实现二分类，而计算回归损失时，只有正样本计算损失，负样本不参与计算。
    """
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call(self, bbox, anchor, img_size):
        # anchor:(S,4),S为anchor数
        img_H, img_W = img_size
        n_anchor = len(anchor)  #一般对应20000个左右anchor
        inside_index = _get_inside_index(anchor, img_H, img_W)  #将那些超出图片范围的anchor全部去掉,只保留位于图片内部的序号
        anchor = anchor[inside_index]  #保留位于图片内部的anchor
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)  #筛选出符合条件的正例128个负例128并给它们附上相应的label
        loc = bbox2loc(anchor, bbox[argmax_ious])  #计算每一个anchor与对应bbox求得iou最大的bbox计算偏移量（注意这里是位于图片内部的每一个）
        label = _unmap(label, n_anchor, inside_index, fill=-1)  # 将位于图片内部的框的label对应到所有生成的20000个框中（label原本为所有在图片中的框的）
        loc = _unmap(loc, n_anchor, inside_index, fill=0)  # 将回归的框对应到所有生成的20000个框中（label原本为所有在图片中的框的）
        return loc, label


    def _create_label(self, inside_index, anchor, bbox):
        label = np.empty((len(inside_index, )), dtype=np.int32)  # inside_index为所有在图片范围内的anchor序号
        label.fill(-1)  #全部填充-1
        # 调用_calc_ious（）函数得到每个anchor与哪个bbox的iou最大以及这个iou值、每个bbox与哪个anchor的iou最大(需要体会从行和列取最大值的区别)
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)
        label[max_ious < self.neg_iou_thresh] = 0
        label[gt_argmax_ious] = 1
        label[max_ious >= self.pos_iou_thresh] = 1
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:  #如果选取出来的正样本数多于预设定的正样本数，则随机抛弃，将那些抛弃的样本的label设为-1
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index - n_pos)), replace=False
            )
            label[disable_index] = -1
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index - n_neg)), replace=False
               )
            label[disable_index] = -1
        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]  # 求出每个anchor与哪个bbox的iou最大，以及最大值，max_ious:[1,N]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]  # 求出每个bbox与哪个anchor的iou最大，以及最大值,gt_max_ious:[1,K]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]  # 然后返回最大iou的索引（每个bbox与哪个anchor的iou最大),有K个
        return argmax_ious, max_ious, gt_argmax_ious


#下面是ProposalTargetCreator代码：ProposalCreator产生2000个ROIS，但是这些ROIS并不都用于训练，经过本ProposalTargetCreator的筛选产生128个用于自身的训练
class ProposalTargetCreator(object):  #为2000个rois赋予ground truth！（严格讲挑出128个赋予ground truth！）
#输入：2000个rois、一个batch（一张图）中所有的bbox ground truth（R，4）、对应bbox所包含的label（R，1）（VOC2007来说20类0-19）
#输出：128个sample roi（128，4）、128个gt_roi_loc（128，4）、128个gt_roi_label（128，1）
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn


    def __call__(self, roi, bbox, label, loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):  #因为这些数据是要放入到整个大网络里进行训练的，比如说位置数据，所以要对其位置坐标进行数据增强处理(归一化处理)
        n_bbox, _ = bbox.shape
        roi = np.concatenate((roi, bbox), axis=0) #首先将2000个roi和m个bbox给concatenate了一下成为新的roi（2000+m，4）。
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)  #n_sample = 128,pos_ratio=0.5，round 对传入的数据进行四舍五入
        iou = bbox_iou(roi, bbox) #计算每一个roi与每一个bbox的iou
        gt_assignment = iou.argmax(axis=1) #按行找到最大值，返回最大值对应的序号以及其真正的IOU。返回的是每个roi与**哪个**bbox的最大，以及最大的iou值
        max_iou = iou.max(axis=1) #每个roi与对应bbox最大的iou
        gt_roi_label = label[gt_assignment] + 1 #从1开始的类别序号，给每个类得到真正的label(将0-19变为1-20)
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]  #同样的根据iou的最大值将正负样本找出来，pos_iou_thresh=0.5
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))   #需要保留的roi个数（满足大于pos_iou_thresh条件的roi与64之间较小的一个）
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)  #找出的样本数目过多就随机丢掉一些

        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]  #neg_iou_thresh_hi=0.5，neg_iou_thresh_lo=0.0
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image # #需要保留的roi个数（满足大于0小于neg_iou_thresh_hi条件的roi与64之间较小的一个）
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)  #找出的样本数目过多就随机丢掉一些

        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # 负样本label 设为0
        sample_roi = roi[keep_index]
#那么此时输出的128*4的sample_roi就可以去扔到 RoIHead网络里去进行分类与回归了。同样， RoIHead网络利用这sample_roi+featue为输入，输出是分类（21类）和回归（进一步微调bbox）的预测值，那么分类回归的groud truth就是ProposalTargetCreator输出的gt_roi_label和gt_roi_loc。
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]]) #求这128个样本的groundtruth
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))  #ProposalTargetCreator首次用到了真实的21个类的label,且该类最后对loc进行了归一化处理，所以预测时要进行均值方差处理
        return sample_roi, gt_roi_loc, gt_roi_label
