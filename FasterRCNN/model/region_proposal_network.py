import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.utils.bbox_tools import generate_base_anchor
from model.utils.creator_tool import ProposalCreator

class RegionProposalNetwork:
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_base_anchor(
            anchor_scales=anchor_scales, ratios=ratios)  # 调用generate_anchor_base（）函数，生成左上角9个anchor_base
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]  # 9
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)


    def forward(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape  # (batch_size，512,H/16,W/16），其中H，W分别为原图的高和宽
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)  # 在9个base_anchor基础上生成hh*ww*9个anchor，对应到原图坐标

        n_anchor = anchor.shape[0] // (hh * ww)  # hh*ww*9/hh*ww=9
        """卷积共享"""
        h = F.relu(self.conv1(x))  # 512个3x3卷积(512, H/16,W/16),后面都不写batch_size了
        """二分类"""
        rpn_locs = self.loc(h)  # n_anchor（9）*4个1x1卷积，回归坐标偏移量。（9*4，hh,ww）
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # 转换为（n，hh，ww，9*4）后变为（n，hh*ww*9，4）
        """边框回归"""
        rpn_scores = self.score(h)  # n_anchor（9）*2个1x1卷积，回归类别。（9*2，hh,ww）
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()  # 转换为（n，hh，ww，9*2）
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2),
                                       dim=4)  # 计算{Softmax}(x_{i}) = \{exp(x_i)}{\sum_j exp(x_j)}
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  # 得到前景的分类概率
        rpn_fg_scores = rpn_fg_scores.view(n, -1)  # 得到所有anchor的前景分类概率
        rpn_scores = rpn_scores.view(n, -1, 2)  # 得到每一张feature map上所有anchor的网络输出值

        rois = list()
        roi_indices = list()
        for i in range(n):  # n为batch_size数
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)  # 调用ProposalCreator函数， rpn_locs维度（hh*ww*9，4），rpn_fg_scores维度为（hh*ww*9），anchor的维度为（hh*ww*9，4）， img_size的维度为（3，H，W），H和W是经过数据预处理后的。计算（H/16）x(W/16)x9(大概20000)个anchor属于前景的概率，取前12000个并经过NMS得到2000个近似目标框G^的坐标。roi的维度为(2000,4)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)  # rois为所有batch_size的roi
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)  # 按行拼接（即没有batch_size的区分，每一个[]里都是一个anchor的四个坐标）
        roi_indices = np.concatenate(roi_indices,
                                     axis=0)  # 这个 roi_indices在此代码中是多余的，因为我们实现的是batch_siae=1的网络，一个batch只会输入一张图象。如果多张图象的话就需要存储索引以找到对应图像的roi
        return rpn_locs, rpn_scores, rois, roi_indices, anchor  # rpn_locs的维度（hh*ww*9，4），rpn_scores维度为（hh*ww*9，2）， rois的维度为（2000,4），roi_indices用不到，anchor的维度为（hh*ww*9，4）



def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """先来解释一下对于feature map的每一个点产生anchor的思想，正如代码中描述的那样，首先是将特征图放大16倍对应回原图，为什么要放大16倍，
    因为原图是经过4次pooling得到的特征图，所以缩小了16倍，对应于代码的"""

    import numpy as np
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((
        shift_y.ravel(), shift_x.ravel(),
        shift_y.ravel(), shift_x.ravel()
    ), axis=1)
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor

def normal_init(m, mean, stddev, truncated=False):
    """权重初始化"""
    if truncated:
        m.weight.data.normal_().fmod_().mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()