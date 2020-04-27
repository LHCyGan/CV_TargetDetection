from pprint import pprint

# 设置训练参数
# 通过在命令行中传递参数将覆盖配置项。
class Config:
    # 数据处理参数
    voc_data_dir = ""  # 数据存储路径
    # 图片大小范围
    max_size = 1000
    min_size = 600
    # 处理cpu核心数
    num_workers = 8
    test_num_workers = 8

    # l1_smooth_loss 的 sigma
    rpn_sigma = 3.
    roi_sigma = 1.

    # 设置优化参数
    # 原文 0.0005  but 0.0001 in tf-faster-rcnn
    weight_deacy = 0.0005
    lr_deacy = 0.1
    lr = 0.0001

    # 预设置
    data = 'voc'
    pretained_model = 'vgg16'

    # 设置训练周期
    epochs = 14

    use_Adam = False  # 是否使用Adam优化器
    use_chainer = False  # 尝试匹配所有内容作为链接器
    use_drop = False  # 在 RoIHead 使用dropout
    # debug
    debug_file = ""

    test_num = 10000
    # 模型地址
    load_path = None

    caffe_pretrain = False  # 使用caffe预训练模型
    caffe_pretrain_path = ""

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)


    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}


opt = Config()