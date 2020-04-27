作者：刘恒 
时间： 2020/4/5
参考：simple_pytorch_faster_rcnn， 三年一梦

<h3>1.初步理解</h3>

<h5>(1).Faster Rcnn编程理解

<img src='S:\CV\Detection\FasterRCNN\imgs\1.jpg'></img>

- Dataset：数据，提供符合要求的数据格式（目前常用数据集是VOC和COCO）
- Extractor： 利用CNN提取图片特征`features`（原始论文用的是ZF和VGG16，后来人们又用ResNet101）
- RPN(*Region Proposal Network):*  负责提供候选区域`rois`（每张图给出大概2000个候选框）
- RoIHead： 负责对`rois`分类和微调。对RPN找出的`rois`，判断它是否包含目标，并修正框的位置和座标

<h5>(2).Faster R-CNN整体的流程可以分为三步：

- 提特征： 图片（`img`）经过预训练的网络（`Extractor`），提取到了图片的特征（`feature`）
- Region Proposal： 利用提取的特征（`feature`），经过RPN网络，找出一定数量的`rois`（region of interests）。
- 分类与回归：将`rois`和图像特征`features`，输入到`RoIHead`，对这些`rois`进行分类，判断都属于什么类别，同时对这些`rois`的位置进行微调。

### **2**.详细实现

<h5>(1).数据预处理

对与每张图片，需要进行如下数据处理：

- 图片进行缩放，使得长边小于等于1000，短边小于等于600（至少有一个等于）。
- 对相应的bounding boxes 也也进行同等尺度的缩放。
- 对于Caffe 的VGG16 预训练模型，需要图片位于0-255，BGR格式，并减去一个均值，使得图片像素的均值为0。

最后返回四个值供模型训练：

- images ： 3×H×W ，BGR三通道，宽W，高H
- bboxes： 4×K ,   K个bounding boxes，每个bounding box的左上角和右下角的座标，形如（Y_min,X_min, Y_max,X_max）,第Y行，第X列。
- labels：K， 对应K个bounding boxes的label（对于VOC取值范围为[0-19]）
- scale: 缩放的倍数, 原图H' ×W'被resize到了HxW（scale=H/H' ）

需要注意的是，目前大多数Faster R-CNN实现都只支持batch-size=1的训练（[这个](https://zhuanlan.zhihu.com/github.com/jwyang/faster-rcnn.pytorch) 和[这个](https://link.zhihu.com/?target=https%3A//github.com/precedenceguo/mx-rcnn)实现支持batch_size>1）。

<h5>(2).特征提取(Extractor)</h5>

Extractor使用的是预训练好的模型提取图片的特征。论文中主要使用的是Caffe的预训练模型VGG16。修改如下图所示：为了节省显存，前四层卷积层的学习率设为0。Conv5_3的输出作为图片特征（feature）。conv5_3相比于输入，下采样了16倍，也就是说输入的图片尺寸为3×H×W，那么`feature`的尺寸就是C×(H/16)×(W/16)。VGG最后的三层全连接层的前两层，一般用来初始化RoIHead的部分参数，这个我们稍后再讲。总之，一张图片，经过extractor之后，会得到一个C×(H/16)×(W/16)的feature map。

<img src='S:\CV\Detection\FasterRCNN\imgs\2.jpg'></img>

<h5><font color='red'>(3).RPN网络（RegionProposalNetwork） <strong>划重点</strong><font></h5>

Faster R-CNN最突出的贡献就在于提出了Region Proposal Network（RPN）代替了Selective Search，从而将候选区域提取的时间开销几乎降为0（2s -> 0.01s）。

注：提一下SSD使用了多尺度feature_map代替RPN

<img src='S:\CV\Detection\FasterRCNN\imgs\3.png'></img>

	首先初始化网络的结构：特征（N，512，h，w）输入进来（原图像的大小：16*h，16*w），首先是加pad的512个3*3大小卷积核，输出仍为（N，512，h，w）。然后左右两边各有一个1*1卷积。左路为18个1*1卷积，输出为（N，18，h，w），即所有anchor的0-1类别概率（h*w约为2400，h*w*9约为20000）。右路为36个1*1卷积，输出为（N，36，h，w），即所有anchor的回归位置参数。

<font color='blue'>(3.1)Anchor(FasterRcnn的一个重点)</font>
	在RPN中，作者提出了anchor。Anchor是大小和尺寸固定的候选框。论文中用到的anchor有三种尺寸和三种比例，如下图所示，三种尺寸分别是小（蓝128）中（红256）大（绿512），三个比例分别是1:1，1:2，2:1。3×3的组合总共有9种anchor。

<img src='S:\CV\Detection\FasterRCNN\imgs\4.jpg'></img>

然后用这9种anchor在特征图（`feature`）左右上下移动，每一个特征图上的点都有9个anchor，最终生成了 (H/16)× (W/16)×9个`anchor`. 对于一个512×62×37的feature map，有 62×37×9~ 20000个anchor。 也就是对一张图片，有20000个左右的anchor。这种做法很像是暴力穷举，20000多个anchor，哪怕是蒙也能够把绝大多数的ground truth bounding boxes蒙中。

<font color='blue'>(3.2)训练RPN</font>

​		<font color='green'>总体架构见上面</font>

anchor的数量和feature map相关，不同的feature map对应的anchor数量也不一样。RPN在`Extractor`输出的feature maps的基础之上，先增加了一个卷积（用来语义空间转换），然后利用两个1x1的卷积分别进行<font color='red'>二分类（判断是背景还是前景）和位置回归</font>。进行分类的卷积核通道数为9×2（9个anchor，每个anchor二分类，使用交叉熵损失），进行回归的卷积核通道数为9×4（9个anchor，每个anchor有4个位置参数）。RPN是一个全卷积网络（FCN），这样对输入图片的尺寸就没有要求了。

接下来RPN做的事情就是利用（`AnchorTargetCreator`）将20000多个候选的anchor选出256个anchor进行分类和回归位置。选择过程如下：

- 对于每一个ground truth bounding box (`gt_bbox`)，选择和它重叠度（IoU）最高的一个anchor作为正样本

- 对于剩下的anchor，从中选择和任意一个`gt_bbox`重叠度超过0.7的anchor，作为正样本，正样本的数目不超过128个。

- 随机选择和`gt_bbox`重叠度小于0.3的anchor作为负样本。负样本和正样本的总数为256。

  <font color='red'>对于每个anchor, gt_label 要么为1（前景），要么为0（背景），而gt_loc则是由4个位置参数(tx,ty,tw,th)组成，这样比直接回归座标更好。计算分类损失用的是交叉熵损失，而计算回归损失用的是Smooth_l1_loss. 在计算回归损失的时候，只计算正样本（前景）的损失，不计算负样本的位置损失。</font>

<h4><font color='green'>boundingbox回归补充</font></h4>

 

​	目的是提高定位表现。在DPM与RCNN中均有运用。

 

​	1) RCNN版本：

 

​        	在RCNN中，利用**class-specific**（特定类别）的bounding box regressor。也即每一个类别学一个回归器，然后对该类的bbox预测结果进行进一步微调。注意在回归的时候要将bbox坐标(左上右下)转为中心点(x,y)与宽高(w,h)。对于bbox的预测结果P和gt_bbox Q来说我们学要学一个变换，使得这个变换可以将P映射到一个新的位置，使其尽可能逼近gt_bbox。与Faster-RCNN不同之处是这个变换的参数组为：

 

![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502165624816-523645747.png)

 

​	这四个参数都是特征的函数，前两个体现为bbox的中心尺度不变性，后两个体现为体现为bbox宽高的对数空间转换。学到这四个参数（函数）后，就可以将P映射到G', 使得G'尽量逼近ground truth G：

 

​                            ![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502192124694-1375599502.png)                                                                                                                                                        	1)

 

​	那么这个参数组![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502192527666-1124302883.png)是怎么得到的呢？它是关于候选框P的pool5 特征的函数。由pool5出来的候选框P的特征我们定义为![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502192836255-136172309.png)，那么我们有：

 

​                            ![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502192924029-1708474977.png)                                                                                                                                                          2)

 

​	其中W就是可学习的参数。也即我们要学习的参数组![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502192527666-1124302883.png)等价于W与特征的乘积。那么回归的目标参数组是什么呢？就是上面四个式子中的逆过程：

 

​                           ![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502193257593-450410677.png)

 

　　　　  　　　![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502193346617-1093430512.png)                                                                                                                                                            3)

 

​	![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502193448887-751414198.png)就是回归的目标参数组。即我们希望对于不同类别分别学一个W，使得对每个类别的候选框在pool5提到的特征与W乘积后可以尽可能的逼近![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502193448887-751414198.png)。很清楚，最小二乘岭回归目标函数：

 

​                            ![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502193911647-2001901247.png)                                                                                                               4)

 

​	因为是岭回归，所以有一个关于W的L2惩罚项，RCNN论文里给的惩罚因子lambda=1000。还有就是这个回归数据对(P,G)不是随便选的，预测的P应该离至少一个ground truth G很近，这样学出来的参数才有意义。近的度量是P、G的IOU>0.6。

 

​	可以看到RCNN的每一个proposal都要经过一次特征提取的过程，这样效率很低，而后续的Fast\Faster-RCNN都是对一张图的feature map上的区域进行bounding box回归。

 

​                       

 

​	2) Faster RCNN版本：

 

​	较于RCNN,主要有两点不同。

 

​	首先，特征不同。RCNN中，回归的特征是每个proposal的经过pool5后的特征，而Faster-RCNN是在整张图的feature map上以3x*3大小的卷积不断滑过，每个3x*3大小的feature map对应于9个anchor。之后是两个并行1x*1的卷积缩小特征通道为4*x9（9个abchor的四个坐标）和2x9（9个anchor的0-1类别），分别用来做回归与分类。这也是RPN网络的工作之一。RPN网络也是Faster-RCNN的主要优势。

 

​	其次，回归器数目与回归目标函数不同。在Faster-RCNN中不再是**class-specific**，而是9个回归器。因为feature map上的每个点对应有9个anchor。这9个anchor对应了9种不同的尺度和宽高比。每个回归器只针对1种尺度与宽高比。所以虽然Faster-RCNN中给出的候选框是9种anchor，但是经过多次回归它可以预测出各种大小形状的bounding box，这也归功于anchor的设计。至于回归损失函数，首先看一下预测和目标公式：

 

​                           ![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502202656349-779618912.png)

 

​	其中x,y,w,h分别为bbox的中心点坐标，宽与高。![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502202925986-1399556423.png)分别是预测box、anchor box、真实box。计算类似于RCNN，前两行是预测的box关于anchor的offset与scales，后两行是真实box与anchor的offset与scales。那回归的目的很明显，即使得![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502204846928-1926544361.png)尽可能相近。回归损失函数利用的是Fast-RCNN中定义的smooth L1函数，对外点更不敏感：(v==t*)

 

​                         ![img](https://images2018.cnblogs.com/blog/1055519/201805/1055519-20180502205213482-1358000377.png)

 

​	损失函数优化权重W，使得测试时bbox经过W运算后可以得到一个较好的offsets与scales，利用这个offsets与scales可在原预测bbox上微调，得到更好的预测结果

<font color='blue'>(3.3)RPN网络生成ROIS</font>

RPN在自身训练的同时，还会提供RoIs（region of interests）给Fast RCNN（RoIHead）作为训练样本。RPN生成RoIs的过程(`ProposalCreator`)如下：

- 对于每张图片，利用它的feature map， 计算 (H/16)× (W/16)×9（大概20000）个anchor属于前景的概率，以及对应的位置参数。<font color='red'>(计算前面提过)</font>
- 选取概率较大的12000个anchor
- 利用回归的位置参数，修正这12000个anchor的位置，得到RoIs
- 利用非极大值（(Non-maximum suppression, NMS）抑制，选出概率最大的2000个RoIs

注意：在inference的时候，为了提高处理速度，12000和2000分别变为6000和300.

注意：这部分的操作不需要进行反向传播，因此可以利用numpy/tensor实现。

RPN的输出：RoIs（形如2000×4或者300×4的tensor）

<h5>(4).RoIHead/Fast R-CNN</h5>

<font color='red'>(4.1)网络结构</font>

RPN只是给出了2000个候选框，RoI Head在给出的2000候选框之上继续进行分类<font color='red'>(21分类)</font>和位置参数的回归。

<img src='S:\CV\Detection\FasterRCNN\imgs\5.png'></img>

由于RoIs给出的2000个候选框，分别对应feature map不同大小的区域。首先利用`ProposalTargetCreator` 挑选出128个sample_rois, 然后使用了RoIPooling 将这些不同尺寸的区域全部pooling到同一个尺度（7×7）上。下图就是一个例子，对于feature map上两个不同尺度的RoI，经过RoIPooling之后，最后得到了3×3的feature map.

![img](https://pic4.zhimg.com/80/v2-d9eb14da175f7ae2ed6b6d77f8993207_720w.jpg)RoIPooling

RoI Pooling 是一种特殊的Pooling操作，给定一张图片的Feature map (512×H/16×W/16) ，和128个候选区域的座标（128×4），RoI Pooling将这些区域统一下采样到 （512×7×7），就得到了128×512×7×7的向量。可以看成是一个batch-size=128，通道数为512，7×7的feature map。

<font color='blue'>为什么要pooling成7×7的尺度？</font>

​	是为了能够共享权重。在之前讲过，除了用到VGG前几层的卷积之外，最后的全连接层也可以继续利用。当所有的RoIs都被pooling成（512×7×7）的feature map后，将它reshape 成一个一维的向量，就可以利用VGG16预训练的权重，初始化前两层全连接。最后再接两个全连接层，分别是：

- FC 21 用来分类，预测RoIs属于哪个类别（20个类+背景）
- FC 84 用来回归位置（21个类，每个类都有4个位置参数） 

<font color='red'>(4.2)训练</font>

前面讲过，RPN会产生大约2000个RoIs，这2000个RoIs不是都拿去训练，而是利用`ProposalTargetCreator` 选择128个RoIs用以训练。选择的规则如下：

- RoIs和gt_bboxes 的IoU大于0.5的，选择一些（比如32个）
- 选择 RoIs和gt_bboxes的IoU小于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本

为了便于训练，对选择出的128个RoIs，还对他们的`gt_roi_loc` 进行标准化处理（减去均值除以标准差）

对于分类问题,直接利用交叉熵损失. 而对于位置的回归损失,一样采用Smooth_L1Loss, 只不过只对正样本计算损失.而且是只对正样本中的这个类别4个参数计算损失。举例来说:

- 一个RoI在经过FC 84后会输出一个84维的loc 向量.  如果这个RoI是负样本,则这84维向量不参与计算 L1_Loss

- 如果这个RoI是正样本,属于label K,那么它的第 K×4, K×4+1 ，K×4+2， K×4+3（即四个坐标的Loss） 这4个数参与计算损失，其余的不参与计算损失。

  <img src='S:\CV\Detection\FasterRCNN\imgs\7.png'></img>

<font color='red'>(4.3)生成预测结果</font>

试的时候对所有的RoIs（大概300个左右) 计算概率，并利用位置参数调整预测候选框的位置。然后再用一遍极大值抑制（之前在RPN的`ProposalCreator`用过）。

注意：

- 在RPN的时候，已经对anchor做了一遍NMS，在RCNN测试的时候，还要再做一遍
- 在RPN的时候，已经对anchor的位置做了回归调整，在RCNN阶段还要对RoI再做一遍
- 在RPN阶段分类是二分类，而Fast RCNN阶段是21分类

<h3>3.整体网络架构</h3>

![preview](https://pic3.zhimg.com/v2-7c388ef5376e1057785e2f93b79df0f6_r.jpg)

<img src='S:\CV\Detection\FasterRCNN\imgs\6.png'></img>


<h4><font color='blue'>4.概念对比</font></h4>



在Faster RCNN中有几个概念，容易混淆，或者具有较强的相似性。在此我列出来并做对比，希望对你理解有帮助。

## **4.1 bbox anchor RoI  loc**

**BBox**：全称是bounding box，边界框。其中Ground Truth Bounding Box是每一张图中人工标注的框的位置。一张图中有几个目标，就有几个框(一般小于10个框)。Faster R-CNN的预测结果也可以叫bounding box，不过一般叫 Predict Bounding Box.

**Anchor**：锚  ：是人为选定的具有一定尺度、比例的框。一个feature map的锚的数目有上万个（比如 20000）。

**RoI**：region of interest，候选框。Faster R-CNN之前传统的做法是利用selective search从一张图上大概2000个候选框框。现在利用RPN可以从上万的anchor中找出一定数目更有可能的候选框。在训练RCNN的时候，这个数目是2000，在测试推理阶段，这个数目是300（为了速度）*我个人实验发现RPN生成更多的RoI能得到更高的mAP。*

RoI不是单纯的从anchor中选取一些出来作为候选框，它还会利用回归位置参数，微调anchor的形状和位置。

可以这么理解：在RPN阶段，先穷举生成千上万个anchor，然后利用Ground Truth Bounding Boxes，训练这些anchor，而后从anchor中找出一定数目的候选区域（RoIs）。RoIs在下一阶段用来训练RoIHead，最后生成Predict Bounding Boxes。

loc： bbox，anchor和RoI，本质上都是一个框，可以用四个数（y_min, x_min, y_max, x_max）表示框的位置，即左上角的座标和右下角的座标。这里之所以先写y，再写x是为了数组索引方便，但也需要千万注意不要弄混了。 我在实现的时候，没注意，导致输入到RoIPooling的座标不对，浪费了好长时间。除了用这四个数表示一个座标之外，还可以用（y，x，h，w）表示，即框的中心座标和长宽。在训练中进行位置回归的时候，用的是后一种的表示。

## **4.2 四类损失**

虽然原始论文中用的`4-Step Alternating Training` 即四步交替迭代训练。然而现在github上开源的实现大多是采用*近似*联合训练（`Approximate joint training`），端到端，一步到位，速度更快。

在训练Faster RCNN的时候有四个损失：

- RPN 分类损失：anchor是否为前景（二分类）
- RPN位置回归损失：anchor位置微调
- RoI 分类损失：RoI所属类别（21分类，多了一个类作为背景）
- RoI位置回归损失：继续对RoI位置微调

四个损失相加作为最后的损失，反向传播，更新参数。

## **4.3 三个creator**

在一开始阅读源码的时候，我常常把Faster RCNN中用到的三个`Creator`弄混。

- `AnchorTargetCreator` ： 负责在训练RPN的时候，从上万个anchor中选择一些(比如256)进行训练，以使得正负样本比例大概是1:1. 同时给出训练的位置参数目标。 即返回`gt_rpn_loc`和`gt_rpn_label`。 
- `ProposalTargetCreator`： 负责在训练RoIHead/Fast R-CNN的时候，从RoIs选择一部分(比如128个)用以训练。同时给定训练目标, 返回（`sample_RoI`, `gt_RoI_loc`, `gt_RoI_label`）
- `ProposalCreator`： 在RPN中，从上万个anchor中，选择一定数目（2000或者300），调整大小和位置，生成RoIs，用以Fast R-CNN训练或者测试。

其中`AnchorTargetCreator`和`ProposalTargetCreator`是为了生成训练的目标，只在训练阶段用到，`ProposalCreator`是RPN为Fast R-CNN生成RoIs，在训练和测试阶段都会用到。三个共同点在于他们都不需要考虑反向传播（因此不同框架间可以共享numpy实现）

## **4.4 感受野与scale**

从直观上讲，**感受野**（*receptive field*）就是视觉感受区域的大小。在卷积神经网络中，感受野的定义是卷积神经网络每一层输出的特征图（feature map）上的像素点在原始图像上映射的区域大小。我的理解是，feature map上的某一点`f`对应输入图片中的一个区域，这个区域中的点发生变化，`f`可能随之变化。而这个区域外的其它点不论如何改变，`f`的值都不会受之影响。VGG16的conv5_3的感受野为228，即feature map上每一个点，都包含了原图一个228×228区域的信息。

**Scale**：输入图片的尺寸比上feature map的尺寸。比如输入图片是3×224×224，feature map 是 512×14×14，那么scale就是 14/224=1/16。可以认为feature map中一个点对应输入图片的16个像素。由于相邻的同尺寸、同比例的anchor是在feature map上的距离是一个点，对应到输入图片中就是16个像素。在一定程度上可以认为**anchor的精度为16个像素**。不过还需要考虑原图相比于输入图片又做过缩放（这也是dataset返回的`scale`参数的作用，这个的`scale`指的是原图和输入图片的缩放尺度，和上面的scale不一样）。