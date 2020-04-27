<h1>Mask RCNN</h1>

最近在做一个目标检测项目，用到了Mask RCNN。我仅仅用了50张训练照片，训练了1000步之后进行测试，发现效果好得令人称奇。就这个任务，很久之前用yolo v1训练则很难收敛。不过把它们拿来比当然不公平，但我更想说的是，mask RCNN效果真的很好。

所以这篇文章来详细地总结一下Mask RCNN。

Mask RCNN沿用了Faster RCNN的思想，特征提取采用ResNet-FPN的架构，另外多加了一个Mask预测分支。可见Mask RCNN综合了很多此前优秀的研究成果。为了更好地讲解Mask RCNN，我会先回顾一下几个部分：

1. Faster RCNN
2. ResNet-FPN
3. ResNet-FPN+Fast RCNN

回顾完之后讲ResNet-FPN+Fast RCNN+Mask，实际上就是Mask RCNN。

## 一、Faster RCNN

Faster RCNN是两阶段的目标检测算法，包括阶段一的Region proposal以及阶段二的bounding box回归和分类。用一张图来直观展示Faster RCNN的整个流程：

![img](https://pic1.zhimg.com/80/v2-a2963871090a60678eac1250a38605d8_720w.jpg)

Faster RCNN使用CNN提取图像特征，然后使用region proposal network（RPN）去提取出ROI，然后使用ROI pooling将这些ROI全部变成固定尺寸，再喂给全连接层进行Bounding box回归和分类预测。

这里只是简单地介绍了Faster RCNN前向预测的过程，但Faster RCNN本身的细节非常多，比一阶段的算法复杂度高不少，并非三言两语能说得清。如果对Faster RCNN算法不熟悉，想了解更多的同学可以看这篇文章：[一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)，这是我看过的解释得最清晰的文章。

## 二、ResNet-FPN

多尺度检测在目标检测中变得越来越重要，对小目标的检测尤其如此。现在主流的目标检测方法很多都用到了多尺度的方法，包括最新的yolo v3。Feature Pyramid Network (**FPN**)则是一种精心设计的多尺度检测方法，下面就开始简要介绍FPN。

FPN结构中包括自下而上，自上而下和横向连接三个部分，如下图所示。这种结构可以将各个层级的特征进行融合，使其同时具有强语义信息和强空间信息，在特征学习中算是一把利器了。

![img](https://pic1.zhimg.com/80/v2-15a45ce0641bbb9cd3af36a58cbf1714_720w.jpg)

FPN实际上是一种通用架构，可以结合各种骨架网络使用，比如VGG，ResNet等。Mask RCNN文章中使用了ResNet-FPN网络结构。如下图：

![img](https://pic4.zhimg.com/80/v2-fc500b77472298d7dacdd303f509c68b_720w.jpg)

ResNet-FPN包括3个部分，自下而上连接，自上而下连接和横向连接。下面分别介绍。

**自下而上**

从下到上路径。可以明显看出，其实就是简单的特征提取过程，和传统的没有区别。具体就是将ResNet作为骨架网络，根据feature map的大小分为5个stage。stage2，stage3，stage4和stage5各自最后一层输出conv2，conv3，conv4和conv5分别定义为 ![[公式]](https://www.zhihu.com/equation?tex=C_2%2CC_3%2CC_4%2CC_5) ，他们相对于原始图片的stride是{4,8,16,32}。需要注意的是，考虑到内存原因，stage1的conv1并没有使用。

**自上而下和横向连接**

自上而下是从最高层开始进行上采样，这里的上采样直接使用的是<font color='red'>最近邻上采样</font>，而不是使用反卷积操作，一方面简单，另外一方面可以减少训练参数。横向连接则是将上采样的结果和自底向上生成的相同大小的feature map进行融合。具体就是对 ![[公式]](https://www.zhihu.com/equation?tex=C_2%2CC_3%2CC_4%2CC_5) 中的每一层经过一个conv 1x1操作（1x1卷积用于降低通道数），无激活函数操作，输出通道全部设置为相同的256通道，然后和上采样的feature map进行<font color='red'>加和操作</font>。在融合之后还会再采用3*3的卷积核对已经融合的特征进行处理，目的是<font color='red'>消除上采样的混叠效应（aliasing effect）</font>。

实际上，**上图少绘制了一个分支**：M5经过步长为2的max pooling下采样得到 P6，作者指出使用P6是想得到更大的anchor尺度512×512。但P6是只用在 RPN中用来得到region proposal的，并不会作为后续Fast RCNN的输入。

总结一下，<font color='red'>ResNet-FPN作为RPN输入的feature map是 ![[公式]](https://www.zhihu.com/equation?tex=%5BP2%2CP3%2CP4%2CP5%2CP6%5D) ，而作为后续Fast RCNN的输入则是 ![[公式]](https://www.zhihu.com/equation?tex=%5BP2%2CP3%2CP4%2CP5%5D) 。</font>

## 三、ResNet-FPN+Fast RCNN

![img](https://pic1.zhimg.com/80/v2-18b0db72ed142c8208c0644c8b5a8090_720w.jpg)

将ResNet-FPN和Fast RCNN进行结合，实际上就是Faster RCNN的了，但与最初的Faster RCNN不同的是，FPN产生了特征金字塔 ![[公式]](https://www.zhihu.com/equation?tex=%5BP2%2CP3%2CP4%2CP5%2CP6%5D) ，而并非只是一个feature map。金字塔经过RPN之后会产生很多region proposal。这些region proposal是分别由 ![[公式]](https://www.zhihu.com/equation?tex=P2%2CP3%2CP4%2CP5%2CP6) 经过RPN产生的，但用于输入到Fast RCNN中的是 ![[公式]](https://www.zhihu.com/equation?tex=%5BP2%2CP3%2CP4%2CP5%5D) ，也就是说要在 ![[公式]](https://www.zhihu.com/equation?tex=%5BP2%2CP3%2CP4%2CP5%5D) 中根据region proposal切出ROI进行后续的分类和回归预测。问题来了，我们要选择哪个feature map来切出这些ROI区域呢？实际上，我们会选择最合适的尺度的feature map来切ROI。<font color='blue'>具体来说，我们通过一个公式来决定宽w和高h的ROI到底要从哪个![[公式]](https://www.zhihu.com/equation?tex=P_k) 来切：</font>

![img](https://pic1.zhimg.com/80/v2-939a0ce072c0d8a2c4423d88b54c2680_720w.jpg)

<font color='blue'>这里224表示用于预训练的ImageNet图片的大小。 ![[公式]](https://www.zhihu.com/equation?tex=k_0) 表示面积为 ![[公式]](https://www.zhihu.com/equation?tex=w%C3%97h%3D224%C3%97224) 的ROI所应该在的层级。作者将 ![[公式]](https://www.zhihu.com/equation?tex=k_0) 设置为4，也就是说 ![[公式]](https://www.zhihu.com/equation?tex=w%C3%97h%3D224%C3%97224) 的ROI应该从 ![[公式]](https://www.zhihu.com/equation?tex=P4) 中切出来。假设ROI的scale小于224（比如说是112 * 112）， ![[公式]](https://www.zhihu.com/equation?tex=k+%3D+k_0-1+%3D+4-1+%3D+3) ，就意味着要从更高分辨率的 ![[公式]](https://www.zhihu.com/equation?tex=P_3) 中产生。另外，![[公式]](https://www.zhihu.com/equation?tex=k) 值会做取整处理，防止结果不是整数</font>。

这种做法很合理，<font color='red'>大尺度的ROI要从低分辨率的feature map上切，有利于检测大目标，小尺度的ROI要从高分辨率的feature map上切，有利于检测小目标</font>。

## 四、ResNet-FPN+Fast RCNN+mask

我们再进一步，将<font color='red'>ResNet-FPN+Fast RCNN+mask，则得到了最终的Mask RCNN</font>，如下图：

![img](https://pic1.zhimg.com/80/v2-7a539f4d5f904db3c4559ebe6c9ef49c_720w.jpg)

Mask RCNN的构建很简单，只是在ROI pooling（实际上用到的是，后面会讲到）之后添加卷积层，进行mask预测的任务。

下面总结一下Mask RCNN的网络：

1. 骨干网络ResNet-FPN，用于特征提取，另外，ResNet还可以是：ResNet-50,ResNet-101,ResNetv2-50,ResNetv2-101；
2. 头部网络，包括边界框识别（分类和回归）+mask预测。头部结构见下图：

![img](https://pic1.zhimg.com/80/v2-a500524ae104ae4eaf9a929befe2ba0c_720w.jpg)

## 五、ROI Align

实际上，Mask RCNN中还有一个很重要的改进，就是ROIAlign。<font color='green'>Faster R-CNN存在的问题是：特征图与原始图像是**不对准**的（mis-alignment），所以会影响检测精度。</font>而Mask R-CNN提出了RoIAlign的方法来取代ROI pooling，RoIAlign可以保留大致的空间位置。

为了讲清楚ROI Align，这里先插入两个知识，<font color='red'>双线性插值和ROI pooling。</font>

## 1.双线性插值

在讲双线性插值之前，还得看最简单的线性插值。

**线性插值**

已知数据 ![[公式]](https://www.zhihu.com/equation?tex=%28x_0%2C+y_0%29) 与 ![[公式]](https://www.zhihu.com/equation?tex=%28x_1%2C+y_1%29) ，要计算 ![[公式]](https://www.zhihu.com/equation?tex=%5Bx_0%2C+x_1%5D+) 区间内某一位置 ![[公式]](https://www.zhihu.com/equation?tex=x) 在直线上的 ![[公式]](https://www.zhihu.com/equation?tex=y) 值，如下图所示。

![img](https://pic2.zhimg.com/80/v2-77d26fd67805cdba3965d28e7fe16fe9_720w.jpg)

计算方法很简单，通过斜率相等就可以构建y和x之间的关系，如下：
![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7By+-+y_0%7D%7Bx+-+x_0%7D+%3D+%5Cfrac%7By+-+y_1%7D%7Bx+-+x_1%7D+%3D%3D%3D%3D%3E+y%3D%5Cfrac%7Bx-x_0%7D%7Bx_1-x_0%7Dy_1%2B%5Cfrac%7Bx_1-x%7D%7Bx_1-x_0%7Dy_0)
仔细看就是用 ![[公式]](https://www.zhihu.com/equation?tex=x) 和 ![[公式]](https://www.zhihu.com/equation?tex=x_0) ， ![[公式]](https://www.zhihu.com/equation?tex=x_1) 的距离作为一个权重（除以 ![[公式]](https://www.zhihu.com/equation?tex=x-x_0) 是归一化的作用），用于 ![[公式]](https://www.zhihu.com/equation?tex=y_0) 和 ![[公式]](https://www.zhihu.com/equation?tex=y_1) 的加权。这个思想很重要，因为知道了这个思想，理解双线性插值就非常简单了。

**双线性插值**

<font color='red'>双线性插值本质上就是在两个方向上做线性插值。</font>

![img](https://pic4.zhimg.com/80/v2-d5504827dd6c3cc170cc12185a812407_720w.jpg)

如图，假设我们想得到P点的插值，我们可以先在x方向上，对 ![[公式]](https://www.zhihu.com/equation?tex=Q_%7B11%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=Q_%7B21%7D) 之间做线性插值得到 ![[公式]](https://www.zhihu.com/equation?tex=R_1) ， ![[公式]](https://www.zhihu.com/equation?tex=R_2) 同理可得。然后在y方向上对 ![[公式]](https://www.zhihu.com/equation?tex=R_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=R_2) 进行线性插值就可以得到最终的P。其实知道这个就已经理解了双线性插值的意思了，如果用公式表达则如下（注意 ![[公式]](https://www.zhihu.com/equation?tex=f) 前面的系数看成权重就很好理解了）。

<font color='blue'>首先在 *x* 方向进行线性插值，得到</font>

![img](https://pic4.zhimg.com/80/v2-1eef061ff3d22a42063b38895c10c3b3_720w.jpg)

<font color='blue'>然后在 *y* 方向进行线性插值，得到</font>

![img](https://pic2.zhimg.com/80/v2-bf1b80af5fc32550b602b794cf2c2751_720w.jpg)


<font color='blue'>这样就得到所要的结果</font>> ![[公式]](https://www.zhihu.com/equation?tex=f%28x%2Cy%29)

![img](https://pic1.zhimg.com/80/v2-9d33958a8627a255028a2d5bb48bf768_720w.jpg)



参考：[维基百科：双线性插值](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/wiki/%25E5%258F%258C%25E7%25BA%25BF%25E6%2580%25A7%25E6%258F%2592%25E5%2580%25BC)

## 2.ROIpooling

ROI pooling就不多解释了，直接通过一个例子来形象理解。假设现在我们有一个8x8大小的feature map，我们要在这个feature map上得到ROI，并且进行ROI pooling到2x2大小的输出。

![img](https://pic2.zhimg.com/80/v2-17d3c689270ac638d9aafd90b96149e5_720w.jpg)

假设ROI的bounding box为 ![[公式]](https://www.zhihu.com/equation?tex=%5Bx_1%2Cy_1%2Cx_2%2Cy_2%5D%3D%5B0%2C3%2C7%2C8%5D) 。如图：

![img](https://pic3.zhimg.com/80/v2-b52b7f305f1aaea6febfe0d22a7d23da_720w.jpg)

将它划分为2x2的网格，因为ROI的长宽除以2是不能整除的，所以会出现每个格子大小不一样的情况。

![img](https://pic4.zhimg.com/80/v2-5ba9049300789586c29f78ebc2a5acb7_720w.jpg)

进行max pooling的最终2x2的输出为：

![img](https://pic1.zhimg.com/80/v2-df2414cb2eb3ae6e3bc30b9289cf5c44_720w.jpg)

最后以一张动图形象概括之：

![img](https://pic4.zhimg.com/v2-e55e66c9c21136efa62beacfcb02f9ef_b.jpg)

参考：[Region of interest pooling explained](https://link.zhihu.com/?target=https%3A//blog.deepsense.ai/region-of-interest-pooling-explained/)

## <font color='red'>3. ROI Align</font>

在Faster RCNN中，有<font color='green'>两次整数化的过程：</font>

1. region proposal的x,y,w,h通常是小数，但是为了方便操作会把它整数化。<font color='green'>(BBOX)</font>
2. 将整数化后的边界区域平均分割成 k x k 个单元，对每一个单元的边界进行整数化。<font color='green'>(ROI Pooling)</font>

两次整数化的过程如下图所示：

![img](https://pic3.zhimg.com/80/v2-36e08338390289e0dd88203b4e8ddda2_720w.jpg)

<font color='blue'>事实上，经过上述两次整数化，此时的候选框已经和最开始回归出来的位置有一定的偏差，这个偏差会影响检测或者分割的准确度。在论文里，作者把它总结为“不匹配问题”（misalignment）。</font>

为了解决这个问题，ROI Align方法取消整数化操作，保留了小数，使用以上介绍的双线性插值的方法获得坐标为浮点数的像素点上的图像数值。但在实际操作中，ROI Align并不是简单地补充出候选区域边界上的坐标点，然后进行池化，而是重新进行设计。

下面通过一个例子来讲解ROI Align操作。如下图所示，虚线部分表示feature map，实线表示ROI，这里将ROI切分成2x2的单元格。如果采样点数是4，那我们<font color='red'>1. 首先,将每个单元格子均分成四个小方格（如红色线所示），每个小方格中心就是采样点,这些采样点的坐标通常是浮点数，所以需要对采样点像素进行双线性插值（如四个箭头所示），就可以得到该像素点的值了;</font>2. <font color='red'>然后,对每个单元格内的四个采样点进行maxpooling，就可以得到最终的ROIAlign的结果。</font>

![img](https://pic1.zhimg.com/80/v2-76b8a15c735d560f73718581de34249c_720w.jpg)

需要说明的是，在相关实验中，作者发现将采样点设为4会获得最佳性能，甚至直接设为1在性能上也相差无几。事实上，ROI Align 在遍历取样点的数量上没有ROIPooling那么多，但却可以获得更好的性能，这主要归功于解决了misalignment的问题。

## 六、损失

L1 loss

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cbegin%7Barray%7D%7Blc%7D++L_1%3D%5Cleft%7C+f%28x%29-Y+%5Cright%7C+%5C%5C+L_1%5E%7B%27%7D%3D+%5Cpm+f%5E%7B%27%7D%28x%29+%5Cend%7Barray%7D+%5Cend%7Bequation%7D) 　 ……………………………………………… (2)

Smooth L1 loss

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cbegin%7Barray%7D%7Blc%7D++Smooth%5Cquad+L_1%3D++%5Cbegin%7Barray%7D%7Blc%7D++0.5+x%5E2%2C+%5Cquad%5Cleft%7C+x+%5Cright%7C%3C1+%5C%5C+%5Cleft%7C+x+%5Cright%7C-0.5%2C+%5Cquad+x%3C-1+%5C%2C+or+%5C%2C+x%3E1+%5Cend%7Barray%7D+%5C%5C+%5C%5C+Smooth%5Cquad+L_1%5E%7B%27%7D%3D+%5Cbegin%7Barray%7D%7Blc%7D+++x%2C+%5Cquad+%5Cleft%7C+x+%5Cright%7C%3C1+%5C%5C+-1%2C+%5Cquad+x%3C-1+%5C%5C+1%2C+%5Cquad+x%3E1+%5Cend%7Barray%7D++%5Cend%7Barray%7D+%5Cend%7Bequation%7D)

Mask RCNN定义多任务损失：
![[公式]](https://www.zhihu.com/equation?tex=L%3DL_%7Bcls%7D%2BL_%7Bbox%7D%2BL_%7Bmask%7D)

![img](https://pic1.zhimg.com/80/v2-a12e9c005947651ac7e78be12938a3f4_720w.jpg)

![[公式]](https://www.zhihu.com/equation?tex=L_%7Bcls%7D)和 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Bbox%7D) 与faster rcnn的定义没有区别。<font color='red'>需要具体说明的是 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Bmask%7D) ，假设一共有K个类别，则mask分割分支的输出维度是 ![[公式]](https://www.zhihu.com/equation?tex=K%2Am%2Am) , 对于 ![[公式]](https://www.zhihu.com/equation?tex=m%2Am) 中的每个点，都会输出K个二值Mask（每个类别使用sigmoid输出）。需要注意的是，计算loss的时候，并不是每个类别的sigmoid输出都计算二值交叉熵损失，而是该像素属于哪个类，哪个类的sigmoid输出才要计算损失(如图红色方形所示)。并且在测试的时候，我们是通过分类分支预测的类别来选择相应的mask预测。这样，mask预测和分类预测就彻底解耦了。</font>

这与FCN方法是不同，`FCN是对每个像素进行多类别softmax分类，然后计算交叉熵损失，很明显，这种做法是会造成类间竞争的，而每个类别使用sigmoid输出并计算二值损失（极大似然），可以避免类间竞争`。实验表明，通过这种方法，可以较好地提升性能。

![img](https://img-blog.csdn.net/20180725102401977?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvcml6b25oZWFydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)