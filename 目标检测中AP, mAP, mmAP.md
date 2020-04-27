<h1>AP, mAP, mmAP</h1>

目标检测**

目标检测（Object Detection）是计算机视觉中非常常见的任务，该任务的输入是一张图像，输出为图像中的所有存在的目标，每个目标都要给出类别信息（是什么？）和位置信息（在哪里？）。这个位置信息通常用一个外接矩形框（俗称bounding box）来表示。

![img](https://pic3.zhimg.com/80/v2-fb17f8e652c5192fa6af2809d3813886_720w.jpg)目标检测

这个任务有一个特点，就是它的输出是非结构化的。也就是说，它的输出具有很强的不确定性。举个例子，图像分类任务的输入也是一张图像，输出为一个标签/类别，代表着这张图的分类，因此分类任务的输出是结构化的，有且仅有一个标签；而目标检测的输出是图像中的所有目标（类别+位置），图像中到底有多少个目标是不确定的。这一特点非常重要，也正因为这一特点，目标检测的性能度量方法要比图像分类任务复杂得多，这在后面的分析中会提到。

**什么样的检测结果是“好”的？**

什么样的检测结果是好的？这个问题看起来很简单，因为我们在看一张图（包含了检测出来的框/分割掩码）时，很容易就能够对比出真实目标和检测结果的“贴合程度”。但是对于计算机视觉任务，我们需要一个明确的数字化的判断标准。（以下均使用目标检测任务作为示例）

下面我们用 GT(类别, 矩形) 来表示一个图中的目标（也就是我们希望找出来的目标，也可称作Ground Truth）,用DT(类别, 矩形)来表示一个我们检测出来的目标（Detection Result）。

首先，一个好的检测结果，至少得把类别搞对。如果类别就错了，位置信息再准确都是白搭，也就是GT(类别)=DT(类别)；其次，DT(矩形)要尽可能地“贴合”GT(矩形)，越“贴合”越好。如何判断这个“贴合程度”呢？最常用的一种评价方式是交集和并集的比值（交并比、Intersection over Union、IoU）。顾名思义，交并比就是DT(矩形)和GT(矩形)的重合部分面积(交集) 与 两者的全部面积（并集，重合的部分不会重复计算面积）。

![img](https://pic3.zhimg.com/80/v2-5b521592a307eed41bccb78b9c99501e_720w.jpg)Intersection over Union（IoU）

这个比值最小为0，也就是两个矩形毫无交集；最大为1，也就是两个矩形完全重合。IoU越大，检测的结果越好。

**评价检测性能之前**

知道了什么样的检测结果是好的，就可以评价检测算法的性能了。但在这之前，先思考一个问题：评价一个检测算法的性能，真的像想象中那么容易吗？

我们都知道，评价一个图像分类结果的性能，只需要看预测类别是否正确即可，在一个数据集上面，我们可以很容易地得出一个平均准确率。可是目标检测的输出目标数量和真实目标数量都是不固定的（前文提到的非结构化的特性），因此评判时要考虑的就不仅是“对错”这么简单了，我们需要考虑的有：如果漏掉了一个目标对性能有多大损伤？如果多检测出了一个目标对性能有多大损伤？如果检测出来的位置信息有所偏差对性能有多大损伤？进一步地，在这么多的检测结果中，总有一些是检测器十分笃定的，有一些是模棱两可的。如果检测器对多检测出来的那个目标本身也不太确定呢？如果检测器最满意最信任的那个检测结果出错了呢？换言之：一个检测结果对性能的影响，是否应该和检测器对它的满意程度（置信度）相关？以及，检测错了一个稀有的物体和检测错了一个常见的物体所带来的性能损伤是否应该相同？......

**正戏来了：mmAP**

刚刚提到的所有问题，mmAP都要一一给出答案。

首先是位置偏差问题。有的使用场景对位置的准确度要求不高，有的则要求精确定位。因此，mmAP先按位置准确度的需求进行划分，设置一组IOU阈值，这组阈值为 (0.5, 0.55, 0.6, ..., 0.9, 0.95),如果DT与GT的IOU超过阈值，则视作检测成功。这样每给定一个阈值就可以计算出一个性能（也就是mAP，后面详述），然后对这些性能取平均（也就是mmAP，后面详述）就是整个检测算法的性能了。

然后是类别平衡问题，这一问题在分类领域非常常见，“将一个白血病患者错分为健康的人“和“将一个健康的人错分为白血病患者“是一样的吗？显然不是，因为白血病患者本身就属于少数，如果一个分类器把所有人都无脑地判断为健康，其正确率就是 健康的人/全部人。这个分类器的正确率很高但是完全失去了判断病患的功能。mmAP为了公平的评价检测器在各个类别上的性能，采用了类别间求平均的方法：先给定一个IOU阈值，然后将所有的GT和DT按照类别先进行划分，用同一类的所有GT和DT计算出一个性能（也就是AP，马上详述），然后对所有类别的性能取平均（mAP），就是检测算法在这个IOU阈值下的性能。

现在我们来看看，给定了一个IOU阈值、并给定了一个类别，如何具体地计算检测的性能。首先，我们要先对所有的检测结果排序，得分越高的排序越靠前，然后依次判断检测是否成功。将排序后的所有结果定义为DTs，所有同类别的真实目标定义为GTs。先依序遍历一遍DTs中的所有DT，每个DT和全部GT都计算一个IOU，如果最大的IOU超过了给定的阈值，那么视为检测成功，算作TP（True Positive），并且最大IOU对应的GT被视为匹配成功；如果该DT与所有GT的IOU都没超过阈值，自然就是FP（False Positive）；同时，每当一个GT被检测成功后，都会从GTs中“被取走”，以免后续的检测结果重复匹配。因此如果有多个检测结果都与同一个GT匹配，那么分数最高的那个会被算为TP，其余均为FP。遍历完成后，我们就知道了所有DTs中，哪些是TP，哪些是FP，而由于被匹配过的GT都会“被取走”，因此GTs中剩下的就是没有被匹配上的FN（False Negative）。以下是为了方便理解的代码（Python），这段代码仅用于理解，效率较低。真实代码请参考MS COCO的官方源码。

![img](https://pic3.zhimg.com/80/v2-605534721f52c16f23297298ff5bcf2a_720w.jpg)TP,FP,FN

有了TP、FP、FN的定义，就可以方便地得出准确率（Precison，P，即所有的检测结果中多少是对的）和召回率（Recall，R，即所有的真实目标中有多少被检测出来了），两者的定义分别为：P = TP / (TP + FP), R = TP / (TP + FN) = TP / len(GTs)。

但是，单纯地用Precision和Recall来评价整个检测器并不公平，因为有的检测任务要求更高的Recall，“错检”几个影响不大；有的检测任务则要求更高的Precision，“漏检”几个影响不大。因此我们需要对Precision和Recall做一个整体的评估，而这个评估就是前文提到的AP（Average Precision），其定义非常简单，对排序好的det结果进行“截取”，依次观察det结果的前1个（也就是只有第一个结果）、前2个、...、前N个，每次观察都能够得到一个对应的P和R，随着观察数量的增大，R一定会变大或不变。因此可以以R为横轴，P为纵轴，将每次的“截取”观察到的P和R画成一个点（R，P）。值得注意的是，当“截取”到的新结果为FP时，因为R没有变化所以并不会有新的（R，P）点诞生。最后利用这些（R，P）点绘制成P-R曲线，定义： ![[公式]](https://www.zhihu.com/equation?tex=AP+%3D+%5Cint_%7B0%7D%5E%7B1%7DPdR) 。通俗点讲，AP就是这个P-R曲线下的面积。AP计算了不同Recall下的Precision，综合性地评价了检测器，并不会对P和R有任何“偏好”，同时，检测分数越高的结果对AP的影响越大，分数越低的对AP的影响越小。

我们再重新过一遍计算方法：给定一组IOU阈值，在每个IOU阈值下面，求所有类别的AP，并将其平均起来，作为这个IOU阈值下的检测性能，称为mAP(比如mAP@0.5就表示IOU阈值为0.5时的mAP)；最后，将所有IOU阈值下的mAP进行平均，就得到了最终的性能评价指标：mmAP。

值得注意的一个细节是：在实际计算时，mmAP并不会把所有检测结果都考虑进来，因为那样总是可以达到Recall=100%，为了更有效地评价检测结果，mmAP只会考虑每张图片的前100个结果。这个数字应该随着数据集变化而改变，比如如果是密集场景数据集，这个数字应该提高。

还有一个细节非常重要：刚才说AP是P-R曲线下的面积，其实这个说法并不准确。实际上在计算AP时，都要对P-R曲线做一次修正，将P值修正为当R>R0时最大的P（R0即为该点对应的R），即 ![[公式]](https://www.zhihu.com/equation?tex=AP%3D%5Cint_%7B0%7D%5E%7B1%7Dmax%28%7B%5Cleft%5C%7B+P%28r%29+%7C++r+%5Cgeq+R+%5Cright%5C%7D%7D%29dR) 。下图即为修正前P-R曲线和修正后P-R曲线。

![img](https://pic3.zhimg.com/80/v2-8a3da2c2d827d784662ceff1e72474ce_720w.jpg)P-R curve

如图，蓝色实曲线为原始P-R曲线、橘色虚曲线为修正后P-R曲线。为什么要修正P-R曲线呢？这是因为评价检测性能时，更应该关心“当R大于某个值时，能达到的最高的P是多少”、而不是“当R等于某个值时，此时的P是多少”。

除了AP、mAP、mmAP之外，还有一个重要的性能是Recall，有时我们也需要关心“检测器能达到的最大Recall是多少？尽管此时的Precision可能非常低”，AR就是度量Recall的。每给定一个IOU阈值和类别，都会得到一个P-R曲线，该曲线P不为0时的最大的R，就称为该IOU阈值下该类别的Recall（其实是“最大Recall”），在类别尺度上平均后，就是该IOU阈值下的AR，通常我们会用AR[0.5:0.95]表示所有IOU阈值下AR的平均值（也就是mAR）。值得注意的是，AR并不是和mmAP同等量级的性能度量指标，因为AP综合考虑了P和R，而AR只是考虑了Recall。计算AR通常是用于帮助我们分析检测器性能特点的。在两个检测结果的mmAP相同时，更高的AR并不意味着更好的检测效果，而仅仅意味着“更扁平的P-R曲线”。



[本文上篇](https://zhuanlan.zhihu.com/p/55575423)**介绍了mmAP这一经典的目标检测评价指标的定义初衷和具体计算方式。下面我们来谈谈mmAP的诸多特点。通过分析我们甚至可以发现某些特点较反直觉，而某些特点可以加以利用达到“涨点”目的……与此同时，mmAP也并不“全能”，它也有一些未考虑到的因素。以下，我们将通过举例的方式来描述mmAP的这些特点以及它的局限。**

**mmAP的几个特点**

先来看一组检测结果。

![img](https://pic4.zhimg.com/80/v2-be4e49519317f633abd296d8cb3b8cf7_720w.jpg)检测结果（蓝色框为Ground Truth，绿色框为检测结果）

凭直觉地看：b、d的检测结果均差于a、c，因为b、d均含有一个多余的FP；而c的结果要优于a，因为其定位更准确。凭直觉对它们的mmAP排序很可能会排出：c>a>b=d的结果。但实际上，这四组结果对应的mmAP如下图。

![img](https://pic2.zhimg.com/80/v2-dac12a072e51d3666165118d88831035_720w.jpg)

实际上，mmAP的排序为：c=d>b>a。为什么呢？

首先回顾一下mmAP的计算方法，mmAP是对每一个IOU阈值计算出一个mAP，然后将其平均得到最终的结果的，因此mAP@0.95和mAP@0.5是同等重要的（本文就暂以0.95和0.5举例，代表两个极端的阈值），虽然b中的两个检测结果其中必有一个为FP，但是在计算mAP@0.5时，分数低但定位更好的结果并不会对mAP产生影响（因为没有分数更低的TP出现了，该FP对P-R曲线毫无影响）；同时，在计算mAP@0.95时，分数高但定位更差的结果变成了FP，但是由于a中在该阈值下根本不存在TP，所以b的mAP仍高于a。在所有阈值下，b的mAP都要好于或等于a，所以会有b的mmAP高于a这样的反直觉的现象产生。

再来看看c为什么大于b，同时会等于d？c和b的区别在于，c不存在定位更差的检测结果，因此在计算mAP@0.5这样的低阈值mAP时，两者完全相同（b中FP为何不会产生负面作用，道理同上文一样），而计算mAP@0.95这样的高阈值mAP时，c优于b，因此综合下来c的性能好于b。而b和d的直观效果一样，为何mmAP不同？因为b和d区别（也是唯一区别）在于，d中定位更好的检测结果的分数更高。这样一来，d的性能就和c一样了，因为那个低分且低性能的FP不会影响结果。

从这几个结果的对比我们可以发现mmAP的几个特点：1.mAP@0.75这样的高阈值mAP提高时，未必是定位的性能变好导致的，也有可能是因为检测器留下了部分本来应该被过滤掉的FP，而碰巧这个FP的定位效果更好，在高阈值mAP上提供了正面价值；2.如果检测器的定位性能更强，那么mmAP一定会有所提高；3.有时FP未必会影响性能，FP是否影响性能，关键要看是否还存在比该FP分数更低的TP。

**如何利用其特点提高mmAP？**

再看一组检测结果。

![img](https://pic1.zhimg.com/80/v2-7462d68015207014900b91b85463d5c4_720w.jpg)检测结果

这一组检测结果更真实一些，更像是性能正常的检测器输出出来的结果。直观地看，我们可以确信g>e，因为其定位效果更好。但是f和h仍然含有多余的FP，并且存在分数小于该FP的TP，那么它们的性能究竟如何呢？

![img](https://pic2.zhimg.com/80/v2-b049feb45d317ddd638c1e59caf8d2b9_720w.jpg)

实际上：g>h>e>f。其实这一组对比是希望分析检测中的一个问题，即“检测器检测出来的结果中，分数高的未必定位更好，如果分数低的检测结果更好，我们应该怎么办呢？”。其中e的做法是最传统的--直接抑制掉（抑制方法为NMS），而f中的做法则是选择保留，很明显，保留之后的性能变差了，这是因为保留之后必然会存在一个FP，而这个FP会对P-R曲线的后半段产生巨大影响；g的做法比较直观--保留那个分数低但是定位更好的，但是如何确定哪个是定位更好的结果呢？最简单的方法就是“预测定位效果”，IOU-Net就是一个可以用于解决这个问题的研究工作，通过预测检测结果与GT的IOU来判断哪个检测结果的定位效果更好；h的做法非常有意思--将那个应该被抑制掉的检测结果留下来，但是并不是像f那样直接保留，而是给它打了一个很低的分数再保留，这样一来：在计算低阈值mAP时，就不存在比这个FP分数更低的TP了，而在计算高阈值mAP时，它又会对性能有所帮助，所以综合下来，h中的操作也可以提高mmAP。等等，是否觉得这种重新打分的操作非常熟悉？没错，它就是Soft-NMS。其实，在大数据集上（比如COCO），是无法保证重新打分后的检测结果的分数低于所有其他TP的。但是只要被Soft-NMS重新打分的那些结果的平均性能优于该分数段的检测结果，那么性能就会提升。可以这样理解，那些本身就应该保留的高分检测结果大部分是“王者”或者“钻石”，而那些被重新打分的都是“王者”或者“青铜”，虽然并不都是好的，但是由于重新打分，我们把它们放到了“白银”分段，和那些低分检测结果混在了一起。所以最终它们还是能提高检测性能的。

**mmAP之外 -- 分数密度**

mmAP也并不是全能的，有一个被mmAP忽略掉的因素就是“分数密度”，也就是每个目标的具体得分情况，mmAP在计算时，只考虑所有检测结果的排序，但并不会考虑检测结果的具体分数，两个mmAP完全相同的检测器的得分可能相差很大。比如下图。

![img](https://pic2.zhimg.com/80/v2-729e76f7f32a8953db98b8cc142f0c91_720w.jpg)检测结果（红色表示FP）

两者的mmAP完全相同，但是FP的分值并不相同，左边的两个FP分别为0.87和0.86，而右边的两个FP为0.25和0.35。在实际使用时，右侧的检测器我们可以将分数阈值设置为0.35到0.88之间的任何一个数字，因为TP和FP之间有很大的分数差距；而对于左边的检测器，我们只能将分数阈值设置在0.87到0.88之间。换言之，有着同样的mmAP的检测器，在实际使用时的泛化能力未必相同。这是因为mmAP并没有考虑到如“分数密度”这样的因素。

**mmAP之外 -- P-R trade-off**

mmAP更高意味着检测结果性能更好，但是当给定一个场景时，却未必总是这样。比如在一个智能收银场景，对图像中所有物体都要检测出来并进行分类，此时我们可能更关心accuracy=TP / (TP+FP+FN)，因为此时漏检（FN）和误检（FP）对于性能的损失是相同的。那么拥有更高mmAP的检测器一定拥有更高的acc吗？答案是否定的。

![img](https://pic1.zhimg.com/80/v2-a4f1a46dab68e6eff1fc691f927681c0_720w.jpg)

如图，蓝色曲线和绿色曲线相比，AP明显更低，但是却可以达到更高的acc。

这个例子并不是用来证明mmAP不够好。实际上之所以有这样的现象，是因为：mmAP是用来评价检测算法的，而acc是用来评价具体场景下的检测器的。检测算法的mmAP更高，那么它在综合所有任务场景上来看就会有更好的性能。但是当我们有一个确定的场景时，mmAP就会因为“考虑得太全面”而不那么适用了，此时我们应该寻找一个其他的评价指标来衡量检测器的性能，这个指标需要考虑很多因素，比如如何在P-R重要性之间进行trade-off（取舍）。

**总结**

**mmAP是目标检测领域非常重要的性能评价指标，它综合考虑了许多因素，比如类别间的均衡、定位精度的需求等。mmAP也有很多特点，比如“FP未必就是没用的”……不过，虽然mmAP很经典，但它也有体现不出来的要素，比如“分数密度“等。总而言之，mmAP是非常值得深入思考和研究的，辩证地看待它、使用它，才能由表及里的设计出优秀的目标检测算法。**





# IOU

IOU(Intersection over Union):预测位置信息bbox(bounding box)与实际位置信息gt(Ground Truth)之间交集和并集的比值

# 评估指标

- 精确度Precision = TP / (TP + FP)
  - 预测的bbox中预测正确的数量
- 召回率Recall = TP / (TP + FN) = TP / len(gt)
  - ground-true box中被正确预测的数量
- FP:误检 | FN:漏检
- 评价目标检测的性能不能只用精确度或者召回率,一定要考虑到误检和漏检两个场景

# mAP

- mAP是对各目标类计算得到的AP取平均值；
- 设置一组IOU阈值，这组阈值为 (0.5, 0.55, 0.6, …, 0.9, 0.95),单个阈值下可以计算出一个性能mAP,每个阈值取性能平均则为mmAP

## 算法过程

### 举个例子

一个例子有助于更好地理解插值平均精度的概念。考虑下面的检测：
![samples_mAP](https://luckmoonlight.github.io/assets/blogImg/samples_mAP.png)

有7个图像，其中15个gt对象由绿色边界框表示，24个predict对象由红色边界框表示。每个predict对象都具有置信得分score，并由字母（A，B，…，Y）标识。
下表显示了具有相应置信度的边界框。最后一列将检测标识为TP或FP。在此示例中，如果IOU为30％，则考虑TP ，否则为FP。通过查看上面的图像，我们可以粗略地判断检测是TP还是FP。
![table_mAP](https://luckmoonlight.github.io/assets/blogImg/table_mAP.png)

**重叠情况**：在一些图像中，存在多于一个与GT重叠的检测（图像2,3,4,5,6和7）。对于这些情况，采用具有最高IOU的检测，丢弃其他检测。此规则适用于PASCAL VOC 2012度量标准：“例如，单个对象的5个检测（TP）被计为1个正确检测和4个错误检测”。
通过计算累积Acc的TP或FP检测的精度和召回值来绘制PR曲线。为此，首先我们需要通过置信度对检测进行排序，然后我们计算每个累积Acc检测的精度和召回率，如下表所示：
![table_2_map](https://luckmoonlight.github.io/assets/blogImg/table_2_map.png)

- 备注

  ：

  - **Acc TP**为按照置信度降序排列的bbox中，预测正确的累加数和
  - **Acc FP**为按照置信度降序排列的bbox中，预测错误的累加数和
  - Precision = Acc TP /(Acc TP + Acc FP)
  - Recall = Acc TP /15
  - 由公式可以看得出，随着数据集增大，Precision由1逐渐减小，Recall由0逐渐增加至1

绘制精度和召回值，我们有以下Precision x Recall曲线：
![precision_recall_example_1_mAP2010before](https://luckmoonlight.github.io/assets/blogImg/precision_recall_example_1_mAP2010before.png)

#### 11点插值

11点插值平均精度的想法是在一组11个召回级别（0,0.1，…，1）处平均精度。插值精度值是通过采用其**召回值大于其当前调用值的最大精度**获得的，如下所示：
![11_point_interpolated](https://luckmoonlight.github.io/assets/blogImg/11_point_interpolated.png)

#### 计算在所有点中执行的插值

通过内插所有点，平均精度（AP）可以解释为PR曲线的近似AUC。目的是**减少曲线中摆动的影响**。通过应用之前给出的方程，我们可以获得这里将要展示的区域。我们还可以通过查看从最高（0.4666）到0（从右到左看图）的Recall来直观地获得插值精度点，并且当我们减少Recall时，我们收集最高的精度值如下图所示：
![interpolated_precision_v2](https://luckmoonlight.github.io/assets/blogImg/interpolated_precision_v2.png)
看一下上图，我们可以将AUC划分为4个区域（A1，A2，A3和A4）：
![interpolated_precision-AUC_v2](https://luckmoonlight.github.io/assets/blogImg/interpolated_precision-AUC_v2.png)
![interpolated_precision](https://luckmoonlight.github.io/assets/blogImg/interpolated_precision.png)

### 过程描述

1.获取每张img的预测框bbox,取前max_detections张大于得分score_threshold的bbox(并不会把所有检测结果都考虑进来，因为那样总是可以达到Recall=100%;只会考虑每张图片的前N个结果。这个数字应该随着数据集变化而改变，比如如果是密集场景数据集，这个数字应该提高)
2.迭代每一个分类，迭代每一张图片并记录真实框的数目gt，迭代每一个预测框bbox
2.计算一个预测框bbox与真实框gt之间的重叠率IOU，获取最大的IOU
3.如果真实框为0，直接将该预测框标记为‘错检’FP
4.如果重叠率IOU大于阈值，且真实框没有被标记，则为TP，并标记真实框bbox(以免重复检测)，否则为FP(错检)。(gt真实框中没有被标记的为FN漏检)
5.每个分类下：TP，FP根据分类得分值进行逆序排序，并进行梯度累加和
6.计算recall = TP / (TP + FN) = TP / len(gt) precision = TP / (TP + FP)
7.计算AP。以R为横轴，P为纵轴，将每次的“截取”观察到的P和R画成一个点（R，P），利用这些（R，P）点绘制成P-R曲线(AP就是这个P-R曲线下的面积AP=∫10PdRAP=∫01PdR )。AP计算了不同Recall下的Precision，综合性地评价了检测器，并不会对P和R有任何“偏好”;实际上在计算AP时，都要对P-R曲线做一次修正，将P值修正为当R>R_0时最大的P（R_0即为该点对应的R），即 AP=∫10max(P(r)|r≥R)dRAP=∫01max(P(r)|r≥R)dR.故得出某个类别下的AP.`ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])` 
![PRCurve](https://luckmoonlight.github.io/assets/blogImg/PRCurve.png)
蓝色实曲线为原始P-R曲线、橘色虚曲线为修正后P-R曲线。为什么要修正P-R曲线呢？这是因为评价检测性能时，更应该关心“当R大于某个值时，能达到的最高的P是多少”、而不是“当R等于某个值时，此时的P是多少”。
8.求所有类别的AP，并将其平均起来，作为这个IOU阈值下的检测性能，称为mAP(比如mAP=0.5就表示IOU阈值为0.5时的mAP)；最后，将所有IOU阈值下的mAP进行平均，就得到了最终的性能评价指标：mmAP

## 代码解析

```python
def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end recall首位拼接上0，1；precision首位拼接上0，0
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope 将precision逆序比较，从大到小排列
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value 求出recall变化的点坐标
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec 求PR曲线下的面积即AP
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    # 初始化数组(numDataSet,numClass)
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    
    with torch.no_grad():
        # 迭代每张图像
        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold 返回index
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                # 得分逆序排序，取前max_detections=280个，返回逆序后的index
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections 选择得分对应的box和label
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1) # (num,6)

                # copy detections to all_detections 将属于该类label的image_detections位置值和得分值存储至all_detections[index][label]中（？,5）
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations 将属于该类label的annotations位置值存储至all_annotations中
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
    generator,
    retinanet,
    iou_threshold=0.7,
    score_threshold=0.5,
    max_detections=280,
    save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """



    # gather all detections and annotations

    all_detections     = _get_detections(generator, retinanet, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)

    average_precisions = {}
    # 迭代每一个分类label
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        # 迭代每一张图片
        for i in range(len(generator)):
            detections           = all_detections[i][label] # 预测框的位置x1y1x2y2
            annotations          = all_annotations[i][label] # 真实框的位置x1y1x2y2
            num_annotations     += annotations.shape[0] #真实框bbox的个数
            detected_annotations = []
            # 迭代每一张图片中预测的边界框
            for d in detections:
                scores = np.append(scores, d[4]) #每张图片中的预测框得分score汇总

                if annotations.shape[0] == 0: # 如果真实值bbox为0,则属于错检FP
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue
                # 计算一个预测框与真实框bbox的重叠率IOU (1,115)
                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1) # 获得重叠率最大的index
                max_overlap         = overlaps[0, assigned_annotation] # 获得最大的重叠率max_overlap
                # 如果最大的重叠率大于阈值,且真实值bbox的index没有被标记detected_annotations,则为TP,否则都为FP
                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)如果预测框数量为0,则ap直接为0
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score 对得分进行逆序排序
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives 按行进行梯度累加和
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
    
    print('\nmAP:')
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        print('{}: {}'.format(label_name, average_precisions[label][0]))
    
    return average_precisions
```





# 1 mAP简介

目标检测里面经常用到的评估标准是：mAP（mean average precision），计算mAP需要涉及到precision 和 recall的计算，mAP，precision，recall的定义含义以及计算方式，网上很多博客都有说明，本文不打算重述。

阅读本文之前，请先仔细阅读如下资料：

- 周志华老师 《机器学习》 模型评估标准一节，主要是precision，recall的计算方式，或者自己网上搜博客
- [多标签图像分类任务的评价方法-mAP](http://blog.sina.com.cn/s/blog_9db078090102whzw.html) 通过一个简单的二分类阐述 mAP的含义与计算
- [average precision](https://sanchom.wordpress.com/tag/average-precision/) 几种不同形式 AP 的计算方式与异同

以博客 [多标签图像分类任务的评价方法-mAP](http://blog.sina.com.cn/s/blog_9db078090102whzw.html) 中的数据为例，下面是这个二分类问题的P-R曲线（precision-recall curve），P-R曲线下面与x轴围成的面积称为 average precision。

[![1527520824881](https://arleyzhang.github.io/articles/c521a01c/1527520824881-1542379892017.png)](https://arleyzhang.github.io/articles/c521a01c/1527520824881-1542379892017.png)

那么问题就在于如何计算AP，这里很显然可以通过积分来计算

AP=∫10P(r)drAP=∫01P(r)dr

但通常情况下都是使用估算或者插值的方式计算：



**approximated average precision**

AP=∑k=1NP(k)Δr(k)AP=∑k=1NP(k)Δr(k)



- 这个计算方式称为 approximated 形式的，插值计算的方式里面这个是最精确的，每个样本点都参与了计算

- 很显然位于一条竖直线上的点对计算AP没有贡献

- 这里N为数据总量，k为每个样本点的索引， Δr(k)=r(k)−r(k−1)Δr(k)=r(k)−r(k−1)

  [![1527520902790](https://arleyzhang.github.io/articles/c521a01c/1527520902790.png)](https://arleyzhang.github.io/articles/c521a01c/1527520902790.png)

**Interpolated average precision**

这是一种插值计算方式：

Pinterp(k)=maxk^≥kP(k^)Pinterp(k)=maxk^≥kP(k^)





∑k=1NPinterp(k)Δr(k)∑k=1NPinterp(k)Δr(k)



- k 为每一个样本点的索引，参与计算的是所有样本点
- Pinterp(k)Pinterp(k) 取第 k 个样本点之后的样本中的最大值
- 这种方式不常用，所以不画图了

插值方式进一步演变：

Pinterp(k)=maxk^≥kP(k^)Pinterp(k)=maxk^≥kP(k^)





∑k=1KPinterp(k)Δr(k)∑k=1KPinterp(k)Δr(k)



- 这是通常意义上的 Interpolated 形式的 AP，这种形式使用的是比较多的，因为这个式子跟上面提到的计算方式在最终的计算结果上来说是一样的，上面那个式子的曲线跟这里图中阴影部分的外部轮廓是一样的

- 当一组数据中的正样本有K个时，那么recall的阈值也有K个，k代表阈值索引，参与计算的只有K个阈值所对应的样本点

- Pinterp(k)Pinterp(k) 取第 k 个阈值所对应的样本点之后的样本中的最大值

  [![1527520981741](https://arleyzhang.github.io/articles/c521a01c/1527520981741.png)](https://arleyzhang.github.io/articles/c521a01c/1527520981741.png)

再进一步演变：

Pinterp(k)=maxr(k^)≥R(k)P(k^)R∈{0,0.1,0.2,…,1.0}Pinterp(k)=maxr(k^)≥R(k)P(k^)R∈{0,0.1,0.2,…,1.0}





∑k=1KPinterp(k)Δr(k)∑k=1KPinterp(k)Δr(k)



- 这是通常意义上的 11points_Interpolated 形式的 AP，选取固定的 {0,0.1,0.2,…,1.0}{0,0.1,0.2,…,1.0} 11个阈值，这个在PASCAL2007中有使用

- 这里因为参与计算的只有11个点，所以 K=11，称为11points_Interpolated，k为阈值索引

- Pinterp(k)Pinterp(k) 取第 k 个阈值所对应的样本点之后的样本中的最大值，只不过这里的阈值被限定在了 {0,0.1,0.2,…,1.0}{0,0.1,0.2,…,1.0} 范围内。

  [![1527521007943](https://arleyzhang.github.io/articles/c521a01c/1527521007943.png)](https://arleyzhang.github.io/articles/c521a01c/1527521007943.png)

从曲线上看，真实 AP< approximated AP < Interpolated AP

11-points Interpolated AP 可能大也可能小，当数据量很多的时候会接近于 Interpolated AP

前面的公式中计算AP时都是对PR曲线的面积估计，然后我看到PASCAL的论文里给出的公式就更加简单粗暴了，如下：

AP=111∑r∈{0,0.1,0.2,…,1.0}Pintep(r)AP=111∑r∈{0,0.1,0.2,…,1.0}Pintep(r)

```xml

```



Pinterp(r)=MAXr^:r^≥rP(r^)Pinterp(r)=MAXr^:r^≥rP(r^)



直接计算11个阈值处的precision的平均值。

不过我看 Itroduction to Modern Information（中译本：王斌《信息检索导论》）一书中也是直接计算平均值的。

对于Interpolated 形式的 AP，因为recall的阈值变化是等差的（或者recall轴是等分的），所以计算面积和直接计算平均值结果是一样的，

对于11points_Interpolated 来说，虽然recall的阈值也是等差的，但是11points计算平均值时会把recall=0那一点的precision算进去，但实际上那一点是人为添加的，所以计算面积和直接计算平均值会有略微差异。

实际上这是一个极限问题，如果recall轴等分且不考虑初始点，那么计算面积和均值的结果是一样的；如果不等分，只有当分割recall无限多的时候，二者无限趋近，这是一个数学问题。

第 4 节的代码包含这两种计算方式，可以用来验证。

以上几种不同形式的 AP 在第4节会有简单的代码实现。

# 2 PASCAL数据集mAP计算方式

> 一定要先看这个博客 [多标签图像分类任务的评价方法-mAP](http://blog.sina.com.cn/s/blog_9db078090102whzw.html) 。

PASCAL VOC最终的检测结构是如下这种格式的：

比如comp3_det_test_car.txt:

```
000004 0.702732 89 112 516 466
000006 0.870849 373 168 488 229
000006 0.852346 407 157 500 213
000006 0.914587 2 161 55 221
000008 0.532489 175 184 232 201
```

每一行依次为 ：

```
<image identifier> <confidence> <left> <top> <right> <bottom>
```

每一行都是一个bounding box，后面四个数定义了检测出的bounding box的左上角点和右下角点的坐标。

在计算mAP时，如果按照二分类问题理解，那么每一行都应该对应一个标签，这个标签可以通过ground truth计算出来。

但是如果严格按照 ground truth 的坐标来判断这个bounding box是否正确，那么这个标准就太严格了，因为这是属于像素级别的检测，所以PASCAL中规定当一个bounding box与ground truth的 IOU>0.5 时就认为这个框是正样本，标记为1；否则标记为0。这样一来每个bounding box都有个得分，也有一个标签，这时你可以认为前面的文件是这样的，后面多了一个标签项：

```
000004 0.702732 89 112 516 466 1
000006 0.870849 373 168 488 229 0
000006 0.852346 407 157 500 213 1
000006 0.914587 2 161 55 221 0
000008 0.532489 175 184 232 201 1
```

进而你可以认为是这样的，后面的标签实际上是通过坐标计算出来的

```
000004 0.702732  1
000006 0.870849  0
000006 0.852346  1
000006 0.914587  0
000008 0.532489  1
```

这样一来就可以根据前面博客中的二分类方法计算AP了。但这是某一个类别的，将所有类别的都计算出来，再做平均即可得到mAP.

# 3 COCO数据集AP计算方式

COCO数据集里的评估标准比PASCAL 严格许多

COCO检测出的结果是json文件格式，比如下面的：

```
[
   		{
            "image_id": 139,
            "category_id": 1,
            "bbox": [
                431.23001,
                164.85001,
                42.580002,
                124.79
            ],
            "score": 0.16355941
     	}, 
    ……
    ……
]
```

我们还是按照前面的形式来便于理解：

```
000004 0.702732 89 112 516 466
000006 0.870849 373 168 488 229
000006 0.852346 407 157 500 213
000006 0.914587 2 161 55 221
000008 0.532489 175 184 232 201
```

前面提到可以使用IOU来计算出一个标签，PASCAL用的是 IOU>0.5即认为是正样本，但是COCO要求IOU阈值在[0.5, 0.95]区间内每隔0.05取一次，这样就可以计算出10个类似于PASCAL的mAP，然后这10个还要再做平均，即为最后的AP，COCO中并不将AP与mAP做区分，许多论文中的写法是 AP@[0.5:0.95]。而COCO中的 AP@0.5 与PASCAL 中的mAP是一样的。