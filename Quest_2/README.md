# 第二部分————基于不同初始化方式的目标检测训练实验
  
## 环境部署
我们在这一部分使用的框架为MMDetection框架。因此，和第一部分类似地，请您先将MMDetection的Github repo代码解压并放置在此文件夹内的mmdetection文件夹下，并且按照MMDetection官方指引来进行环境配置。
  
## 代码部署
进行了上述的环境部署后，请您将我们新添加的一些代码放入到./mmdetection文件夹下：

- 对于实验3（i.e. 使用Mask R-CNN模型的Backbone参数来对Faster R-CNN的Backbone进行初始化），请先下载我们处理后的，只有Backbone部分权重的Mask R-CNN模型权重。下载链接为：https://pan.baidu.com/s/1xNiitmghRynXJPg5VtT8Dw?pwd=tpmo ，提取码为tpmo。里面名为mask_rcnn_r50_fpn_1x_coco_onlybackbone.pth的模型权重即为我们处理后的Mask R-CNN模型权重。请您下载后，将其放置在./checkpoints文件夹下。之后，请将此文件夹拷贝或移动到./mmdetection文件夹下。
- 请您对./experiments/mask_pretrain_refine/faster_rcnn_r50_fpn_1x_voc0712.py进行修改。修改的具体位置为第14至15行，请将checkpoint变量后的路径根据您的情况进行修改，使其正确指向您所下载的我们的Mask R-CNN模型权重。修改完毕后，请您将整个experiments文件夹拷贝或移动到./mmdetection文件夹下。experiments文件夹内包含了各个实验所对应的训练及测试配置文件。
- 请您将此文件夹下的./train.py拷贝或移动到./mmdetection/tools文件夹下。在原本的./mmdetection/tools文件夹下，已有一个train.py，因此我们的这个操作将会覆盖掉此原文件。请不用担心，我们仅仅做了一个小改动，其在大部分情况下并不会影响其它正常的训练/测试。若您感到不放心，可以先对原文件进行备份。


## 数据部署
我们此实验要在VOC数据集上进行训练及测试，因此要先下载数据集。请按照MMDetection官方的指引，在./mmdetection/data文件夹下准备好VOC数据集。

## 根据报告中的配置进行训练实验
我们的报告中总共进行了三个实验。若想重新进行此实验对应的训练，请设置工作目录为./mmdetection，并对应运行以下命令：
  
### 完全随机初始化后进行训练（实验1）
  
请运行命令
  
```
python tools/train.py experiments/no_pretrain_refine/faster_rcnn_r50_fpn_1x_voc0712.py --work-dir experiments/no_pretrain_refine/
```
  
运行后，日志文件（包括tensorboard的event文件）以及模型的checkpoints将会保存到文件夹./mmdetection/experiments/no_pretrain_refine下。

### ImageNet预训练Backbone后进行训练（实验2）
  
请运行命令
  
```
python tools/train.py experiments/image_pretrain_refine/faster_rcnn_r50_fpn_1x_voc0712.py --work-dir experiments/image_pretrain_refine/
```
  
运行后，日志文件（包括tensorboard的event文件）以及模型的checkpoints将会保存到文件夹./mmdetection/experiments/image_pretrain_refine下。

### Mask R-CNN在COCO上的预训练模型进行初始化Backbone后训练（实验3）
  
请运行命令
  
```
python tools/train.py experiments/mask_pretrain_refine/faster_rcnn_r50_fpn_1x_voc0712.py --work-dir experiments/mask_pretrain_refine/
```
  
运行后，日志文件（包括tensorboard的event文件）以及模型的checkpoints将会保存到文件夹./mmdetection/experiments/mask_pretrain_refine下。
  
## 对训练好的模型进行测试
为了方便保存测试结果，请在./mmdetection文件夹下新建一个文件夹test，并在test文件夹下同样新建mask_pretrain_refine，image_pretrain_refine等文件夹。
  
运行下列命令：
```
python tools/test.py experiments/$选用的文件夹$/faster_rcnn_r50_fpn_1x_voc0712.py experiments/$选用的文件夹$/latest.pth --work-dir test/$选用的文件夹$/ --eval mAP
```
  
其中，\text{$选用的文件夹$}可以是no_pretrain_refine、image_pretrain_refine和mask_pretrain_refine，分别对应实验1，2和3的结果。上述命令运行结束后将会输出在测试集上的mAP指标和AP50指标。
  
若想对测试图片的inference结果进行图像保存，可运行以下命令：
  
```
 python tools/test.py experiments/$选用的文件夹$/faster_rcnn_r50_fpn_1x_voc0712.py experiments/$选用的文件夹$/latest.pth --work-dir test/$选用的文件夹$/ --show-dir test/$选用的文件夹$/
```
  
\$选用的文件夹\$的设置同上述相同。上述命令运行后，得到的结果将保存在test/$选用的文件夹$下。
  
## 自定义设置训练参数
最后，若您想自定义修改训练的参数或是其它配置，请在对应文件夹下的配置文件内进行修改。例如，如果您想用更小的学习率重新进行实验2的训练，那么请您打开./mmdetection/experiments/image_pretrain_refine/faster_rcnn_r50_fpn_1x_voc0712.py文件，找到其中关于学习率的设置部分（在226行，设置优化器optimizer处），将对应的lr进行修改。修改后，重新运行上述命令即可。
