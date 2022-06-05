# 第二部分————基于不同初始化方式的目标检测训练实验
  
## 环境部署
我们在这一部分使用的框架为MMDetection框架。因此，和第一部分类似地，请您先将MMDetection的Github repo代码解压并放置在此文件夹内的mmdetection文件夹下，并且按照MMDetection官方指引来进行环境配置。
  
## 代码部署
进行了上述的环境部署后，请您将我们新添加的一些代码放入到./mmdetection文件夹下：

- 对于实验3（i.e. 使用Mask R-CNN模型的Backbone参数来对Faster R-CNN的Backbone进行初始化），请先下载我们处理后的，只有Backbone部分权重的Mask R-CNN模型权重。下载链接为：https://pan.baidu.com/s/1xNiitmghRynXJPg5VtT8Dw?pwd=tpmo ，提取码为tpmo。里面名为mask_rcnn_r50_fpn_1x_coco_onlybackbone.pth的模型权重即为我们处理后的Mask R-CNN模型权重。请您下载后，将其放置在./checkpoints文件夹下。之后，请将此文件夹拷贝或移动到./mmdetection文件夹下。
- 请您将整个experiments文件夹拷贝或移动到./mmdetection文件夹下。里面包含了各个实验所对应的训练及测试配置文件。
- 请您将此文件夹下的./train.py拷贝或移动到./mmdetection/tools文件夹下。在原本的./mmdetection/tools文件夹下，已有一个train.py，因此我们的这个操作将会覆盖掉此原文件。请不用担心，我们仅仅做了一个小改动，其在大部分情况下并不会影响其它正常的训练/测试。若您感到不放心，可以先对原文件进行备份。


## 数据部署
我们此实验要在VOC数据集上进行训练及测试，因此要先下载数据集。请按照MMDetection官方的指引，在./mmdetection/data文件夹下准备好VOC数据集。

## 根据报告中的配置进行训练实验
