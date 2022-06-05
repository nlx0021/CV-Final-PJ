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
  
- 完全随机初始化后进行训练（实验1）。请运行命令```python tools/train.py experiments/no_pretrain_refine/faster_rcnn_r50_fpn_1x_voc0712.py --work-dir experiments/no_pretrain_refine/```。运行后，日志文件（包括tensorboard的event文件）以及模型的checkpoints将会保存到文件夹./mmdetection/experiments/no_pretrain_refine下。
- 
