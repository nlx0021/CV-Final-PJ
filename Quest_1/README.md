# 第一部分———— 行车记录视频语义分割
## 环境配置
我们这部分的代码所使用的框架为MMSegmentation。因此，在进行视频的语义分割测试之前，请先按照MMSegmntation的代码解压后放置在此文件夹内的mmsegmentation文件夹下，并且根据其官方指引进行环境部署。
## 数据配置
环境配置完毕之后，我们进行数据的准备。
  
首先，在mmsegmentation文件夹下新建data文件夹和checkpoints文件夹（若已有就不用创建）。前者用于盛放待分割的行车视频数据，后者用于盛放模型的权重。
  
从我们的下载链接中下载行车记录视频。下载链接：https://pan.baidu.com/s/1QRvTBocnz8AuRXiLY4MPfg?pwd=3rsg ，提取码为3rsg。下载后，请将待分割视频放至./mmsegmentation/data文件夹下。
  
根据
