# 第一部分———— 行车记录视频语义分割
## 环境配置
我们这部分的代码所使用的框架为MMSegmentation。因此，在进行视频的语义分割测试之前，请先按照MMSegmntation的代码解压后放置在此文件夹内的mmsegmentation文件夹下，并且根据其官方指引进行环境部署。
## 数据配置
环境配置完毕之后，我们进行数据的准备。
  
首先，在mmsegmentation文件夹下新建data文件夹和checkpoints文件夹（若已有就不用创建）。前者用于盛放待分割的行车视频数据，后者用于盛放模型的权重。
  
从我们的下载链接中下载行车记录视频。下载链接：https://pan.baidu.com/s/1QRvTBocnz8AuRXiLY4MPfg?pwd=3rsg ，提取码为3rsg。下载后，请将待分割视频放至./mmsegmentation/data文件夹下。
  
我们使用的是SegFormer模型。在进行inference之前，请先下载权重模型至./mmsegmentation/checkpoints文件夹下。下载的链接可以在mmsegmentation的github repo中找到：在其官方仓库中进入configs下的segformer文件夹，里面的README.md文件中记录了模型参数的下载链接，请选择一个合适的进行下载。我们所选用的模型为segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth。
  
## 进行结果测试
准备好后，设置工作目录为./mmsegmentation，运行以下命令：
  
```
python demo/video_demo.py data/$待测试视频的文件名$ configs/segformer/$所选用的配置文件$ checkpoints/$所对应的模型参数文件$ --output-file $输出的文件路径$ --opacity $不透明度$
```
