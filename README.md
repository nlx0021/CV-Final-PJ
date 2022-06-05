# CV-Final-PJ
The Final Project of CV.

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
  
上面用$$括住的内容由您根据情况来进行指定。我们给出我们运行时的命令来作为一个例子，下面是我们所运行的命令：
  
```
python demo/video_demo.py data/result.mp4 configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth --output-file experiments/seg_result.mp4 --opacity 1
```
  
上述命令运行完后，输出的结果视频将被保存在路径 experiments/seg_result.mp4下（若没有experiments文件夹，请先创建）。
  
注意，最终输出的视频虽然为mp4格式文件，但是其格式编码方式可能与大部分Windows自带的视频播放软件所默认的不同。这可能导致视频播放软件无法解码此文件，导致无法播放。经过测试，我们发现将此文件传到微信后，可以通过微信自带的视频播放器进行播放。这可以预览我们的视频，但若要避免微信进行压缩，请选择合适的视频播放器进行播放。


## Part3：Vision Transformer
第三部分Vision Transformer的代码放在Question_3中。
### 数据与环境部署
首先请下载CIFAR-100数据集：cifar-100-python.tar.gz，并置于data文件夹下。

接着请运行以下命令，安装Transformer所需要的库：
```
pip install vit-pytorch
```
### 模型训练
模型的训练在train.py文件中。以下是运行训练的命令示例：
```
python train.py --experiment experiment_name --lr 5e-3 --epochs_n 100
```
训练时会在tensorboard文件夹中生成一个名字为experiment_name的文件夹，里面记录了tensorboard的event文件，可运行此命令查看：
```
tensorboard --logdir tensorboard/experiment_name
```
本项目已经内置了报告中CNN与ViT的学习曲线，将上述命令中的experiment_name换成compare即可。

另外，训练结束后，与这次实验相关的信息会记录在experiment文件夹中的experiment_name文件夹中，里面包含相关的曲线以及训练信息（文本文件）。
  
训练结束后，模型会保存在model文件夹下。
  
可以调节训练的具体参数，包括设置是否使用cutmix、mixup以及cutout。若想进行设置，请进入train.py文件内修改，请在对应位置进行修改（有注释）。

### 模型测试
若要测试预训练的模型模型，以及获取其参数量，请从第第三题的百度网盘链接中下载vit_weights.pth文件并放在model文件夹下，随后运行以下命令。
```
python test.py
```

若要对model文件夹下自己训练的模型进行测试，请运行以下命令：
```
python test.py --model [model_name].pth
```
其中\[model_name\]在model路径下的模型权重名称。

运行后将会输出测试的结果以及该模型的参数量。
