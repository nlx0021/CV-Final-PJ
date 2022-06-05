# CV-Final-PJ
The Final Project of CV.

# 第一部分———— 行车记录视频语义分割
具体的README内容请移步到Quest_1文件夹下阅读。

# 第二部分———— 

# 第三部分———— Vision Transformer
第三部分Vision Transformer的代码放在Question_3中。
## 数据与环境部署
首先请下载CIFAR-100数据集：cifar-100-python.tar.gz，并置于data文件夹下。

接着请运行以下命令，安装Transformer所需要的库：
```
pip install vit-pytorch
```
## 模型训练
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

## 模型测试
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
