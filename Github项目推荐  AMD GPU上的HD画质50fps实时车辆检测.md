## Github项目推荐 | AMD GPU上的HD画质50fps实时车辆检测

AI研习社 [AI开发者](javascript:void(0);) *前天*

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibR92ngrLpk5dPKGX955O5TQiclFPkvBYglwfBwXYySWbjxreg4spJiahOUIw32bVOVrNzVZiclGxdvLw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Real-time Vehicle Detection with 50 HD Frames/sec on an AMD GPU

by Rohit Sharma

**Medium：**https://medium.com/towards-artificial-intelligence/real-time-hd-vehicle-detection-with-amd-rocm-e9c2eea73852

**PDF：****http://t.cn/Ai0cdG8O（百度网盘）**

本项目使用深度学习网络Yolo-V2以高清分辨率（1920x1080）以惊人的50帧/秒的速度检测实时交通中的汽车/公共汽车。项目中使用的模型针对使用MIVisionX工具包在AMD-GPU上的推理性能进行了优化。



**Github项目地址：**

**https://github.com/srohit0/trafficVision/**



MIVisionX工具包是一个综合的计算机视觉和机器智能库，实用程序和应用程序捆绑在一个工具包中。Site：https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/

![1561968532149191.gif](https://mmbiz.qpic.cn/mmbiz_gif/bicdMLzImlibQoaf8xJjuwshq323vFEibgZRerb7lV60Rdn4rg212QmNQDibt759I1mmTsptUz5b12BhDuTVTS7jZg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

## **项目特点**

- 带边界框的车辆检测

- 车辆行驶方向（向上，向下）检测

- 车速估算

- 车型：公共汽车/汽车。

  

## **如何运行**

### **使用模型**

![1561968589234891.jpg](https://mmbiz.qpic.cn/mmbiz_jpg/bicdMLzImlibQoaf8xJjuwshq323vFEibgZQHejPTaYWicqMlDSgrKgcDmQopiaEeF7s5mOpdr6XvH4obTxOiaYhDZTg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

###  

### **Demo**

如果没有提供其他选项，应用程序将启动演示。演示将使用存储在media/ 目录中的视频。



- 
- 
- 
- 
- 
- 
- 
- 

```
% ./main.py('Loaded', 'yoloOpenVX')OK: loaded 22 kernels from libvx_nn.soOK: OpenVX using GPU device#0 (gfx900) [OpenCL 1.2 ] [SvmCaps 0 1]OK: annCreateInference: successfulProcessed a total of  102 framesOK: OpenCL buffer usage: 87771380, 46/46%
```

检测汽车、边界框、车速和置信度分数的YouTube视频：https://youtu.be/YASOovwds_A



### **其他例子**

- #### 录制视频

- 

```
./main.py --video /vid.mp4
```

- #### 交通监测相机的ip

- 

```
./main.py --cam_ip 'http://166.149.104.112:8082/snap.jpg'
```



## **安装** 

### **先决条件**

1. GPU：Radeon Instinct或Vega系列产品，带有ROCm和OpenCL开发套件
2. 安装AMD的MIVisionX工具包：AMD的MIVisionX工具包是一个全面的计算机视觉和机器智能库，实用程序
3. CMake, Caffe
4. 谷歌的Protobuf

###  

### **安装步骤**

#### *1.模型转换*

此步骤为voc数据集下载yolov2-tiny并转换为MIVision的openVX模型。



- 
- 

```
% cd trafficVision/model% bash ./prepareModel.sh
```

有关 models/ 目录中模型转换的先决条件（如caffe）的更多详细信息，敬请查看相关链接。

#### *2. MIVision模型编译*



- 
- 

```
% cd trafficVision% make
```

#### *3.测试App*



- 
- 

```
% cd trafficVision% make test
```

它将显示检测media/ 目录中的所有视频。



## **设计**

本节是开发人员的指南，他们希望将视觉和对象检测模型从其他框架(包括 tensorflow, caffe 或 pytorch.)移植到AMD的Radeon gpu。

### **高层次设计**

![img](https://mmbiz.qpic.cn/mmbiz_jpg/bicdMLzImlibQoaf8xJjuwshq323vFEibgZ4RIjAVkooX9RRnZmFfkXFLRGPVhpQFB9d16TNsu2aq1rB1c41icA2vA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

###  

### **低层次模块**

这些较低层次的模块可以在本项目中找到python模块（文件）或包（目录）。

![1561969274763653.jpg](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

## **开发**

### **模型转换**

建议遵循类似于下面描述的模型转换过程。

![1561969441309589.jpg](https://mmbiz.qpic.cn/mmbiz_jpg/bicdMLzImlibQoaf8xJjuwshq323vFEibgZE1TibyoP4lPph780B3gQDE4yqz9HjDdxoAl4sGfBkRSEqicyKMM8r0eQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### **开发基础**

在开始为推理移植神经网络模型之前，请确保已安装以下基础条件。

![1561969474487050.jpg](https://mmbiz.qpic.cn/mmbiz_jpg/bicdMLzImlibQoaf8xJjuwshq323vFEibgZpU1ibicQW0ucmmbE7H9Luces8MavytsquJicfCRyh87iafIPhAazmIBslw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

##  

## **开发与测试环境**

### 1.硬件

- AMD Ryzen Threadripper 1900X 8核处理器
- 加速器=RadeonInstinct MI25加速器

### 2.软件

- Ubuntu 16.04 LTS OS
- Python 2.7
- MIVisionX 1.7.0
- AMD OpenVX 0.9.9
- GCC 5.4



## **致谢**

## MIVisionX 团队



## **参考项目**

1. yoloV2 paper
2. Tiny Yolo aka Darknet reference network
3. MiVisionX Setup
4. AMD OpenVX
5. Optimization with OpenVX Graphs
6. Measuring Traffic Speed With Deep Learning Object Detection

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibQLia6ULtngClLe1mLFqDZ8Kvf4ZibnjYbaiaMcQfqNh4RE4d3khCClVtg4grEB8p74Mn03Bdre3IQ6Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_jpg/bicdMLzImlibTmT1puDIYXqAsWSSAUVuMQp5H8v6A3XE8ZRJzLWHDGlTgiaxb0XKib22icIkbdwCnSdX8r0yajqibmbA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_jpg/bicdMLzImlibSaXjy7ThIUmG36EvXzsqO4AU08SloYfvibT1pmnKsVj2AAGXOFIrYtIBewvKluDia5sBtnWUkXfUVw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**![img](https://mmbiz.qpic.cn/mmbiz_gif/bicdMLzImlibRaG1nQYOWDDjLntJBsBkAU2iaEA80YQ6o2ElxEM9Ixx2QI6ONUzS03hzQ7BMrpbvv0wbibFOnpI4Zw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)**  点击 **阅读原文**，查看本文更多内容

[阅读原文](https://mp.weixin.qq.com/s?__biz=MjM5ODU3OTIyOA==&mid=2650677868&idx=2&sn=5d562a621dc57ea34b8a49b139a97e3e&chksm=bec21f1f89b596095f40f33f70d731946ad55b1964dbc7544a9fd054242803527a8c1f3f271b&mpshare=1&scene=1&srcid=0806VB0zaehrzfuBHung21JP&sharer_sharetime=1565071959993&sharer_shareid=1e7d9aa9beaa0062e9315bebf7cfec21&key=6eec20dd63b3d5eedad44b51434f7aef68af394d434e97dfa18cb5151ad15bae326d5a7ea4bc0851f7ee1d432cb632b4618be817602d35fdb1544e105f97182c12ee2cce3665804a659b9b27eec3ec67&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=dpBPCKo404lgwW7tqawVQjUItFFLz3dqxRR7V5NBQUuQeXoxb6QvDd46aF%2BafEPt##)








 