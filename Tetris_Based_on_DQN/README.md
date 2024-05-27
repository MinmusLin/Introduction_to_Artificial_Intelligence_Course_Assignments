# Tetris Based on DQN

## 项目名称

Tetris_Based_on_DQN

## 项目简介

Reinforcement Learning Lab: Tetris Game Based on Deep Q-Network.

强化学习实验：基于 Deep Q-Network 的俄罗斯方块游戏。

![](assets/model_3000.gif)

## 项目组成

* `/assets`
存放 `README.md` 文件所需的相关图片资源

* `/models`
模型文件

* `/src`
代码文件

  * `deep_q_network.py`
  `DeepQNetwork` 类

  * `tetris.py`
  `Tetris` 类

* `test.py`
模型测试脚本

* `train.py`
模型训练脚本

## 环境搭建

1. 安装 [NVIDIA GeForce 驱动程序](https://www.nvidia.cn/geforce/drivers)，并检查 GeForce Game Ready 为最新版本

   ![](assets/2024-05-08_18-22-37.png)

2. 在命令提示符输入 `nvidia-smi` 查看驱动版本和 CUDA 版本

   ![](assets/2024-05-08_18-26-10.png)

3. 安装对应版本的 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)，并在命令提示符输入 `nvcc -V` 查看版本信息

   ![](assets/2024-05-08_18-53-38.png)

4. 安装对应版本的 [cuDNN](https://developer.nvidia.com/cudnn-downloads)，将解压后的内容（ `bin`、`include`、`lib` 文件夹 ）移动到 CUDA Toolkit 的安装路径中（默认安装路径为 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4` ），打开 `include` 文件夹中的 `cudnn_version.h` 查看版本信息

   ```c
   #define CUDNN_MAJOR 9
   #define CUDNN_MINOR 1
   #define CUDNN_PATCHLEVEL 1
   ```

5. 将 CUDA Toolkit 安装路径中的 `bin`、`include`、`lib`、`libnvvp` 文件夹添加至系统环境变量（默认安装路径为 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4` ）

   ![](assets/2024-05-08_19-05-57.png)

6. 安装 [Anaconda](https://www.anaconda.com)，设置 Anaconda 环境变量，并在命令提示符输入 `conda --version` 查看版本信息

   ![](assets/2024-05-08_19-10-08.png)

8. 创建 Reinforcement Learning conda 环境，Python 版本 3.9.19

   ```bash
   conda create -n rl python==3.9.19
   ```

9. 激活 Reinforcement Learning conda 环境

   ```bash
   conda activate rl
   ```

10. 安装对应版本的 [PyTorch](https://pytorch.org/get-started/locally)

    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

11. 验证 GPU 可用性，查询当前系统中可用的 NVIDIA GPU 设备

    ```python
    import torch
    torch.cuda.is_available()
    torch.cuda.device_count()
    torch.cuda.current_device()
    torch.cuda.get_device_name(0)
    ```

    ![](assets/2024-05-08_19-36-17.png)

12. 安装 TensorBoard

    ```bash
    pip install tensorboard tensorboardX
    ```

## 项目运行

* 测试模型

  ```bash
  python test.py
  ```

* 训练模型

  ```bash
  python train.py
  ```

* 查看训练数据

  ```bash
  tensorboard --logdir=tensorboard
  ```

## 不同迭代次数表现

<p align="center">
  <img src="assets/model_300.gif" width="19%">
  <img src="assets/model_600.gif" width="19%">
  <img src="assets/model_900.gif" width="19%">
  <img src="assets/model_1200.gif" width="19%">
  <img src="assets/model_1500.gif" width="19%">
  <br>300 / 600 / 900 / 1,200 / 1,500 Iterations<br>
  <img src="assets/model_1800.gif" width="19%">
  <img src="assets/model_2100.gif" width="19%">
  <img src="assets/model_2400.gif" width="19%">
  <img src="assets/model_2700.gif" width="19%">
  <img src="assets/model_3000.gif" width="19%">
  <br>1,800 / 2,100 / 2,400 / 2,700 / 3,000 Iterations<br>
</p>

## 文档更新日期

2024年5月27日