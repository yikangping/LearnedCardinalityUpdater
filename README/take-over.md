# 概要
接手此项目时，请根据如下流程搭建环境。

---
# 新建虚拟环境
以anaconda为例。
## 新建一个anaconda环境，用python=3.9
```shell
conda create --name cardi python=3.9
```
## 激活环境
```shell
conda activate cardi
```

---
# 安装库
## 安装依赖
```shell
pip install -r requirement.txt
```
如遇到依赖版本冲突问题/依赖找不到的问题，尝试从requirement.txt中删除依赖，重新安装

## 安装cuda版torch（如有NVIDIA GPU）
参考 https://pytorch.org/get-started/pytorch-2.0/ 进行安装。

检查安装情况：运行`README/verify-requirements/check_cuda.py`。

## 安装其他依赖
```shell
# 安装Jupyter
pip install jupyter

# 安装nflows
pip install nflows

# 进入/FACE/torchquadMy目录，安装重写的torchquad
pip install .
```