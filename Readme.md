# LearnedCardinalityUpdater

## Introduction

本课题的主要内容是在数据库查询优化器中的基数估计问题背景中，针对现有的机器学习模型的“静态”问题，面向数据库数据发生更新的场景，设计准确、高效的模型更新算法。

#### 什么是基数估计问题？

$$
给定数据库D，和查询Q，估计查询结果R_D(Q)的大小|R_D(Q)|
$$

基数估计器是数据库查询优化器中一个非常重要的模块，该模块会预测查询语句返回的行数，从而成为查询成本估计的重要依据，指导最优查询计划的选择

注：数据库查询优化器（query optimizer）的相关知识，以及基数估计器（cardinality estimator）在查询优化器中起到的作用，感兴趣可自行查阅

#### 本课题想要解决什么问题场景？

$$
假定有数据库\mathbb{D}，包含k张表\{T_1, T_2, …, T_k\}，每张表包含m个属性\{A_1^t,A_2^t,…,A_m^t\}。\\假设有一个工作负载\mathbb{W}，包含若干SQL语句以及数据更新语句(INSERT/DELETE/UPDATE)，\\其中一条标准的SQL语句定义如下：\\
SELECT\ *\ FROM\ T_1 \Join T_2 \Join T_3 \dots\Join T_n\ WHERE\ F_1\ AND \dots\ AND\ F_d.
$$

我们拟基于现有的学习型基数估计模型，设计自适应模型更新方法，使其能够适应上述环境。

目标：1）最小化SQL语句的基数估计误差；2）自动识别数据漂移（data drifts）并进行自适应更新

#### 关键技术问题

- 数据漂移的自适应识别：如何在数据库进行数据更新之后，准确识别数据漂移
- 模型的高效更新：如何基于更新后的数据，实现模型的增量更新，用尽量小的训练开销，实现更好的训练效果

## Implementation

目前代码基于两个之前工作的开源代码分别对我设计的漂移检测方法和模型更新方法进行了实现，但是***实验效果还有待进行反复的测试***

- data drifts detection：基于Face[1]实现，在FACE目录下测试运行
- model update：基于DDUp[2]实现，在Naru目录下测试运行

#### 如何运行？

- model update：

```shell
cd ./LearnedCardinalityUpdater

# 训练初始模型，dataset参数有两个可用[census(default), forest]，全局统一
# epoch参数默认值20，设定为200一般就会收敛，其余参数设置参考代码，模型保存在./models目录下，origin打头
python Naru/train_model.py --dataset census --epoch 200

# 生成漂移更新数据，并将更新后的数据集保存在./data目录相应数据集文件夹下的permuted_dataset.csv中。
# eval_type参数有两种[estimate(default)，drift]，前者分析现有模型的基数估计准确率，后者生成漂移数据集并检测是否发生了漂移
python Naru/eval_model.py --dataset census --eval_type drift

# 基于permuted_dataset.csv对模型进行增量训练，共有三种更新方法：Finetune、Update（DDUp）、Adapt（our），分别生成FT、update、adapt打头的三个更新模型
# epoch默认20，这里可以设置一个较小的数值检测固定轮数下模型的优化效率，也可以试着训练到收敛为止，这里需要记录一下不同方法的增量更新时间作为实验结果的一部分
python Naru/incremental_train.py --dataset census --epoch 30

#从models目录中收集所有带--dataset名的模型，在更新数据集上自动生成50条SQL语句，测试基数估计的准确性，这里的准确性指标需要记录
python Naru/eval_model.py --dataset census
```

