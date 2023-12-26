# TODO

# 端到端TODO
## 文件功能
eval_model.py --eval_type drift
- 更新数据
- 检测是否漂移

eval_model.py --eval_type estimate
- 检测query准确度

incremental_train.py
- 模型更新，ddup或adapt

## 工作流程
定义2种workload，随机进
1. query
2. 更新数据(permute(原eval_model.py)/sample/permute/single)
   - 加3中，实现数据增加 

每一次workload 2后，检测一次datadrift(eval_type drift)
- 若检测到drift，就更新模型

运行效果：
1. 所有query的准确度
2. 更新模型的总时间开销

1次实验中，每个workload用的drift_test是同一种(共2种)
- JS_divergence
- 原Naru

## 输入&输出
### 输入参数
- dataset_name
  - bjaq
  - census
  - forest
  - power
- drift_test_method (python Naru/eval_model.py --dataset census --eval_type drift
  - js-divergence (our)
  - ddup
- model_update_method (incremental_train.py)
  - ddup -> Update (DDUp) + finetune (baseline)
  - js-divergence -> Adapt (our) +  finetune (baseline)
- data_update_method
  - permute (DDUp)
  - sample (FACE)
  - permute (FACE)
  - single (our) TODO: 想其他更新方法
- model
  - naru
  - face

### 输出
展示运行效果，存到文件里
