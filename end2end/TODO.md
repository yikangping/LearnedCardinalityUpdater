# TODO

# 端到端TODO
## 文件功能
eval_model.py --eval_type drift 对应数据更新workload
- 更新数据
- 检测是否漂移

eval_model.py --eval_type estimate 对应查询workload
- 检测query准确度

incremental_train.py 对应检测到drift后的模型更新
- 模型更新，ddup或adapt

## 工作流程
定义2种workload，随机进
1. query
2. 更新数据(permute(原eval_model.py)/sample/permute/single)
   - 加3种
   - 实现数据增加 

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
- drift_test (Naru/eval_model.py --eval_type drift)
  - js (JS-divergence, our)
  - ddup
- data_update (Naru/eval_model.py --eval_type drift)
  - permute-ddup (DDUp)
  - sample (FACE)
  - permute (FACE)
  - single (our)
- model_update (Naru/incremental_train.py)
  - update (DDUp -> ddup)
  - adapt (our -> js)
  - finetune (baseline -> ddup & js)
- model
  - naru
  - face

### 输出
展示运行效果，存到文件里


# 更新
对于forest
每次新增50000左右，固定

对于QueryWorkload
统计origin-forest_43.843 	max: 83.0000	99th: 66.6667	95th: 3.3958	median: 1.0327	mean: 3.8793
