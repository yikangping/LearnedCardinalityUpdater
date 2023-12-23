# 数据集总览
共有4个数据集，其中：
- Naru里面有2个数据集（census和forest），源格式为csv；
- FACE里面有2个数据集（BJAQ和power），源格式为npy。


# 数据集详细信息
BJAQ: 
- Intro from README/reference/papers/FACE.pdf:
  - BJAQ includes hourly air pollutants data from 12 air-quality monitoring sites.
  - It has medium domain sizes (1K-2K).
- 共5列，distribution较小

Power: 
- Intro from README/reference/papers/FACE.pdf:
  - Power is a household electric power consumption data gathered in 47 months. 
  - It has large domain sizes in all columns (each ≈ 2M) and all columns are numerical data.
- 共6列，distribution极大，训练很慢


# 注意事项
- 根据数据集属性数量设置Naru/eval_model.py中num_filters参数
  - 生成SQL的模式是随机生成若干个过滤谓词，这个过滤器的数量和数据集的属性数量是正相关的，
  - 所以需要确认一下SQL生成模块里面生成的过滤器数量是不是和数据集原始属性数量匹配
