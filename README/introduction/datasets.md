# 数据集情况
## 总览
共有4个数据集，其中：
- Naru里面有2个数据集（census和forest），源格式为csv
- FACE里面有2个数据集（BJAQ和power），源格式为npy


## 详细信息

| Dataset | Rows      | Columns | Type     |
|---------|-----------|---------|----------|
| Census  | 32,561    | 13      | str, num |
| Forest  | 581,012   | 10      | num      |
| BJAQ    | 382,168   | 5       | num      |
| Power   | 2,049,280 | 6       | num      |

注意：此表中的行数与列数来源于本项目中的数据集文件，可能与原数据集不同。

BJAQ: 
- Intro from README/reference/papers/FACE.pdf:
  - BJAQ includes hourly air pollutants data from 12 air-quality monitoring sites.
  - It has medium domain sizes (1K-2K).
- distribution较小

Power: 
- Intro from README/reference/papers/FACE.pdf:
  - Power is a household electric power consumption data gathered in 47 months. 
  - It has large domain sizes in all columns (each ≈ 2M) and all columns are numerical data.
- distribution极大，训练很慢


## 注意事项
- 根据数据集属性数量设置Naru/eval_model.py中num_filters参数
  - 生成SQL的模式是随机生成若干个过滤谓词，这个过滤器的数量和数据集的属性数量是正相关的，
  - 所以需要确认一下SQL生成模块里面生成的过滤器数量是不是和数据集原始属性数量匹配
