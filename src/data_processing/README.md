# Alloy Data Processing

This module provides tools for processing and analyzing alloy data, with a focus on aluminum alloys and high entropy alloys.

## 模块结构

```
data_processing/
├── data_processor.py      # 基础数据处理类
├── json_processor.py      # JSON数据处理
├── description_generator.py # 工艺描述生成器
├── data_filter.py         # 数据过滤工具
├── data_merger.py         # 数据集合并工具
├── utils.py              # 通用工具函数
├── constants.py          # 常量定义
└── main.py              # 主程序入口
```

## 功能特点

1. 数据处理

   - 支持CSV和JSON格式的数据读写
   - 自动处理缺失值
   - 数据格式转换
2. 工艺描述生成

   - 为铝合金生成工艺描述
   - 为高熵合金生成工艺描述
   - 支持自定义描述模板
3. 数据过滤

   - 按温度范围过滤
   - 按成分范围过滤
   - 按性能值范围过滤
4. 数据集合并

   - 合并多个铝合金数据集
   - 合并多个高熵合金数据集
   - 支持自定义合并列
5. 工具函数

   - 成分归一化
   - 构型熵计算
   - 成分验证
   - 温度和成分格式化

## 使用示例

### 处理铝合金数据

```python
from data_processing import DataProcessor, DescriptionGenerator, DataFilter
from data_processing.constants import DEFAULT_COLUMNS

# 初始化处理器
processor = DataProcessor()
desc_generator = DescriptionGenerator()
filter = DataFilter()

# 加载数据
df = processor.load_data('datasets/Al_Alloys/Al_alloys.csv')

# 按成分过滤（Al含量在90-95%之间）
df = filter.filter_by_composition(
    df=df,
    element_columns=['Al'],
    min_composition=90,
    max_composition=95
)

# 按性能过滤（抗拉强度大于300MPa）
df = filter.filter_by_property(
    df=df,
    property_column='tensile_strength',
    min_value=300
)

# 生成工艺描述
df = desc_generator.generate_al_description(df)

# 保存处理后的数据
processor.save_data(df, 'processed_al_alloys.csv')
```

### 处理高熵合金数据

```python
from data_processing import DataProcessor, DescriptionGenerator, DataFilter
from data_processing.constants import DEFAULT_TEMP_RANGES

# 初始化处理器
processor = DataProcessor()
desc_generator = DescriptionGenerator()
filter = DataFilter()

# 加载数据
df = processor.load_data('datasets/HEA_data/merged_HEA_datasets.csv')

# 按温度范围过滤（室温）
room_temp_range = DEFAULT_TEMP_RANGES['room_temp']
df = filter.filter_by_temperature(
    df=df,
    temp_column='Temperature(K)',
    min_temp=room_temp_range[0],
    max_temp=room_temp_range[1]
)

# 按成分过滤（各元素含量在5-35%之间）
df = filter.filter_by_composition(
    df=df,
    element_columns=['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni'],
    min_composition=5,
    max_composition=35
)

# 生成工艺描述
df = desc_generator.generate_hea_description(df)

# 保存处理后的数据
processor.save_data(df, 'processed_hea_data.csv')
```

### 合并数据集

```python
from data_processing import DataMerger

# 初始化合并器
merger = DataMerger()

# 加载数据集
df1 = merger.load_data('datasets/HEA_data/dataset1.csv')
df2 = merger.load_data('datasets/HEA_data/dataset2.csv')

# 合并数据集
merged_df = merger.merge_hea_datasets(df1, df2)

# 保存合并后的数据
merger.save_data(merged_df, 'datasets/HEA_data/merged_HEA_datasets.csv')
```

### 使用命令行工具

```bash
# 处理铝合金数据
python -m data_processing.main process-al datasets/Al_Alloys/Al_alloys.csv processed_al_alloys.csv

# 处理高熵合金数据
python -m data_processing.main process-hea datasets/HEA_data/merged_HEA_datasets.csv processed_hea_data.csv

# 合并数据集
python -m data_processing.main merge datasets/HEA_data/dataset1.csv datasets/HEA_data/dataset2.csv datasets/HEA_data/merged_HEA_datasets.csv --type hea
```

## 数据文件结构

### 铝合金数据 (Al_alloys.csv)

- 必需列：
  - `Al`, `Cu`, `Mg` 等元素成分列（百分比）
  - `temperature`：处理温度（°C）
  - `time`：处理时间（小时）
  - `tensile_strength`：抗拉强度（MPa）
  - `yield_strength`：屈服强度（MPa）
  - `elongation`：延伸率（%）

### 高熵合金数据 (merged_HEA_datasets.csv)

- 必需列：
  - `Al`, `Co`, `Cr`, `Cu`, `Fe`, `Ni` 等元素成分列（百分比）
  - `Temperature(K)`：测试温度（K）
  - `processing_method`：处理方法
  - `tensile_strength`：抗拉强度（MPa）
  - `yield_strength`：屈服强度（MPa）
  - `elongation`：延伸率（%）

## 依赖项

- Python 3.6+
- pandas
- numpy

## 安装

```bash
pip install -r requirements.txt
```

## 注意事项

1. 数据格式

   - CSV文件应包含必要的列名
   - JSON文件应遵循预定义的结构
   - 所有数值应为数值类型
2. 成分数据

   - 成分总和应接近100%
   - 使用百分比表示
   - 支持自动归一化
3. 温度数据

   - 默认使用摄氏度
   - 支持开尔文温度转换
   - 范围检查
4. 性能数据

   - 使用标准单位
   - 支持单位转换
   - 数据验证
