import pandas as pd
import os

# 读取原始CSV文件
# Read the original CSV file
input_path = 'datasets/Al_Alloys/USTB/USTB_Al_alloys_processed.csv'
output_path = 'datasets/Al_Alloys/USTB/USTB_Al_alloys_processed_split.csv'

# 检查文件是否存在
# Check if the file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"File not found: {input_path}")

# 读取数据
# Read data
# encoding='utf-8' 以防止中文乱码
# encoding='utf-8' to avoid Chinese garbled characters
df = pd.read_csv(input_path, encoding='utf-8')

# 目标列名
# Target column names
st_col = 'ST（℃）多级固溶机制以|隔开'
time_col = '固溶时间（min）多级固溶机制以|隔开'

# 定义拆分函数
# Define split function
def split_to_three(val):
    # 处理缺失值
    # Handle missing values
    if pd.isna(val):
        return [0, 0, 0]
    # 拆分字符串
    # Split string
    parts = str(val).split('|')
    # 转为数字，去除空格
    # Convert to number, remove spaces
    parts = [int(float(p.strip())) if p.strip() else 0 for p in parts]
    # 补齐到3项
    # Pad to 3 items
    while len(parts) < 3:
        parts.append(0)
    # 只保留前三项
    # Only keep first 3 items
    return parts[:3]

# 拆分温度和时间列
# Split temperature and time columns
st_split = df[st_col].apply(split_to_three).apply(pd.Series)
time_split = df[time_col].apply(split_to_three).apply(pd.Series)
st_split.columns = ['ST1', 'ST2', 'ST3']
time_split.columns = ['TIME1', 'TIME2', 'TIME3']

# 交替插入新列
# Interleave new columns
interleaved = pd.DataFrame()
for i in range(1, 4):
    interleaved[f'ST{i}'] = st_split[f'ST{i}']
    interleaved[f'TIME{i}'] = time_split[f'TIME{i}']

# 拼接到原始数据后
# Concatenate to the original dataframe
df_out = pd.concat([df, interleaved], axis=1)

# 修改列名，将包含℃的列名改为英文
# Modify column names, change columns containing ℃ to English
column_mapping = {
    '冷变形（%）': 'Cold_Deformation_percent',
    '一级时效温度(℃)': 'First_Aging_Temp_C',
    '一级时效时间（h）': 'First_Aging_Time_h',
    '二级时效温度（℃）': 'Second_Aging_Temp_C',
    '二级时效时间（h）': 'Second_Aging_Time_h',
    '三级时效温度（℃）': 'Third_Aging_Temp_C',
    '三级时效时间（h）': 'Third_Aging_Time_h'
}

# 重命名列
# Rename columns
df_out = df_out.rename(columns=column_mapping)

# 保存新CSV文件
# Save new CSV file
df_out.to_csv(output_path, index=False, encoding='utf-8')

print(f"已完成拆分和列名修改，结果保存为: {output_path}") 