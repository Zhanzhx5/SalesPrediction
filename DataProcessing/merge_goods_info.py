import pandas as pd
import numpy as np

# 需要合并的文件
files = [
    '文胸信息.csv',
    '内裤信息.csv',
    '塑身信息.csv',
    '保暖信息.csv',
    '家居服信息.csv',
]

# 统一的主键字段名
skc_col = 'sgoodsskc'

# 读取所有表，标准化主键列名
all_dfs = []
for f in files:
    df = pd.read_csv(f, dtype=str, encoding='utf-8-sig')
    # 兼容不同表头可能有空格或全角冒号
    cols = [c.replace('：', ':').replace(' ', '') for c in df.columns]
    df.columns = cols
    if skc_col not in df.columns:
        raise ValueError(f"{f} 缺少主键列 {skc_col}")
    all_dfs.append(df)

# 合并所有属性列
all_columns = set()
for df in all_dfs:
    all_columns.update(df.columns)
all_columns = [skc_col] + [c for c in all_columns if c != skc_col]

# 合并所有数据
merged = pd.DataFrame(columns=all_columns)
for df in all_dfs:
    merged = pd.concat([merged, df], ignore_index=True, sort=False)

# 按主键去重，保留第一个
merged = merged.drop_duplicates(subset=[skc_col], keep='first')

# 填充N/A表示不适用
merged = merged.fillna('N/A')

# 导出
merged.to_csv('商品信息.csv', index=False, encoding='utf-8-sig')
print('已生成: 商品信息.csv') 