import pandas as pd

file = '销售信息.csv'

df = pd.read_csv(file, dtype=str, encoding='utf-8-sig', keep_default_na=False)

# 自动检测所有包含“日期”或“date”字样的字段
date_cols = [col for col in df.columns if '日期' in col or 'date' in col.lower()]

for col in date_cols:
    # 排除N/A、空等无效值
    vals = df[col][~df[col].isin(['', 'N/A', 'n/a', None])]
    if vals.empty:
        print(f'{col}: 无有效日期')
        continue
    try:
        dates = pd.to_datetime(vals, errors='coerce')
        min_date = dates.min()
        max_date = dates.max()
        print(f'{col}: 最早 {min_date}, 最晚 {max_date}')
    except Exception as e:
        print(f'{col}: 日期解析失败，错误: {e}') 