import pandas as pd

files = ['店铺信息.csv', '商品信息.csv']

for f in files:
    print(f'\n==== {f} ====')
    df = pd.read_csv(f, dtype=str, encoding='utf-8-sig')
    for col in df.columns:
        vals = df[col].dropna().unique()
        sample = vals[:5] if len(vals) > 5 else vals
        print(f'{col}: 样例值: {sample}, 非空唯一值数: {len(vals)}') 