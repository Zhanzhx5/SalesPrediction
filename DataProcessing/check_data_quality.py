import pandas as pd
import numpy as np
import os

def check_file(filename, key_col=None):
    print(f'\n==== 检查文件: {filename} ====')
    if not os.path.exists(filename):
        print('文件不存在！')
        return
    df = pd.read_csv(filename, dtype=str, encoding='utf-8-sig')
    print(f'总行数: {len(df)}, 总列数: {len(df.columns)}')
    print('字段名:', list(df.columns))
    # 缺失值统计
    na_count = df.isna().sum()
    na_cols = na_count[na_count > 0]
    if not na_cols.empty:
        print('有缺失值的字段:')
        print(na_cols)
    else:
        print('无缺失值')
    # 主键唯一性
    if key_col and key_col in df.columns:
        dup = df.duplicated(subset=[key_col]).sum()
        na_key = df[key_col].isna().sum() + (df[key_col] == '').sum()
        print(f'主键字段“{key_col}”重复: {dup}，缺失: {na_key}')
    # 简单异常值检查（如全为N/A或空、唯一值过少等）
    for col in df.columns:
        uniq = df[col].nunique(dropna=False)
        if uniq == 1:
            print(f'字段“{col}”所有值都一样: {df[col].iloc[0]}')
        if df[col].isna().all() or (df[col] == "N/A").all():
            print(f'字段“{col}”全为空或N/A')
    # 重复行
    dup_rows = df.duplicated().sum()
    if dup_rows:
        print(f'重复行数: {dup_rows}')
    # 简单预览
    print('前5行数据:')
    print(df.head())

# 检查店铺信息
shop_file = '店铺信息.csv'
# 检查商品信息
goods_file = '商品信息.csv'

check_file(shop_file)
check_file(goods_file, key_col='sgoodsskc:商品SKC码') 