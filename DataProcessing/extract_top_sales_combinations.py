import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import gc
import os
warnings.filterwarnings('ignore')

print("🚀 开始提取前10%销量组合...")
print(f"开始时间: {datetime.now()}")

# 1. 读取基础数据
print("📖 读取基础数据...")
# 读取全部商品信息
goods_df = pd.read_csv('商品信息.csv', dtype=str, encoding='utf-8-sig', keep_default_na=False)
store_df = pd.read_csv('店铺信息.csv', dtype=str, encoding='utf-8-sig', keep_default_na=False)

goods_df = goods_df.rename(columns={'sgoodsskc': 'item_id'})
store_df = store_df.rename(columns={'sstoreno': 'store_id'})

print(f"全部商品信息: {len(goods_df)}行, 店铺信息: {len(store_df)}行")

# 2. 设置参数
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2025-06-30')

# 🚀 处理全部店铺和全部商品
all_stores_full = store_df['store_id'].unique()
all_stores = list(all_stores_full)  # 全部店铺
all_items = goods_df['item_id'].unique()  # 全部商品

print(f"🎯 数据范围: 处理全部店铺，共{len(all_stores_full)}个")
print(f"🎯 商品范围: 全部商品 {len(all_items)}个")

# 🚀 性能优化：预处理商品店铺特征，避免重复merge
print("⚡ 预处理商品店铺特征...")
# 设置商品和店铺ID为索引，提升merge效率
goods_df = goods_df.set_index('item_id')
store_df = store_df.set_index('store_id')
print("✅ 商品店铺特征预处理完成")

# 3. 分块读取销售数据并计算总销量
print("💾 分块读取销售数据并计算总销量...")
# 设置分块大小（根据内存情况调整）
CHUNK_SIZE = 1000000  # 每次读取100万行

# 初始化总销量数据存储
total_sales_by_combination = {}

# 分块读取销售数据
chunk_count = 0
for chunk in pd.read_csv('销售信息_精简.csv', 
                        dtype={'dtdate': str, 'sstoreno': str, 'sgoodsskc': str, 'fquantity': str},
                        encoding='utf-8-sig', 
                        keep_default_na=False,
                        chunksize=CHUNK_SIZE):
    
    chunk_count += 1
    print(f"📖 处理第 {chunk_count} 块数据 ({len(chunk):,} 行)...")
    
    # 数据清理（向量化）
    chunk = chunk.rename(columns={'sstoreno': 'store_id', 'sgoodsskc': 'item_id'})
    valid_mask = chunk['dtdate'].str.match(r'^\d{4}-\d{2}-\d{2}$', na=False)
    chunk = chunk[valid_mask]

    chunk['dtdate'] = pd.to_datetime(chunk['dtdate'], errors='coerce')
    chunk['sales'] = pd.to_numeric(chunk['fquantity'], errors='coerce').fillna(0)

    # 将负数的销量改为0
    chunk['sales'] = chunk['sales'].clip(lower=0)

    chunk = chunk[(chunk['dtdate'] >= start_date) & (chunk['dtdate'] <= end_date)]
    chunk = chunk.dropna(subset=['dtdate'])

    # 🎯 只保留全部店铺的销售数据
    chunk = chunk[chunk['store_id'].isin(all_stores) & chunk['item_id'].isin(all_items)]
    
    if len(chunk) > 0:
        # 聚合销售数据（按店铺-商品组合求和）
        chunk_agg = chunk.groupby(['store_id', 'item_id'])['sales'].sum().reset_index()
        
        # 更新总销量字典
        for _, row in chunk_agg.iterrows():
            key = (row['store_id'], row['item_id'])
            if key in total_sales_by_combination:
                total_sales_by_combination[key] += row['sales']
            else:
                total_sales_by_combination[key] = row['sales']
    
    # 显示进度
    if chunk_count % 5 == 0:
        print(f"   已处理 {chunk_count} 块，当前累计组合数: {len(total_sales_by_combination):,}")

# 转换为DataFrame
print("📊 转换为DataFrame并添加商品店铺信息...")
sales_analysis_df = pd.DataFrame([
    {'store_id': store_id, 'item_id': item_id, 'total_sales': sales}
    for (store_id, item_id), sales in total_sales_by_combination.items()
])

print(f"✅ 总销量计算完成，共 {len(sales_analysis_df):,} 个店铺-商品组合")

# 添加商品和店铺信息
sales_analysis_df = sales_analysis_df.merge(goods_df, left_on='item_id', right_index=True, how='left')
sales_analysis_df = sales_analysis_df.merge(store_df, left_on='store_id', right_index=True, how='left')

# 4. 按销量排序并提取前10%
print("📈 按销量排序并提取前10%组合...")

# 按总销量排序（降序）
sales_analysis_df = sales_analysis_df.sort_values('total_sales', ascending=False).reset_index(drop=True)

# 计算前10%的组合数量
total_combinations = len(sales_analysis_df)
top_10_percent_count = int(total_combinations * 0.1)

print(f"📊 总组合数: {total_combinations:,}")
print(f"📊 前10%组合数: {top_10_percent_count:,}")

# 提取前10%的组合
top_10_percent_df = sales_analysis_df.head(top_10_percent_count).copy()

# 添加排名信息
top_10_percent_df['rank'] = range(1, len(top_10_percent_df) + 1)
top_10_percent_df['rank_percentage'] = (top_10_percent_df['rank'] / total_combinations) * 100

# 计算累计销量和占比
total_sales_sum = sales_analysis_df['total_sales'].sum()
top_10_percent_df['cumulative_sales'] = top_10_percent_df['total_sales'].cumsum()
top_10_percent_df['sales_percentage'] = (top_10_percent_df['total_sales'] / total_sales_sum) * 100
top_10_percent_df['cumulative_sales_percentage'] = (top_10_percent_df['cumulative_sales'] / total_sales_sum) * 100

# 5. 输出统计信息
print("=" * 60)
print("📊 前10%销量组合统计")
print("=" * 60)

print(f"总组合数: {total_combinations:,}")
print(f"前10%组合数: {len(top_10_percent_df):,}")
print(f"前10%组合总销量: {top_10_percent_df['total_sales'].sum():,.0f}")
print(f"前10%组合销量占比: {(top_10_percent_df['total_sales'].sum() / total_sales_sum) * 100:.2f}%")

print(f"\n前10%组合销量统计:")
print(f"  最大销量: {top_10_percent_df['total_sales'].max():,.0f}")
print(f"  最小销量: {top_10_percent_df['total_sales'].min():,.0f}")
print(f"  平均销量: {top_10_percent_df['total_sales'].mean():,.2f}")
print(f"  中位数销量: {top_10_percent_df['total_sales'].median():,.2f}")

# 6. 保存前10%组合到CSV文件
print("💾 保存前10%组合到CSV文件...")

# 只保留store_id和item_id
final_df = top_10_percent_df[['store_id', 'item_id']]

# 保存文件
output_file = '../top_10_percent_sales_combinations.csv'
final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ 前10%组合已保存为 {output_file}")

# 7. 输出文件信息
print("\n" + "=" * 60)
print("📋 输出文件信息")
print("=" * 60)

print(f"文件名: {output_file}")
print(f"行数: {len(final_df):,}")
print(f"列数: {len(final_df.columns)}")
print(f"文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

print(f"\n列信息:")
for i, col in enumerate(final_df.columns, 1):
    print(f"  {i:2d}. {col}")

# 8. 输出前10个组合的示例
print("\n" + "=" * 60)
print("🔍 前10个组合示例")
print("=" * 60)

print(f"{'排名':<4} {'店铺ID':<12} {'商品ID':<12}")
print("-" * 30)

for i, row in enumerate(final_df.head(10).iterrows(), 1):
    print(f"{i:<4} {row[1]['store_id']:<12} {row[1]['item_id']:<12}")

# 9. 输出完成信息
print("\n" + "=" * 60)
print("🎉 前10%销量组合提取完成!")
print(f"结束时间: {datetime.now()}")
print("=" * 60)
print("输出文件:")
print(f"- 📊 前10%组合: {output_file}")
print("=" * 60)
