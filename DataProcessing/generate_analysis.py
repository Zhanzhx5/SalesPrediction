import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
warnings.filterwarnings('ignore')

print("🚀 开始销量分布分析...")
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

# 4. 计算销量分布统计
print("📈 计算销量分布统计...")

# 按总销量排序
sales_analysis_df = sales_analysis_df.sort_values('total_sales', ascending=False).reset_index(drop=True)

# 计算总销量
total_sales_sum = sales_analysis_df['total_sales'].sum()
print(f"📊 总销量: {total_sales_sum:,.0f}")

# 计算各百分位的销量占比
percentiles = list(range(5, 101, 5))  # 5%, 10%, 15%, ..., 100%
percentile_stats = []

for p in percentiles:
    # 计算前p%的组合数量
    n_combinations = len(sales_analysis_df)
    top_n = int(n_combinations * p / 100)
    
    if top_n > 0:
        # 前p%的总销量
        top_sales = sales_analysis_df.head(top_n)['total_sales'].sum()
        # 销量占比
        sales_percentage = (top_sales / total_sales_sum) * 100
        
        percentile_stats.append({
            'percentile': p,
            'combinations_count': top_n,
            'combinations_percentage': p,
            'sales_sum': top_sales,
            'sales_percentage': sales_percentage
        })

percentile_df = pd.DataFrame(percentile_stats)

# 5. 输出统计结果
print("=" * 60)
print("📊 销量分布分析结果")
print("=" * 60)

print(f"总店铺-商品组合数: {len(sales_analysis_df):,}")
print(f"总销量: {total_sales_sum:,.0f}")
print(f"平均每个组合销量: {total_sales_sum / len(sales_analysis_df):,.2f}")
print(f"中位数销量: {sales_analysis_df['total_sales'].median():,.2f}")

print("\n📈 各百分位销量占比:")
print("-" * 80)
print(f"{'百分位':<8} {'组合数':<10} {'组合占比':<10} {'销量':<15} {'销量占比':<10}")
print("-" * 80)

for _, row in percentile_df.iterrows():
    print(f"{row['percentile']:>3}%    {row['combinations_count']:>8,} {row['combinations_percentage']:>8.1f}% "
          f"{row['sales_sum']:>12,.0f} {row['sales_percentage']:>8.1f}%")

# 6. 生成可视化图表
print("\n🎨 生成可视化图表...")

# 设置中文字体
plt.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus']=False

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('店铺-商品组合销量分布分析', fontsize=16, fontweight='bold')

# 1. 销量占比曲线
ax1 = axes[0, 0]
ax1.plot(percentile_df['percentile'], percentile_df['sales_percentage'], 'b-', linewidth=2, marker='o')
ax1.set_xlabel('组合排名百分位')
ax1.set_ylabel('累计销量占比 (%)')
ax1.set_title('销量集中度分析')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)

# 添加关键点标注
key_points = [10, 25, 50, 75, 90]
for point in key_points:
    if point in percentile_df['percentile'].values:
        row = percentile_df[percentile_df['percentile'] == point].iloc[0]
        ax1.annotate(f'{row["sales_percentage"]:.1f}%', 
                    xy=(point, row['sales_percentage']), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# 2. 销量分布直方图
ax2 = axes[0, 1]
# 使用对数刻度显示销量分布
log_sales = np.log10(sales_analysis_df['total_sales'] + 1)  # +1避免log(0)
ax2.hist(log_sales, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax2.set_xlabel('销量 (log10)')
ax2.set_ylabel('组合数量')
ax2.set_title('销量分布直方图')
ax2.grid(True, alpha=0.3)

# 3. 帕累托图（前20%的组合）
ax3 = axes[1, 0]
top_20_percent = int(len(sales_analysis_df) * 0.2)
top_20_data = sales_analysis_df.head(top_20_percent)
top_20_data['cumulative_sales'] = top_20_data['total_sales'].cumsum()
top_20_data['cumulative_percentage'] = (top_20_data['cumulative_sales'] / total_sales_sum) * 100

ax3_twin = ax3.twinx()
ax3.bar(range(len(top_20_data)), top_20_data['total_sales'], alpha=0.7, color='lightcoral')
ax3_twin.plot(range(len(top_20_data)), top_20_data['cumulative_percentage'], 'r-', linewidth=2, marker='o')

ax3.set_xlabel('组合排名')
ax3.set_ylabel('销量', color='lightcoral')
ax3_twin.set_ylabel('累计销量占比 (%)', color='red')
ax3.set_title('前20%组合的帕累托分析')
ax3.grid(True, alpha=0.3)

# 4. 销量区间分布
ax4 = axes[1, 1]
# 定义销量区间
sales_bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, float('inf')]
sales_labels = ['0-10', '11-50', '51-100', '101-500', '501-1K', '1K-5K', '5K-10K', '10K+']
sales_analysis_df['sales_range'] = pd.cut(sales_analysis_df['total_sales'], bins=sales_bins, labels=sales_labels, include_lowest=True)

range_counts = sales_analysis_df['sales_range'].value_counts().sort_index()
ax4.bar(range(len(range_counts)), range_counts.values, alpha=0.7, color='lightgreen')
ax4.set_xlabel('销量区间')
ax4.set_ylabel('组合数量')
ax4.set_title('销量区间分布')
ax4.set_xticks(range(len(range_counts)))
ax4.set_xticklabels(range_counts.index, rotation=45)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../sales_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存为 sales_distribution_analysis.png")

# 7. 保存详细统计结果
print("💾 保存详细统计结果...")
output_file = '../sales_distribution_stats.csv'
percentile_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ 统计结果已保存为 {output_file}")

# 8. 输出关键洞察
print("\n" + "=" * 60)
print("🔍 关键洞察")
print("=" * 60)

# 前10%的销量占比
top_10_percent = percentile_df[percentile_df['percentile'] == 10].iloc[0]
print(f"📊 前10%的店铺-商品组合贡献了 {top_10_percent['sales_percentage']:.1f}% 的总销量")

# 前25%的销量占比
top_25_percent = percentile_df[percentile_df['percentile'] == 25].iloc[0]
print(f"📊 前25%的店铺-商品组合贡献了 {top_25_percent['sales_percentage']:.1f}% 的总销量")

# 前50%的销量占比
top_50_percent = percentile_df[percentile_df['percentile'] == 50].iloc[0]
print(f"📊 前50%的店铺-商品组合贡献了 {top_50_percent['sales_percentage']:.1f}% 的总销量")

# 销量集中度分析
print(f"\n🎯 销量集中度分析:")
print(f"   - 前10%组合: {top_10_percent['sales_percentage']:.1f}% 的销量")
print(f"   - 前25%组合: {top_25_percent['sales_percentage']:.1f}% 的销量")
print(f"   - 前50%组合: {top_50_percent['sales_percentage']:.1f}% 的销量")
print(f"   - 后50%组合: {100 - top_50_percent['sales_percentage']:.1f}% 的销量")

# 长尾分析
bottom_50_percent = percentile_df[percentile_df['percentile'] == 50].iloc[0]
long_tail_percentage = 100 - bottom_50_percent['sales_percentage']
print(f"\n📈 长尾效应分析:")
print(f"   - 后50%的组合仅贡献 {long_tail_percentage:.1f}% 的销量")
print(f"   - 存在明显的长尾效应，少数组合贡献大部分销量")

# 9. 输出完成信息
print("\n" + "=" * 60)
print("🎉 销量分布分析完成!")
print(f"结束时间: {datetime.now()}")
print("=" * 60)
print("输出文件:")
print(f"- 📊 统计结果: {output_file}")
print(f"- 📈 可视化图表: sales_distribution_analysis.png")
print("=" * 60)
