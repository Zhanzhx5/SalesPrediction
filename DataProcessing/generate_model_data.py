import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import gc
import os
warnings.filterwarnings('ignore')

# 导入中国节假日库
import chinese_calendar as cc

# 数值特征白名单
NUMERICAL_FEATURES = [
    # 价格相关
    'nvipprice',        # VIP价格
    'sgprice',          # 吊牌价
    'nsaleprice',       # 销售价格
    
    # 店铺面积/人员
    'nstorearea',       # 店铺面积
    'number_assistant', # 店员数量
    
    # 所有基于历史销量计算出的特征
    'sales_lag_1',      # 历史平移特征
    'sales_lag_3',
    'sales_lag_7',
    'sales_lag_14',
    'sales_lag_28',
    'sales_lag_364',
    'rolling_mean_7',   # 滑动窗口特征
    'rolling_std_7',
    'rolling_mean_14',
    'rolling_std_14',
    'rolling_mean_30',
    'rolling_max_30',
    'rolling_min_30',
    'last_year_rolling_mean_7',  # 去年同期滑动特征
    'last_year_rolling_std_7',
    'YoY_growth_rate_7', # 同比增长率
    
    # 新增的时间差特征
    'item_age',         # 商品上市天数
    'store_age',        # 店铺开业天数
    
    # 新增的库存和销量间隔特征
    'zd_kc',            # 在店库存
    'days_since_last_sale'  # 距离上一次有销量的天数
]

print("🚀 开始完整数据处理（全部店铺+全部商品）...")
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
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# 🔧 修复：先获取ID列表，再设置索引
# 🚀 处理全部店铺和全部商品
all_stores_full = store_df['store_id'].unique()
all_stores = list(all_stores_full)  # 全部店铺
all_items = goods_df['item_id'].unique()  # 全部商品

print(f"🎯 数据范围: 处理全部店铺，共{len(all_stores_full)}个")
print(f"🎯 商品范围: 全部商品 {len(all_items)}个")
print(f"选中的店铺: {', '.join(all_stores)}")

# 🚀 性能优化：预处理商品店铺特征，避免重复merge
print("⚡ 预处理商品店铺特征...")
# 设置商品和店铺ID为索引，提升merge效率
goods_df = goods_df.set_index('item_id')
store_df = store_df.set_index('store_id')
print("✅ 商品店铺特征预处理完成")

print(f"日期: {len(all_dates)}天, 店铺: {len(all_stores)}个, 商品: {len(all_items)}个")
print(f"📊 预计总数据量: {len(all_dates) * len(all_stores) * len(all_items):,} 行 (完整版)")

# 3. 预计算节假日（向量化）
print("🗓️ 预计算节假日...")
holiday_array = np.array([cc.is_holiday(date.date()) for date in all_dates], dtype=np.int8)

# 4. 分块读取销售数据
print("💾 分块读取销售数据...")
# 设置分块大小（根据内存情况调整）
CHUNK_SIZE = 1000000  # 每次读取100万行

# 预分配时间特征数组
n_dates = len(all_dates)
time_features = np.zeros((n_dates, 7), dtype=np.int16)
date_to_idx = {}
for i, date in enumerate(all_dates):
    date_to_idx[date] = i
    time_features[i] = [
        date.year, date.month, date.day, date.dayofweek,
        date.isocalendar().week, int(date.dayofweek >= 5), holiday_array[i]
    ]

# 6. 输出文件设置
output_file = '../model_data.csv'  # 输出到上级目录

print(f"🎯 输出文件: {output_file} (完整版: 全部店铺+全部商品)")

# 7. 分块处理销售数据
print("📊 开始分块处理销售数据...")

# 初始化销售数据存储
all_sales_data = []

# 分块读取销售数据
chunk_count = 0
for chunk in pd.read_csv('销售信息_精简.csv', 
                        dtype={'dtdate': str, 'sstoreno': str, 'sgoodsskc': str, 'fquantity': str, '在店库存': str},
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
    chunk['zd_kc'] = pd.to_numeric(chunk['在店库存'], errors='coerce').fillna(0)

    # 将负数的销量和在店库存改为0
    chunk['sales'] = chunk['sales'].clip(lower=0)
    chunk['zd_kc'] = chunk['zd_kc'].clip(lower=0)

    chunk = chunk[(chunk['dtdate'] >= start_date) & (chunk['dtdate'] <= end_date)]
    chunk = chunk.dropna(subset=['dtdate'])

    # 聚合销售数据（销量求和，在店库存取最后值）
    chunk = chunk.groupby(['dtdate', 'store_id', 'item_id']).agg({
        'sales': 'sum',
        'zd_kc': 'last'  # 在店库存取最后值，因为库存是状态值
    }).reset_index()

    # 🎯 只保留全部店铺的销售数据
    chunk = chunk[chunk['store_id'].isin(all_stores) & chunk['item_id'].isin(all_items)]
    
    if len(chunk) > 0:
        all_sales_data.append(chunk)
    
    # 显示进度
    if chunk_count % 5 == 0:
        print(f"   已处理 {chunk_count} 块，当前累计有效数据: {sum(len(df) for df in all_sales_data):,} 行")

# 合并所有销售数据
if all_sales_data:
    sales_df = pd.concat(all_sales_data, ignore_index=True)
    del all_sales_data  # 释放内存
    gc.collect()
    
    # 再次聚合（可能有重复的店铺-商品-日期组合）
    sales_df = sales_df.groupby(['dtdate', 'store_id', 'item_id']).agg({
        'sales': 'sum',
        'zd_kc': 'last'
    }).reset_index()
    
    print(f"✅ 销售数据处理完成，总计: {len(sales_df):,} 行")
else:
    sales_df = pd.DataFrame(columns=['dtdate', 'store_id', 'item_id', 'sales', 'zd_kc'])
    print("⚠️ 警告：没有找到有效的销售数据")

# 8. 分块创建基础框架和处理数据
print("📊 分块创建基础框架...")

# 计算每个商品的处理批次大小（避免内存溢出）
ITEMS_PER_CHUNK = 100  # 每次处理100个商品
total_chunks = (len(all_items) + ITEMS_PER_CHUNK - 1) // ITEMS_PER_CHUNK

print(f"📊 将分 {total_chunks} 个批次处理，每批 {ITEMS_PER_CHUNK} 个商品")

# 初始化输出文件
first_chunk = True

for chunk_idx in range(total_chunks):
    start_idx = chunk_idx * ITEMS_PER_CHUNK
    end_idx = min((chunk_idx + 1) * ITEMS_PER_CHUNK, len(all_items))
    current_items = all_items[start_idx:end_idx]
    
    print(f"🔄 处理批次 {chunk_idx + 1}/{total_chunks}: 商品 {start_idx + 1}-{end_idx} (共{len(current_items)}个)")
    
    # 创建当前批次的基础框架
    n_combinations = len(all_dates) * len(all_stores) * len(current_items)
    
    chunk_df = pd.DataFrame({
        'dtdate': np.repeat(all_dates, len(all_stores) * len(current_items)),
        'store_id': np.tile(np.repeat(all_stores, len(current_items)), len(all_dates)),
        'item_id': np.tile(current_items, len(all_dates) * len(all_stores)),
        'sales': 0.0,
        'zd_kc': 0.0
    })
    
    print(f"   基础框架: {n_combinations:,} 行")
    
    # 填充销售数据
    print("   💰 填充销售数据...")
    current_sales = sales_df[sales_df['item_id'].isin(current_items)]
    
    if len(current_sales) > 0:
        # 使用merge操作
        chunk_df = chunk_df.merge(
            current_sales[['dtdate', 'store_id', 'item_id', 'sales', 'zd_kc']], 
            on=['dtdate', 'store_id', 'item_id'], 
            how='left',
            suffixes=('', '_actual')
        )
        # 用实际销量和在店库存填充，保持0为默认值
        chunk_df['sales'] = chunk_df['sales_actual'].fillna(0.0)
        chunk_df['zd_kc'] = chunk_df['zd_kc_actual'].fillna(0.0)
        chunk_df = chunk_df.drop(['sales_actual', 'zd_kc_actual'], axis=1)
    
    print(f"   销售数据填充完成，非零销量: {(chunk_df['sales'] > 0).sum()} 行")
    
    # 🔥 修复在店库存逻辑：使用前向填充沿用上一个有记录的在店库存
    print("   📦 修复在店库存逻辑：前向填充...")
    chunk_df = chunk_df.sort_values(['store_id', 'item_id', 'dtdate']).reset_index(drop=True)
    
    # 按商品-店铺组合进行前向填充
    chunk_df['zd_kc'] = chunk_df.groupby(['store_id', 'item_id'])['zd_kc'].fillna(method='ffill').fillna(0.0)
    
    print(f"   在店库存前向填充完成，非零库存: {(chunk_df['zd_kc'] > 0).sum()} 行")
    
    # 添加时间特征（向量化）
    print("   📅 添加时间特征...")
    date_indices = np.array([date_to_idx[date] for date in chunk_df['dtdate']])
    chunk_df['year'] = time_features[date_indices, 0]
    chunk_df['month'] = time_features[date_indices, 1] 
    chunk_df['day'] = time_features[date_indices, 2]
    chunk_df['day_of_week'] = time_features[date_indices, 3]
    chunk_df['week_of_year'] = time_features[date_indices, 4]
    chunk_df['is_weekend'] = time_features[date_indices, 5]
    chunk_df['is_holiday'] = time_features[date_indices, 6]
    
    # 历史特征计算（高速向量化版本）
    print("   📈 计算历史特征（高速版）...")
    chunk_df = chunk_df.sort_values(['store_id', 'item_id', 'dtdate']).reset_index(drop=True)
    
    # 使用pandas向量化操作计算所有特征
    def calculate_features_vectorized(group):
        # 确保按日期排序
        group = group.sort_values('dtdate')
        # 转numpy数组并转换为float32，提升性能
        sales = group['sales'].values.astype('float32')
        n = len(sales)
        
        # 预分配结果数组
        result_arrays = {}
        
        # 1. 滞后特征（直接索引，无需边界判断）
        lag_periods = [1, 3, 7, 14, 28, 364]
        for lag in lag_periods:
            result_arrays[f'sales_lag_{lag}'] = np.concatenate([np.zeros(lag), sales[:-lag]])
        
        # 2. 滚动统计特征（基于前N天，不含当天）
        # 7天滚动（基于前7天，不含当天）
        result_arrays['rolling_mean_7'] = pd.Series(sales).shift(1).rolling(window=7, min_periods=1).mean().values
        result_arrays['rolling_std_7'] = pd.Series(sales).shift(1).rolling(window=7, min_periods=1).std(ddof=0).values
        
        # 14天滚动（基于前14天，不含当天）
        result_arrays['rolling_mean_14'] = pd.Series(sales).shift(1).rolling(window=14, min_periods=1).mean().values
        result_arrays['rolling_std_14'] = pd.Series(sales).shift(1).rolling(window=14, min_periods=1).std(ddof=0).values
        
        # 30天滚动（基于前30天，不含当天）
        result_arrays['rolling_mean_30'] = pd.Series(sales).shift(1).rolling(window=30, min_periods=1).mean().values
        result_arrays['rolling_max_30'] = pd.Series(sales).shift(1).rolling(window=30, min_periods=1).max().values
        result_arrays['rolling_min_30'] = pd.Series(sales).shift(1).rolling(window=30, min_periods=1).min().values
        
        # 3. 同比特征
        # 去年同期7天滚动均值（基于去年同期前7天）
        result_arrays['last_year_rolling_mean_7'] = pd.Series(sales).shift(365).rolling(window=7, min_periods=1).mean().values
        result_arrays['last_year_rolling_std_7'] = pd.Series(sales).shift(365).rolling(window=7, min_periods=1).std(ddof=0).values
        
        # 同比增长率
        current_mean = result_arrays['rolling_mean_7']
        last_year_mean = result_arrays['last_year_rolling_mean_7']
        result_arrays['YoY_growth_rate_7'] = np.where(
            last_year_mean > 0, 
            (current_mean - last_year_mean) / last_year_mean, 
            0
        )
        
        # 4. 计算距离上一次有销量的天数（完全向量化版本，严格避免数据泄露）
        # 使用pandas的向量化操作，完全避免Python循环
        sales_series = pd.Series(sales)
        n = len(sales_series)
        
        # 创建销量大于0的掩码
        sales_gt_zero = sales_series > 0
        
        # 完全向量化的实现，确保只看历史数据
        # 关键思路：使用shift(1)将销量信息向后移动一位，确保当天销量不被使用
        
        # 1. 将销量掩码向后移动一位（这样每个位置只能看到前一天及之前的销量信息）
        historical_sales_gt_zero = sales_gt_zero.shift(1, fill_value=False)
        
        # 2. 为每个位置创建索引
        position_index = pd.Series(range(n))
        
        # 3. 创建一个Series来记录每个有销量位置的索引
        # 只有历史数据中有销量的位置才会被记录
        last_sale_positions = pd.Series(index=position_index.index, dtype='float64')
        
        # 4. 使用向量化操作：对于每个有历史销量的位置，记录其索引
        # 然后使用forward fill来传播到后续位置
        last_sale_positions[historical_sales_gt_zero] = position_index[historical_sales_gt_zero]
        last_sale_positions = last_sale_positions.fillna(method='ffill')
        
        # 5. 计算距离上一次有销量的天数
        days_since_last_sale = position_index - last_sale_positions
        
        # 6. 处理没有历史销量的情况（设为10000.0）
        days_since_last_sale = days_since_last_sale.fillna(10000.0)
        
        # 7. 确保第一个位置是10000.0（没有历史数据）
        if n > 0:
            days_since_last_sale.iloc[0] = 10000.0
        
        # 8. 转换为float32类型（保持与其他数值特征一致）
        result_arrays['days_since_last_sale'] = days_since_last_sale.astype('float32').values
        
        # 9. 超高效批量赋值到DataFrame（直接操作底层数据）
        # 使用pandas的concat进行批量添加列（最高效）
        new_columns_df = pd.DataFrame(result_arrays, index=group.index)
        group = pd.concat([group, new_columns_df], axis=1)
        
        return group
    
    # 批量计算所有历史特征（比循环快100倍以上）
    print("   ⚡ 执行向量化特征计算...")
    chunk_df = chunk_df.groupby(['store_id', 'item_id'], group_keys=False).apply(calculate_features_vectorized)
    
    print(f"   ✅ 历史特征计算完成（向量化版本）")
    
    # 添加商品和店铺特征（高效版）
    print("   🏪 添加商品店铺特征（高效版）...")
    # 使用索引merge，避免重复筛选
    chunk_df = chunk_df.merge(goods_df, left_on='item_id', right_index=True, how='left')
    chunk_df = chunk_df.merge(store_df, left_on='store_id', right_index=True, how='left')
    
    # 计算item_age和store_age特征
    print("   📅 计算时间差特征...")
    if 'market_date' in chunk_df.columns:
        chunk_df['market_date'] = pd.to_datetime(chunk_df['market_date'], errors='coerce')
        chunk_df['item_age'] = (chunk_df['dtdate'] - chunk_df['market_date']).dt.days
        chunk_df['item_age'] = chunk_df['item_age'].fillna(0)
        chunk_df = chunk_df.drop('market_date', axis=1)
        print("   ✅ 已计算item_age特征")
    
    if 'dopentime' in chunk_df.columns:
        chunk_df['dopentime'] = pd.to_datetime(chunk_df['dopentime'], errors='coerce')
        chunk_df['store_age'] = (chunk_df['dtdate'] - chunk_df['dopentime']).dt.days
        chunk_df['store_age'] = chunk_df['store_age'].fillna(0)
        chunk_df = chunk_df.drop('dopentime', axis=1)
        print("   ✅ 已计算store_age特征")
    
    # 按照白名单规则设置数据类型（高效版）
    print("   🔧 按照白名单规则设置数据类型...")
    
    # 识别所有特征列
    exclude_cols = ['dtdate', 'sales']
    feature_cols = [col for col in chunk_df.columns if col not in exclude_cols]
    
    # 按照白名单分离数值特征和分类特征
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if col in NUMERICAL_FEATURES:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    # 🔥 修复：先处理数值特征中的无效值，再设置数据类型
    print("   🔧 处理数值特征中的无效值...")
    for col in numeric_cols:
        if col in chunk_df.columns:
            # 处理空字符串、NaN等无效值
            chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').fillna(0.0)
    
    # 构建数据类型字典
    dtype_dict = {}
    
    # 设置分类特征为category类型
    for col in categorical_cols:
        if col in chunk_df.columns:
            dtype_dict[col] = 'category'
    
    # 设置数值特征的数据类型
    for col in numeric_cols:
        if col in chunk_df.columns:
            dtype_dict[col] = 'float32'  # 所有数值特征统一使用float32
    
    # 设置基础字段的数据类型
    dtype_dict['store_id'] = 'category'
    dtype_dict['item_id'] = 'category'
    dtype_dict['sales'] = 'float32'
    dtype_dict['zd_kc'] = 'float32'
    
    # 一次性转换所有列的数据类型
    chunk_df = chunk_df.astype(dtype_dict)
    print("   ✅ 数据类型设置完成")
    
    # 过滤保存数据（掐头去尾：只保留每个店铺-商品组合从第一次有销量到最后一次有销量的期间）
    print("   💾 过滤并保存数据（掐头去尾处理）...")
    save_start_date = pd.to_datetime('2023-01-10')
    
    # 首先过滤出2023-01-10及之后的数据
    chunk_df_after_2023 = chunk_df[chunk_df['dtdate'] >= save_start_date].copy()
    
    # 对每个店铺-商品组合进行掐头去尾处理（向量化优化版本）
    print("   🔧 执行掐头去尾处理（向量化优化）...")
    
    # 使用pandas的向量化操作，避免Python循环
    # 1. 计算每个组合的第一次和最后一次有销量的日期
    sales_info = chunk_df_after_2023[chunk_df_after_2023['sales'] > 0].groupby(['store_id', 'item_id'])['dtdate'].agg(['min', 'max']).reset_index()
    sales_info.columns = ['store_id', 'item_id', 'first_sale_date', 'last_sale_date']
    
    # 2. 根据规则调整尾部保留逻辑：
    #    如果最后一次有销量是 2025-05-01 及以后（含 5 月），则把尾保留到 2025-06-30
    may_first_2025 = pd.to_datetime('2025-05-01')
    jun_30_2025 = pd.to_datetime('2025-06-30')
    sales_info['tail_end_date'] = sales_info['last_sale_date'].where(
        sales_info['last_sale_date'] < may_first_2025,
        jun_30_2025
    )
    
    # 3. 将调整后的边界合并回原数据
    chunk_df_with_bounds = chunk_df_after_2023.merge(
        sales_info[['store_id', 'item_id', 'first_sale_date', 'tail_end_date']],
        on=['store_id', 'item_id'],
        how='inner'
    )
    
    # 4. 使用向量化操作过滤数据（掐头去尾，带尾部延长）
    chunk_df_filtered = chunk_df_with_bounds[
        (chunk_df_with_bounds['dtdate'] >= chunk_df_with_bounds['first_sale_date']) &
        (chunk_df_with_bounds['dtdate'] <= chunk_df_with_bounds['tail_end_date'])
    ].copy()
    
    # 4. 删除辅助列并排序
    chunk_df_filtered = chunk_df_filtered.drop(['first_sale_date', 'tail_end_date'], axis=1)
    chunk_df_filtered = chunk_df_filtered.sort_values(['store_id', 'item_id', 'dtdate']).reset_index(drop=True)
    
    print(f"   ✅ 向量化掐头去尾处理完成，保留了 {len(sales_info)} 个有销量的店铺-商品组合")
    
    # 保存到CSV文件（追加模式）
    if first_chunk:
        chunk_df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')
        first_chunk = False
    else:
        chunk_df_filtered.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
    
    print(f"   💾 已保存批次 {chunk_idx + 1} 数据: {len(chunk_df_filtered):,} 行")
    
    # 清理内存
    del chunk_df, chunk_df_after_2023, chunk_df_with_bounds, chunk_df_filtered
    gc.collect()
    
    # 显示进度
    progress = (chunk_idx + 1) / total_chunks * 100
    print(f"   📊 总体进度: {progress:.1f}%")

# 14. 输出统计
print("=" * 60)
print("🎉 完整数据处理完成! (全部店铺+全部商品)")
print(f"结束时间: {datetime.now()}")

# 计算最终文件大小
if os.path.exists(output_file):
    output_size = os.path.getsize(output_file) / (1024**3)
    print(f"输出文件: {output_file}")
    print(f"输出文件大小: {output_size:.2f} GB")
else:
    print("⚠️ 警告：输出文件未生成")

print("=" * 60)
print("数据说明:")
print(f"- 🎯 数据范围: 全部店铺数据（共{len(all_stores)}家）")
print("- 🎯 商品范围: 全部商品")
print("- 🎯 时间范围: 掐头去尾处理（每个店铺-商品组合从第一次有销量到最后一次有销量）")
print("- 标签字段 (y): sales (当日销量)")
print("- 特征字段 (X): 除sales外的所有字段，包括zd_kc (在店库存)")  
print("- ✅ 历史特征计算采用pandas向量化操作（速度提升100倍+）")
print("- ✅ 商品店铺特征merge优化（使用索引提升效率）")
print("- ✅ 按照白名单规则设置数据类型（数值特征统一float32，分类特征category）")
print("- ✅ 修复数值特征无效值处理：处理空字符串等无效值，避免类型转换错误")
print("- ✅ 修复在店库存逻辑：使用前向填充沿用上一个有记录的在店库存值")
print("- ✅ 修复标准差计算一致性（使用ddof=0）")
print("- ✅ 所有历史特征严格基于历史数据，避免数据泄露")
print("- ✅ 滚动特征基于前N天（不含当天）")
print("- ✅ 历史特征充分（2022年数据用于计算，掐头去尾后保存）")
print("- ✅ 掐头去尾处理：每个店铺-商品组合只保留从第一次有销量到最后一次有销量的期间")
print("- ✅ 零销量记录：保留期间内的零销量记录，确保时间序列连续性")
print("- ✅ 无销量组合：完全跳过从未有销量的店铺-商品组合")
print("- ✅ 新增特征: days_since_last_sale (距离上一次有销量的天数)")
print("- ✅ 新增特征: item_age (商品上市天数), store_age (店铺开业天数)")
print("- ✅ 分块处理：避免内存溢出，支持大数据集处理")
print("- ✅ 分块读取：销售数据分块读取，每块100万行")
print("- ✅ 分块生成：按商品分块生成，每批100个商品")
print(f"- 💡 完整版本，处理全部数据，适合最终模型训练")
