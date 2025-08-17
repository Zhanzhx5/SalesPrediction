#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
销量预测项目 - 数据分析模块

简化的数据分析模块，专注于数据加载、基础分析和数据集划分
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data(file_path='model_data_mini_shaping.csv'):
    """
    加载和分析数据，返回处理后的数据框和数据集划分掩码
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        tuple: (df, train_mask, val_mask, test_mask)
    """
    print("🔍 开始加载数据...")
    
    # 加载数据
    df = pd.read_csv(file_path)
    print(f"✅ 数据加载完成: {df.shape[0]:,} 行, {df.shape[1]} 列")
    
    # 转换日期格式
    df['dtdate'] = pd.to_datetime(df['dtdate'])
    
    # 基础数据检查
    print(f"📊 数据基本信息:")
    print(f"   - 时间范围: {df['dtdate'].min().strftime('%Y-%m-%d')} 至 {df['dtdate'].max().strftime('%Y-%m-%d')}")
    print(f"   - 店铺数量: {df['store_id'].nunique()}")
    print(f"   - 商品数量: {df['item_id'].nunique()}")
    print(f"   - 总记录数: {len(df):,}")
    
    # 处理销量数据
    negative_count = (df['sales'] < 0).sum()
    if negative_count > 0:
        print(f"⚠️  发现 {negative_count} 个负销量值，将裁剪为0")
        df['sales'] = df['sales'].clip(lower=0)
    
    
    # ======================= 诊断代码：开始数据法医排查 =======================
    print("\n🕵️ 开始数据法医排查...")

    # 1. 检查所有数值列中是否存在 NaN (空值)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    nan_counts = df[numerical_cols].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print("\n❌ 发现NaN值，这是导致爆炸的头号嫌疑！")
        print(nan_cols)
        print("-> 正在用0填充NaN值...")
        df.fillna(0, inplace=True)
    else:
        print("✅ 所有数值列均无NaN值。")

    # 2. 检查所有数值列中是否存在 Inf (无穷大值)
    inf_counts = df[numerical_cols].isin([np.inf, -np.inf]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if not inf_cols.empty:
        print("\n❌ 发现无穷大值（Inf），这也是导致爆炸的头号嫌疑！")
        print(inf_cols)
        print("-> 正在用0替换无穷大值...")
        df.replace([np.inf, -np.inf], 0, inplace=True)
    else:
        print("✅ 所有数值列均无无穷大值。")

    # 3. 检查是否存在极端异常值
    print("\n🔍 检查数值特征的范围...")
    pd.options.display.float_format = '{:.2f}'.format
    print(df[numerical_cols].describe().T[["min", "max", "mean", "std"]])

    print("🕵️ 数据法医排查结束。\n")
    # ======================= 诊断代码：结束 =======================

    # 数据集划分
    train_start = datetime(2023, 1, 10)
    train_end = datetime(2025, 4, 30)
    val_start = datetime(2025, 5, 1)
    val_end = datetime(2025, 5, 30)
    test_start = datetime(2025, 6, 1)
    test_end = datetime(2025, 6, 30)
    
    train_mask = (df['dtdate'] >= train_start) & (df['dtdate'] <= train_end)
    val_mask = (df['dtdate'] >= val_start) & (df['dtdate'] <= val_end)
    test_mask = (df['dtdate'] >= test_start) & (df['dtdate'] <= test_end)
    
    print(f"\n📅 数据集划分:")
    print(f"   - 训练集: {train_start.strftime('%Y-%m-%d')} 至 {train_end.strftime('%Y-%m-%d')} - {train_mask.sum():,} 行")
    print(f"   - 验证集: {val_start.strftime('%Y-%m-%d')} 至 {val_end.strftime('%Y-%m-%d')} - {val_mask.sum():,} 行")
    print(f"   - 测试集: {test_start.strftime('%Y-%m-%d')} 至 {test_end.strftime('%Y-%m-%d')} - {test_mask.sum():,} 行")
    
    # 销量统计
    print(f"\n💰 销量统计:")
    print(f"   - 商品-店铺组合每日平均销量: {df['sales'].mean():.2f}")
    print(f"   - 商品-店铺组合单日最大销量: {df['sales'].max():.2f}")
    print(f"   - 商品-店铺组合销量方差: {df['sales'].var():.2f}")
    print(f"   - 商品-店铺组合零销量比例: {(df['sales'] == 0).mean()*100:.1f}%")
    
    return df, train_mask, val_mask, test_mask

def create_data_summary_plot(df, save_path='data_summary.png'):
    """创建数据概览图"""
    plt.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus']=False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 销量分布（分别显示零销量和有销量）
    zero_sales = (df['sales'] == 0).sum()
    non_zero_sales = (df['sales'] > 0).sum()
    
    axes[0, 0].bar(['零销量', '有销量'], [zero_sales, non_zero_sales], 
                   color=['lightgray', 'skyblue'], alpha=0.8)
    axes[0, 0].set_title('销量分布（零销量 vs 有销量）')
    axes[0, 0].set_ylabel('记录数')
    
    # 在柱状图上添加数值标签
    for i, v in enumerate([zero_sales, non_zero_sales]):
        axes[0, 0].text(i, v + v*0.01, f'{v:,}', ha='center', va='bottom', fontsize=10)
    
    # 2. 时间序列销量趋势
    daily_sales = df.groupby('dtdate')['sales'].sum()
    axes[0, 1].plot(daily_sales.index, daily_sales.values, alpha=0.7, linewidth=1)
    axes[0, 1].set_title('日销量趋势')
    axes[0, 1].set_xlabel('日期')
    axes[0, 1].set_ylabel('总销量')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 店铺销量分布
    store_sales = df.groupby('store_id')['sales'].sum().sort_values(ascending=False)
    axes[1, 0].bar(range(len(store_sales)), store_sales.values, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('店铺销量分布')
    axes[1, 0].set_xlabel('店铺')
    axes[1, 0].set_ylabel('总销量')
    
    # 4. 商品销量分布（只显示有销量的商品）
    item_sales = df.groupby('item_id')['sales'].sum()
    non_zero_item_sales = item_sales[item_sales > 0].sort_values(ascending=False)
    
    if len(non_zero_item_sales) > 0:
        # 使用对数尺度显示商品销量分布
        axes[1, 1].hist(non_zero_item_sales, bins=30, alpha=0.7, color='orange', log=True)
        axes[1, 1].set_title('商品销量分布（有销量商品，对数尺度）')
        axes[1, 1].set_xlabel('商品总销量')
        axes[1, 1].set_ylabel('商品数量（对数尺度）')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, '没有有销量的商品', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('商品销量分布')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 数据概览图已保存: {save_path}")
    
    # 额外打印一些有用的统计信息
    print(f"\n📈 销量详细统计:")
    print(f"   - 零销量记录: {zero_sales:,} ({zero_sales/len(df)*100:.1f}%)")
    print(f"   - 有销量记录: {non_zero_sales:,} ({non_zero_sales/len(df)*100:.1f}%)")
    if len(non_zero_item_sales) > 0:
        total_items = df['item_id'].nunique()
        print(f"   - 有销量商品数: {len(non_zero_item_sales):,} ({len(non_zero_item_sales)/total_items*100:.1f}%)")
        print(f"   - 商品销量范围（店铺无关）: {non_zero_item_sales.min():.2f} - {non_zero_item_sales.max():.2f}")
        print(f"   - 商品销量中位数（店铺无关）: {non_zero_item_sales.median():.2f}")
        print(f"   - 商品销量均值（店铺无关）: {non_zero_item_sales.mean():.2f}")
        print(f"   - 商品销量方差（店铺无关）: {non_zero_item_sales.var():.2f}")

if __name__ == "__main__":
    # 测试数据分析
    df, train_mask, val_mask, test_mask = load_and_analyze_data()
    create_data_summary_plot(df) 