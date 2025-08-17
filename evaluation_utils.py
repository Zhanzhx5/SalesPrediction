#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
销量预测项目 - 评估模块

统一的模型评估和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_wape(actual, predicted):
    """计算WAPE (Weighted Absolute Percentage Error)"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 避免除零
    if np.sum(np.abs(actual)) == 0:
        return 0.0
    
    return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual))

def calculate_mae(actual, predicted):
    """计算MAE (Mean Absolute Error)"""
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

def calculate_rmse(actual, predicted):
    """计算RMSE (Root Mean Squared Error)"""
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

def evaluate_model(df, predictions, mask, model_name="模型"):
    """
    评估模型性能
    
    Args:
        df: 数据框
        predictions: 预测结果
        mask: 评估数据掩码
        model_name: 模型名称
    
    Returns:
        评估结果字典
    """
    if mask.sum() == 0:
        return {}
    
    actual = df[mask]['sales'].values
    pred = predictions[:len(actual)]  # 确保长度匹配
    
    # 基础指标
    wape = calculate_wape(actual, pred)
    mae = calculate_mae(actual, pred)
    rmse = calculate_rmse(actual, pred)
    
    # 总量指标
    total_actual = np.sum(actual)
    total_predicted = np.sum(pred)
    prediction_bias = (total_predicted - total_actual) / total_actual if total_actual > 0 else 0
    
    # 非零销量指标
    non_zero_mask = actual > 0
    non_zero_wape = calculate_wape(actual[non_zero_mask], pred[non_zero_mask]) if non_zero_mask.sum() > 0 else 0
    
    results = {
        'wape': wape,
        'mae': mae,
        'rmse': rmse,
        'total_actual': total_actual,
        'total_predicted': total_predicted,
        'prediction_bias': prediction_bias,
        'non_zero_wape': non_zero_wape,
        'non_zero_count': non_zero_mask.sum(),
        'record_count': len(actual)
    }
    
    print(f"✅ {model_name}评估完成 - WAPE: {wape:.4f}, MAE: {mae:.4f}")
    
    return results

def create_evaluation_visualization(baseline_results, tft_results, save_path='evaluation_comparison.png'):
    """
    创建模型评估对比可视化
    
    Args:
        baseline_results: 基线模型结果
        tft_results: TFT模型结果
        save_path: 保存路径
    """
    plt.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus']=False
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle('模型评估对比 (按月)', fontsize=16, fontweight='bold')
    
    # 准备数据
    datasets = ['test']  # 只对比test集
    models = ['Baseline', 'TFT']
    
    # 1. WAPE对比 (按月)
    wape_data = []
    for dataset in datasets:
        if dataset in baseline_results and dataset in tft_results:
            wape_data.append([baseline_results[dataset]['monthly_wape'], tft_results[dataset]['monthly_wape']])
    
    if wape_data:
        wape_data = np.array(wape_data)
        x = np.arange(len(datasets))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, wape_data[:, 0], width, label='Baseline', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, wape_data[:, 1], width, label='TFT', alpha=0.8, color='orange')
        axes[0, 0].set_title('WAPE对比 (按月)')
        axes[0, 0].set_ylabel('WAPE')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(datasets)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MAE对比 (按月)
    mae_data = []
    for dataset in datasets:
        if dataset in baseline_results and dataset in tft_results:
            mae_data.append([baseline_results[dataset]['monthly_mae'], tft_results[dataset]['monthly_mae']])
    
    if mae_data:
        mae_data = np.array(mae_data)
        axes[0, 1].bar(x - width/2, mae_data[:, 0], width, label='Baseline', alpha=0.8, color='skyblue')
        axes[0, 1].bar(x + width/2, mae_data[:, 1], width, label='TFT', alpha=0.8, color='orange')
        axes[0, 1].set_title('MAE对比 (按月)')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(datasets)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. RMSE对比 (按月)
    rmse_data = []
    for dataset in datasets:
        if dataset in baseline_results and dataset in tft_results:
            rmse_data.append([baseline_results[dataset]['monthly_rmse'], tft_results[dataset]['monthly_rmse']])
    
    if rmse_data:
        rmse_data = np.array(rmse_data)
        axes[0, 2].bar(x - width/2, rmse_data[:, 0], width, label='Baseline', alpha=0.8, color='skyblue')
        axes[0, 2].bar(x + width/2, rmse_data[:, 1], width, label='TFT', alpha=0.8, color='orange')
        axes[0, 2].set_title('RMSE对比 (按月)')
        axes[0, 2].set_ylabel('RMSE')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(datasets)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 预测偏差对比 (按月)
    bias_data = []
    for dataset in datasets:
        if dataset in baseline_results and dataset in tft_results:
            bias_data.append([baseline_results[dataset]['monthly_prediction_bias'], tft_results[dataset]['monthly_prediction_bias']])
    
    if bias_data:
        bias_data = np.array(bias_data)
        axes[1, 0].bar(x - width/2, bias_data[:, 0], width, label='Baseline', alpha=0.8, color='skyblue')
        axes[1, 0].bar(x + width/2, bias_data[:, 1], width, label='TFT', alpha=0.8, color='orange')
        axes[1, 0].set_title('预测偏差对比 (按月)')
        axes[1, 0].set_ylabel('预测偏差')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(datasets)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # 5. 总量对比（测试集按月）
    if 'test' in baseline_results and 'test' in tft_results:
        total_actual = baseline_results['test']['monthly_total_actual']
        baseline_total = baseline_results['test']['monthly_total_predicted']
        tft_total = tft_results['test']['monthly_total_predicted']
        
        axes[1, 1].bar(['实际总量', 'Baseline预测', 'TFT预测'], 
                      [total_actual, baseline_total, tft_total], 
                      color=['green', 'skyblue', 'orange'], alpha=0.8)
        axes[1, 1].set_title('测试集总量对比 (按月)')
        axes[1, 1].set_ylabel('总销量')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 非零WAPE对比 (按月)
    non_zero_wape_data = []
    for dataset in datasets:
        if dataset in baseline_results and dataset in tft_results:
            non_zero_wape_data.append([baseline_results[dataset]['monthly_non_zero_wape'], tft_results[dataset]['monthly_non_zero_wape']])
    
    if non_zero_wape_data:
        non_zero_wape_data = np.array(non_zero_wape_data)
        axes[1, 2].bar(x - width/2, non_zero_wape_data[:, 0], width, label='Baseline', alpha=0.8, color='skyblue')
        axes[1, 2].bar(x + width/2, non_zero_wape_data[:, 1], width, label='TFT', alpha=0.8, color='orange')
        axes[1, 2].set_title('非零销量WAPE对比 (按月)')
        axes[1, 2].set_ylabel('非零WAPE')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(datasets)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Test集每日销量折线图
    if ('test' in tft_results and 'detailed_predictions' in tft_results['test'] and 
        'test' in baseline_results and 'detailed_predictions' in baseline_results['test']):
        tft_test_df = tft_results['test']['detailed_predictions']
        baseline_test_df = baseline_results['test']['detailed_predictions']

        # 直接用 dtdate 分组
        tft_daily = tft_test_df.groupby('dtdate').agg({
            'sales': 'sum',
            'prediction': 'sum'
        }).reset_index()
        baseline_daily = baseline_test_df.groupby('dtdate').agg({
            'sales': 'sum',
            'prediction': 'sum'
        }).reset_index()

        # 确保 dtdate 是 datetime 类型
        tft_daily['dtdate'] = pd.to_datetime(tft_daily['dtdate'])
        baseline_daily['dtdate'] = pd.to_datetime(baseline_daily['dtdate'])

        # 画图
        axes[2, 0].plot(tft_daily['dtdate'], tft_daily['sales'], 's-', label='实际值', color='black', linewidth=2, markersize=4)
        axes[2, 0].plot(tft_daily['dtdate'], tft_daily['prediction'], 'o-', label='TFT预测', color='orange', linewidth=2, markersize=4)
        axes[2, 0].plot(baseline_daily['dtdate'], baseline_daily['prediction'], 'o-', label='Baseline预测', color='skyblue', linewidth=2, markersize=4)

        axes[2, 0].set_title('Test集每日销量对比')
        axes[2, 0].set_xlabel('日期')
        axes[2, 0].set_ylabel('总销量')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 设置 x 轴日期格式
        axes[2, 0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        axes[2, 0].tick_params(axis='x', rotation=45)
    
    # 8. 店铺WAPE分布箱线图 (按月)
    if ('test' in tft_results and 'detailed_predictions' in tft_results['test'] and 
        'test' in baseline_results and 'detailed_predictions' in baseline_results['test']):
        
        def prepare_store_wape(series_df, model_label):
            df_ = series_df.copy()
            grouped = df_.groupby('store_id', as_index=False).agg({'sales': 'sum', 'prediction': 'sum'})
            grouped = grouped[grouped['sales'] > 0]
            grouped['wape'] = (grouped['sales'] - grouped['prediction']).abs() / grouped['sales']
            grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=['wape'])
            grouped = grouped[grouped['wape'] <= 10]
            return pd.DataFrame({'模型': model_label, 'WAPE': grouped['wape'].astype(float)})
        
        store_wape_baseline = prepare_store_wape(baseline_results['test']['detailed_predictions'], 'Baseline')
        store_wape_tft = prepare_store_wape(tft_results['test']['detailed_predictions'], 'TFT')
        store_wape_long = pd.concat([store_wape_baseline, store_wape_tft], ignore_index=True)
        
        if len(store_wape_long) > 0:
            sns.boxplot(data=store_wape_long, x='模型', y='WAPE', ax=axes[2, 1], width=0.6, showfliers=True)
            sns.stripplot(data=store_wape_long, x='模型', y='WAPE', ax=axes[2, 1], color='black', alpha=0.4, size=3, jitter=0.2)
            axes[2, 1].set_title('店铺WAPE分布箱线图 (按月)')
            axes[2, 1].set_ylabel('WAPE')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, '无有效店铺WAPE数据', ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('店铺WAPE分布箱线图 (按月)')
    
    # 9. 非零销量WAPE分布箱线图 (按月)
    if ('test' in tft_results and 'detailed_predictions' in tft_results['test'] and 
        'test' in baseline_results and 'detailed_predictions' in baseline_results['test']):
        
        def prepare_nonzero_item_store_wape(series_df, model_label):
            df_ = series_df.copy()
            df_ = df_[df_['sales'] > 0]
            if len(df_) == 0:
                return pd.DataFrame({'模型': [], 'WAPE': []})
            grouped = df_.groupby(['store_id', 'item_id'], as_index=False).agg({'sales': 'sum', 'prediction': 'sum'})
            grouped = grouped[grouped['sales'] > 0]
            grouped['wape'] = (grouped['sales'] - grouped['prediction']).abs() / grouped['sales']
            grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=['wape'])
            grouped = grouped[grouped['wape'] <= 10]
            return pd.DataFrame({'模型': model_label, 'WAPE': grouped['wape'].astype(float)})
        
        nonzero_wape_baseline = prepare_nonzero_item_store_wape(baseline_results['test']['detailed_predictions'], 'Baseline')
        nonzero_wape_tft = prepare_nonzero_item_store_wape(tft_results['test']['detailed_predictions'], 'TFT')
        nonzero_wape_long = pd.concat([nonzero_wape_baseline, nonzero_wape_tft], ignore_index=True)
        
        if len(nonzero_wape_long) > 0:
            sns.boxplot(data=nonzero_wape_long, x='模型', y='WAPE', ax=axes[2, 2], width=0.6, showfliers=True)
            sns.stripplot(data=nonzero_wape_long, x='模型', y='WAPE', ax=axes[2, 2], color='black', alpha=0.4, size=2.5, jitter=0.2)
            axes[2, 2].set_title('非零销量WAPE分布箱线图 (按月)')
            axes[2, 2].set_ylabel('WAPE')
            axes[2, 2].grid(True, alpha=0.3)
        else:
            axes[2, 2].text(0.5, 0.5, '无有效非零WAPE数据', ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('非零销量WAPE分布箱线图 (按月)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 评估对比图已保存: {save_path}")

def print_evaluation_summary(baseline_results, tft_results):
    """
    打印评估结果摘要
    
    Args:
        baseline_results: 基线模型结果
        tft_results: TFT模型结果
    """
    print("\n" + "="*80)
    print("📋 模型评估结果摘要")
    print("="*80)
    
    datasets = ['validation', 'test']
    
    for dataset in datasets:
        print(f"\n{dataset.upper()} 集:")
        print("-" * 40)
        
        if dataset in baseline_results and dataset in tft_results:
            baseline = baseline_results[dataset]
            tft = tft_results[dataset]
            
            # 按天指标
            print("按天指标:")
            print(f"{'指标':<15} {'Baseline':<12} {'TFT':<12} {'改进':<12}")
            print("-" * 60)
            
            # WAPE
            wape_improvement = baseline['daily_wape'] - tft['daily_wape']
            print(f"{'WAPE':<15} {baseline['daily_wape']:<12.4f} {tft['daily_wape']:<12.4f} {wape_improvement:<12.4f}")
            
            # MAE
            mae_improvement = baseline['daily_mae'] - tft['daily_mae']
            print(f"{'MAE':<15} {baseline['daily_mae']:<12.4f} {tft['daily_mae']:<12.4f} {mae_improvement:<12.4f}")
            
            # RMSE
            rmse_improvement = baseline['daily_rmse'] - tft['daily_rmse']
            print(f"{'RMSE':<15} {baseline['daily_rmse']:<12.4f} {tft['daily_rmse']:<12.4f} {rmse_improvement:<12.4f}")
            
            # 预测偏差
            bias_improvement = abs(baseline['daily_prediction_bias']) - abs(tft['daily_prediction_bias'])
            print(f"{'预测偏差':<15} {baseline['daily_prediction_bias']:<12.4f} {tft['daily_prediction_bias']:<12.4f} {bias_improvement:<12.4f}")
            
            # 非零WAPE
            non_zero_improvement = baseline['daily_non_zero_wape'] - tft['daily_non_zero_wape']
            print(f"{'非零WAPE':<15} {baseline['daily_non_zero_wape']:<12.4f} {tft['daily_non_zero_wape']:<12.4f} {non_zero_improvement:<12.4f}")
            
            print(f"\n按天总量对比:")
            print(f"  实际总量: {baseline['daily_total_actual']:.2f}")
            print(f"  Baseline预测: {baseline['daily_total_predicted']:.2f}")
            print(f"  TFT预测: {tft['daily_total_predicted']:.2f}")
            
            # 按月指标
            print("\n按月指标:")
            print(f"{'指标':<15} {'Baseline':<12} {'TFT':<12} {'改进':<12}")
            print("-" * 60)
            
            # WAPE
            wape_improvement = baseline['monthly_wape'] - tft['monthly_wape']
            print(f"{'WAPE':<15} {baseline['monthly_wape']:<12.4f} {tft['monthly_wape']:<12.4f} {wape_improvement:<12.4f}")
            
            # MAE
            mae_improvement = baseline['monthly_mae'] - tft['monthly_mae']
            print(f"{'MAE':<15} {baseline['monthly_mae']:<12.4f} {tft['monthly_mae']:<12.4f} {mae_improvement:<12.4f}")
            
            # RMSE
            rmse_improvement = baseline['monthly_rmse'] - tft['monthly_rmse']
            print(f"{'RMSE':<15} {baseline['monthly_rmse']:<12.4f} {tft['monthly_rmse']:<12.4f} {rmse_improvement:<12.4f}")
            
            # 预测偏差
            bias_improvement = abs(baseline['monthly_prediction_bias']) - abs(tft['monthly_prediction_bias'])
            print(f"{'预测偏差':<15} {baseline['monthly_prediction_bias']:<12.4f} {tft['monthly_prediction_bias']:<12.4f} {bias_improvement:<12.4f}")
            
            # 非零WAPE
            non_zero_improvement = baseline['monthly_non_zero_wape'] - tft['monthly_non_zero_wape']
            print(f"{'非零WAPE':<15} {baseline['monthly_non_zero_wape']:<12.4f} {tft['monthly_non_zero_wape']:<12.4f} {non_zero_improvement:<12.4f}")
            
            print(f"\n按月总量对比:")
            print(f"  实际总量: {baseline['monthly_total_actual']:.2f}")
            print(f"  Baseline预测: {baseline['monthly_total_predicted']:.2f}")
            print(f"  TFT预测: {tft['monthly_total_predicted']:.2f}")
        else:
            print("  数据不完整，无法对比")

def save_evaluation_report(baseline_results, tft_results, save_path='evaluation_report.txt'):
    """
    保存评估报告
    
    Args:
        baseline_results: 基线模型结果
        tft_results: TFT模型结果
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("销量预测项目 - 模型评估报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        datasets = ['validation', 'test']
        
        for dataset in datasets:
            f.write(f"{dataset.upper()} 集评估结果:\n")
            f.write("-" * 40 + "\n")
            
            if dataset in baseline_results and dataset in tft_results:
                baseline = baseline_results[dataset]
                tft = tft_results[dataset]
                
                # 按天指标
                f.write("按天指标:\n")
                f.write(f"{'指标':<15} {'Baseline':<12} {'TFT':<12} {'改进':<12}\n")
                f.write("-" * 60 + "\n")
                
                # WAPE
                wape_improvement = baseline['daily_wape'] - tft['daily_wape']
                f.write(f"{'WAPE':<15} {baseline['daily_wape']:<12.4f} {tft['daily_wape']:<12.4f} {wape_improvement:<12.4f}\n")
                
                # MAE
                mae_improvement = baseline['daily_mae'] - tft['daily_mae']
                f.write(f"{'MAE':<15} {baseline['daily_mae']:<12.4f} {tft['daily_mae']:<12.4f} {mae_improvement:<12.4f}\n")
                
                # RMSE
                rmse_improvement = baseline['daily_rmse'] - tft['daily_rmse']
                f.write(f"{'RMSE':<15} {baseline['daily_rmse']:<12.4f} {tft['daily_rmse']:<12.4f} {rmse_improvement:<12.4f}\n")
                
                # 预测偏差
                bias_improvement = abs(baseline['daily_prediction_bias']) - abs(tft['daily_prediction_bias'])
                f.write(f"{'预测偏差':<15} {baseline['daily_prediction_bias']:<12.4f} {tft['daily_prediction_bias']:<12.4f} {bias_improvement:<12.4f}\n")
                
                # 非零WAPE
                non_zero_improvement = baseline['daily_non_zero_wape'] - tft['daily_non_zero_wape']
                f.write(f"{'非零WAPE':<15} {baseline['daily_non_zero_wape']:<12.4f} {tft['daily_non_zero_wape']:<12.4f} {non_zero_improvement:<12.4f}\n")
                
                f.write(f"\n按天总量对比:\n")
                f.write(f"  实际总量: {baseline['daily_total_actual']:.2f}\n")
                f.write(f"  Baseline预测: {baseline['daily_total_predicted']:.2f}\n")
                f.write(f"  TFT预测: {tft['daily_total_predicted']:.2f}\n")
                
                # 按月指标
                f.write("\n按月指标:\n")
                f.write(f"{'指标':<15} {'Baseline':<12} {'TFT':<12} {'改进':<12}\n")
                f.write("-" * 60 + "\n")
                
                # WAPE
                wape_improvement = baseline['monthly_wape'] - tft['monthly_wape']
                f.write(f"{'WAPE':<15} {baseline['monthly_wape']:<12.4f} {tft['monthly_wape']:<12.4f} {wape_improvement:<12.4f}\n")
                
                # MAE
                mae_improvement = baseline['monthly_mae'] - tft['monthly_mae']
                f.write(f"{'MAE':<15} {baseline['monthly_mae']:<12.4f} {tft['monthly_mae']:<12.4f} {mae_improvement:<12.4f}\n")
                
                # RMSE
                rmse_improvement = baseline['monthly_rmse'] - tft['monthly_rmse']
                f.write(f"{'RMSE':<15} {baseline['monthly_rmse']:<12.4f} {tft['monthly_rmse']:<12.4f} {rmse_improvement:<12.4f}\n")
                
                # 预测偏差
                bias_improvement = abs(baseline['monthly_prediction_bias']) - abs(tft['monthly_prediction_bias'])
                f.write(f"{'预测偏差':<15} {baseline['monthly_prediction_bias']:<12.4f} {tft['monthly_prediction_bias']:<12.4f} {bias_improvement:<12.4f}\n")
                
                # 非零WAPE
                non_zero_improvement = baseline['monthly_non_zero_wape'] - tft['monthly_non_zero_wape']
                f.write(f"{'非零WAPE':<15} {baseline['monthly_non_zero_wape']:<12.4f} {tft['monthly_non_zero_wape']:<12.4f} {non_zero_improvement:<12.4f}\n")
                
                f.write(f"\n按月总量对比:\n")
                f.write(f"  实际总量: {baseline['monthly_total_actual']:.2f}\n")
                f.write(f"  Baseline预测: {baseline['monthly_total_predicted']:.2f}\n")
                f.write(f"  TFT预测: {tft['monthly_total_predicted']:.2f}\n")
            else:
                f.write("  数据不完整，无法对比\n")
            
            f.write("\n")
    
    print(f"📄 评估报告已保存: {save_path}")

if __name__ == "__main__":
    # 测试评估模块
    print("🧪 测试评估模块")
    
    # 模拟数据
    np.random.seed(42)
    actual = np.random.exponential(2, 1000)
    baseline_pred = actual * (1 + np.random.normal(0, 0.3, 1000))
    tft_pred = actual * (1 + np.random.normal(0, 0.2, 1000))
    
    # 创建模拟结果
    baseline_results = {
        'validation': evaluate_model(pd.DataFrame({'sales': actual}), baseline_pred, np.ones(1000, dtype=bool), "Baseline"),
        'test': evaluate_model(pd.DataFrame({'sales': actual}), baseline_pred, np.ones(1000, dtype=bool), "Baseline")
    }
    
    tft_results = {
        'validation': evaluate_model(pd.DataFrame({'sales': actual}), tft_pred, np.ones(1000, dtype=bool), "TFT"),
        'test': evaluate_model(pd.DataFrame({'sales': actual}), tft_pred, np.ones(1000, dtype=bool), "TFT")
    }
    
    # 创建可视化
    create_evaluation_visualization(baseline_results, tft_results)
    
    # 打印摘要
    print_evaluation_summary(baseline_results, tft_results)
    
    # 保存报告
    save_evaluation_report(baseline_results, tft_results) 