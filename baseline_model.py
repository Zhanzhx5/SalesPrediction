#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
销量预测项目 - 基线模型

简化的基线模型：按历史销量预测今年销量
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BaselineModel:
    """历史同期销量基线模型"""
    
    def __init__(self):
        """初始化基线模型"""
        self.is_fitted = False
        self.historical_data = None
        print("🚀 基线模型初始化完成")
        
    def fit(self, df, train_mask):
        """
        训练基线模型（存储历史数据）
        
        Args:
            df: 数据框
            train_mask: 训练集掩码
        """
        print("🔧 训练基线模型...")
        
        # 转换日期格式
        df['dtdate'] = pd.to_datetime(df['dtdate'])
        
        # 获取训练数据
        train_data = df[train_mask].copy()
        
        # 存储历史数据，用于查找同期
        self.historical_data = train_data.groupby(['store_id', 'item_id', 'month', 'day']).agg({
            'sales': ['mean', 'median', 'sum', 'count']
        }).reset_index()
        
        # 扁平化列名
        self.historical_data.columns = ['store_id', 'item_id', 'month', 'day', 
                                      'sales_mean', 'sales_median', 'sales_sum', 'sales_count']
        
        self.is_fitted = True
        print(f"✅ 基线模型训练完成，历史数据: {len(self.historical_data)} 条记录")
        
    def predict(self, df, prediction_mask):
        """
        预测
        
        Args:
            df: 数据框
            prediction_mask: 预测掩码
            
        Returns:
            预测结果数组
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        print("🔮 基线模型预测中...")
        
        # 获取预测数据
        pred_data = df[prediction_mask].copy()
        pred_data['dtdate'] = pd.to_datetime(pred_data['dtdate'])
        
        # 添加月日信息用于匹配
        pred_data['month'] = pred_data['dtdate'].dt.month
        pred_data['day'] = pred_data['dtdate'].dt.day
        
        # 合并历史数据
        merged = pred_data.merge(
            self.historical_data,
            on=['store_id', 'item_id', 'month', 'day'],
            how='left'
        )

        # 如果没有历史同期，则使用近3个月同日的平均值
        # 计算前1/2/3个月
        pred_data = merged.copy()
        pred_data['prev1_month'] = ((pred_data['month'] - 2) % 12) + 1
        pred_data['prev2_month'] = ((pred_data['month'] - 3) % 12) + 1
        pred_data['prev3_month'] = ((pred_data['month'] - 4) % 12) + 1

        # 为前1个月合并均值
        tmp1 = self.historical_data[['store_id', 'item_id', 'month', 'day', 'sales_mean']].copy()
        tmp1.columns = ['store_id', 'item_id', 'prev1_month', 'day', 'sales_mean_prev1']
        pred_data = pred_data.merge(
            tmp1,
            on=['store_id', 'item_id', 'prev1_month', 'day'],
            how='left'
        )

        # 为前2个月合并均值
        tmp2 = self.historical_data[['store_id', 'item_id', 'month', 'day', 'sales_mean']].copy()
        tmp2.columns = ['store_id', 'item_id', 'prev2_month', 'day', 'sales_mean_prev2']
        pred_data = pred_data.merge(
            tmp2,
            on=['store_id', 'item_id', 'prev2_month', 'day'],
            how='left'
        )

        # 为前3个月合并均值
        tmp3 = self.historical_data[['store_id', 'item_id', 'month', 'day', 'sales_mean']].copy()
        tmp3.columns = ['store_id', 'item_id', 'prev3_month', 'day', 'sales_mean_prev3']
        pred_data = pred_data.merge(
            tmp3,
            on=['store_id', 'item_id', 'prev3_month', 'day'],
            how='left'
        )

        # 计算回退均值（忽略缺失）
        fallback_mean = pred_data[[
            'sales_mean_prev1', 'sales_mean_prev2', 'sales_mean_prev3'
        ]].mean(axis=1, skipna=True)

        # 基线预测：历史同期均值和近3个月同日均值都乘以110%涨幅；若都缺失则置0
        baseline_pred = (pred_data['sales_mean'] * 1.1).fillna(fallback_mean * 1.1).fillna(0)
        
        print(f"✅ 基线预测完成，预测数据: {len(baseline_pred)} 条")
        return baseline_pred.values

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

def evaluate_baseline_model(model, df, val_mask, test_mask):
    """
    评估基线模型
    
    Args:
        model: 训练好的基线模型
        df: 数据框
        val_mask: 验证集掩码
        test_mask: 测试集掩码
        
    Returns:
        评估结果字典
    """
    print("📊 评估基线模型...")
    
    results = {}
    
    # 验证集评估
    if val_mask.sum() > 0:
        val_df = df[val_mask].copy()
        val_df['dtdate'] = pd.to_datetime(val_df['dtdate'])
        val_pred = model.predict(df, val_mask)
        
        # 将预测值添加到DataFrame中
        val_df['prediction'] = val_pred
        
        # 按天评估验证集指标
        val_daily_actual = val_df['sales'].values
        val_daily_pred = val_df['prediction'].values
        val_daily_non_zero_mask = val_daily_actual > 0
        val_daily_non_zero_wape = calculate_wape(val_daily_actual[val_daily_non_zero_mask], val_daily_pred[val_daily_non_zero_mask]) if val_daily_non_zero_mask.sum() > 0 else 0
        
        # 按月评估验证集指标
        val_df['year_month'] = val_df['dtdate'].dt.to_period('M')
        val_monthly = val_df.groupby(['store_id', 'item_id', 'year_month']).agg({
            'sales': 'sum',
            'prediction': 'sum'
        }).reset_index()
        
        val_monthly_actual = val_monthly['sales'].values
        val_monthly_pred = val_monthly['prediction'].values
        val_monthly_non_zero_mask = val_monthly_actual > 0
        val_monthly_non_zero_wape = calculate_wape(val_monthly_actual[val_monthly_non_zero_mask], val_monthly_pred[val_monthly_non_zero_mask]) if val_monthly_non_zero_mask.sum() > 0 else 0
        
        results['validation'] = {
            # 按天指标
            'daily_wape': calculate_wape(val_daily_actual, val_daily_pred),
            'daily_mae': calculate_mae(val_daily_actual, val_daily_pred),
            'daily_rmse': calculate_rmse(val_daily_actual, val_daily_pred),
            'daily_total_actual': np.sum(val_daily_actual),
            'daily_total_predicted': np.sum(val_daily_pred),
            'daily_prediction_bias': (np.sum(val_daily_pred) - np.sum(val_daily_actual)) / np.sum(val_daily_actual) if np.sum(val_daily_actual) > 0 else 0,
            'daily_non_zero_wape': val_daily_non_zero_wape,
            'daily_non_zero_count': val_daily_non_zero_mask.sum(),
            'daily_record_count': len(val_daily_actual),
            
            # 按月指标
            'monthly_wape': calculate_wape(val_monthly_actual, val_monthly_pred),
            'monthly_mae': calculate_mae(val_monthly_actual, val_monthly_pred),
            'monthly_rmse': calculate_rmse(val_monthly_actual, val_monthly_pred),
            'monthly_total_actual': np.sum(val_monthly_actual),
            'monthly_total_predicted': np.sum(val_monthly_pred),
            'monthly_prediction_bias': (np.sum(val_monthly_pred) - np.sum(val_monthly_actual)) / np.sum(val_monthly_actual) if np.sum(val_monthly_actual) > 0 else 0,
            'monthly_non_zero_wape': val_monthly_non_zero_wape,
            'monthly_non_zero_count': val_monthly_non_zero_mask.sum(),
            'monthly_record_count': len(val_monthly_actual),
            
            # 详细预测数据
            'detailed_predictions': val_df[['store_id', 'item_id', 'sales', 'dtdate', 'prediction', 'sdeptname']]
        }
        
        print(f"✅ 验证集评估完成 - 按天WAPE: {results['validation']['daily_wape']:.4f}, 按月WAPE: {results['validation']['monthly_wape']:.4f}")
    
    # 测试集评估
    if test_mask.sum() > 0:
        test_df = df[test_mask].copy()
        test_df['dtdate'] = pd.to_datetime(test_df['dtdate'])
        test_pred = model.predict(df, test_mask)
        
        # 将预测值添加到DataFrame中
        test_df['prediction'] = test_pred
        
        # 按天评估测试集指标
        test_daily_actual = test_df['sales'].values
        test_daily_pred = test_df['prediction'].values
        test_daily_non_zero_mask = test_daily_actual > 0
        test_daily_non_zero_wape = calculate_wape(test_daily_actual[test_daily_non_zero_mask], test_daily_pred[test_daily_non_zero_mask]) if test_daily_non_zero_mask.sum() > 0 else 0
        
        # 按月评估测试集指标
        test_df['year_month'] = test_df['dtdate'].dt.to_period('M')
        test_monthly = test_df.groupby(['store_id', 'item_id', 'year_month']).agg({
            'sales': 'sum',
            'prediction': 'sum'
        }).reset_index()
        
        test_monthly_actual = test_monthly['sales'].values
        test_monthly_pred = test_monthly['prediction'].values
        test_monthly_non_zero_mask = test_monthly_actual > 0
        test_monthly_non_zero_wape = calculate_wape(test_monthly_actual[test_monthly_non_zero_mask], test_monthly_pred[test_monthly_non_zero_mask]) if test_monthly_non_zero_mask.sum() > 0 else 0
        
        # 创建详细的预测数据DataFrame
        test_detailed_df = test_df[['store_id', 'item_id', 'sales', 'dtdate', 'prediction', 'sdeptname']].copy()
        
        # 保存测试集预测结果到CSV文件
        cols_to_save = ['store_id', 'item_id', 'sales', 'dtdate', 'prediction', 'sdeptname']
        test_detailed_df[cols_to_save].to_csv('baseline_test_predictions.csv', index=False)
        print("✅ 已保存Baseline测试集详细预测结果到 baseline_test_predictions.csv")
        
        results['test'] = {
            # 按天指标
            'daily_wape': calculate_wape(test_daily_actual, test_daily_pred),
            'daily_mae': calculate_mae(test_daily_actual, test_daily_pred),
            'daily_rmse': calculate_rmse(test_daily_actual, test_daily_pred),
            'daily_total_actual': np.sum(test_daily_actual),
            'daily_total_predicted': np.sum(test_daily_pred),
            'daily_prediction_bias': (np.sum(test_daily_pred) - np.sum(test_daily_actual)) / np.sum(test_daily_actual) if np.sum(test_daily_actual) > 0 else 0,
            'daily_non_zero_wape': test_daily_non_zero_wape,
            'daily_non_zero_count': test_daily_non_zero_mask.sum(),
            'daily_record_count': len(test_daily_actual),
            
            # 按月指标
            'monthly_wape': calculate_wape(test_monthly_actual, test_monthly_pred),
            'monthly_mae': calculate_mae(test_monthly_actual, test_monthly_pred),
            'monthly_rmse': calculate_rmse(test_monthly_actual, test_monthly_pred),
            'monthly_total_actual': np.sum(test_monthly_actual),
            'monthly_total_predicted': np.sum(test_monthly_pred),
            'monthly_prediction_bias': (np.sum(test_monthly_pred) - np.sum(test_monthly_actual)) / np.sum(test_monthly_actual) if np.sum(test_monthly_actual) > 0 else 0,
            'monthly_non_zero_wape': test_monthly_non_zero_wape,
            'monthly_non_zero_count': test_monthly_non_zero_mask.sum(),
            'monthly_record_count': len(test_monthly_actual),
            
            # 详细预测数据
            'detailed_predictions': test_detailed_df
        }
        
        print(f"✅ 测试集评估完成 - 按天WAPE: {results['test']['daily_wape']:.4f}, 按月WAPE: {results['test']['monthly_wape']:.4f}")
    
    return results

if __name__ == "__main__":
    # 测试基线模型
    from data_analysis import load_and_analyze_data
    
    # 加载数据
    df, train_mask, val_mask, test_mask = load_and_analyze_data()
    
    # 训练基线模型
    model = BaselineModel()
    model.fit(df, train_mask)
    
    # 评估模型
    results = evaluate_baseline_model(model, df, val_mask, test_mask)
    
    print("\n📋 基线模型评估结果:")
    for dataset, metrics in results.items():
        print(f"\n{dataset.upper()} 集:")
        print("按天指标:")
        print(f"  daily_wape: {metrics['daily_wape']:.4f}")
        print(f"  daily_mae: {metrics['daily_mae']:.4f}")
        print(f"  daily_rmse: {metrics['daily_rmse']:.4f}")
        print(f"  daily_prediction_bias: {metrics['daily_prediction_bias']:.4f}")
        print(f"  daily_non_zero_wape: {metrics['daily_non_zero_wape']:.4f}")
        print("按月指标:")
        print(f"  monthly_wape: {metrics['monthly_wape']:.4f}")
        print(f"  monthly_mae: {metrics['monthly_mae']:.4f}")
        print(f"  monthly_rmse: {metrics['monthly_rmse']:.4f}")
        print(f"  monthly_prediction_bias: {metrics['monthly_prediction_bias']:.4f}")
        print(f"  monthly_non_zero_wape: {metrics['monthly_non_zero_wape']:.4f}") 