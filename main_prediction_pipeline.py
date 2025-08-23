#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
销量预测项目 - 主预测管道

简化的销量预测自动化系统
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_analysis import load_and_analyze_data, create_data_summary_plot
from baseline_model import BaselineModel, evaluate_baseline_model
from tft_model import TFTModel
from evaluation_utils import create_evaluation_visualization, print_evaluation_summary, save_evaluation_report

class SalesPredictionPipeline:
    """销量预测自动化管道"""
    
    def __init__(self, data_file='model_data_top10percent.csv'):
        """
        初始化预测管道
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.df = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.baseline_model = None
        self.tft_model = None
        self.baseline_results = {}
        self.tft_results = {}
        
        print("🚀 销量预测自动化管道初始化完成")
        print(f"📊 数据文件: {data_file}")
    
    def step1_data_analysis(self):
        """步骤1: 数据分析"""
        print("\n" + "="*60)
        print("📖 步骤1: 数据分析")
        print("="*60)
        
        try:
            # 加载和分析数据
            self.df, self.train_mask, self.val_mask, self.test_mask = load_and_analyze_data(self.data_file)
            
            # 创建数据概览图
            create_data_summary_plot(self.df, 'data_summary.png')
            
            print(f"✅ 数据分析完成")
            print(f"   - 总数据量: {len(self.df):,} 行")
            print(f"   - 训练集: {self.train_mask.sum():,} 行")
            print(f"   - 验证集: {self.val_mask.sum():,} 行")
            print(f"   - 测试集: {self.test_mask.sum():,} 行")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据分析失败: {str(e)}")
            return False
    
    def step2_baseline_model(self):
        """步骤2: 基线模型"""
        print("\n" + "="*60)
        print("🔧 步骤2: 基线模型")
        print("="*60)
        
        try:
            # 创建和训练基线模型
            self.baseline_model = BaselineModel()
            self.baseline_model.fit(self.df, self.train_mask)
            
            # 评估基线模型
            self.baseline_results = evaluate_baseline_model(
                self.baseline_model, self.df, self.val_mask, self.test_mask
            )
            
            print("✅ 基线模型完成")
            return True
            
        except Exception as e:
            print(f"❌ 基线模型失败: {str(e)}")
            return False
    
    def step3_tft_model(self):
        """步骤3: TFT模型"""
        print("\n" + "="*60)
        print("🤖 步骤3: TFT模型")
        print("="*60)
        
        try:
            # 创建TFT模型
            self.tft_model = TFTModel(
                prediction_length=30,
                encoder_length=90,
                learning_rate=0.0002,
                hidden_size=64,
                attention_head_size=8,
                dropout=0.2,
                hidden_continuous_size=32,
                batch_size=1024,  # 减小batch_size避免NaN
                max_epochs=30,   
                patience=8,
                random_seed=42
            )
            
            # 训练TFT模型
            self.tft_model.fit(self.data_file)
            
            # 获取预处理后的数据用于评估
            df = self.tft_model.load_and_preprocess_data(self.data_file)
            
            # 评估TFT模型
            self.tft_results = self.tft_model.predict_and_evaluate(df)
            
            print("✅ TFT模型完成")
            return True
            
        except Exception as e:
            print(f"❌ TFT模型失败: {str(e)}")
            return False
    
    def step4_evaluation(self):
        """步骤4: 评估和可视化"""
        print("\n" + "="*60)
        print("📊 步骤4: 评估和可视化")
        print("="*60)
        
        try:
            # 创建评估可视化
            create_evaluation_visualization(
                self.baseline_results, self.tft_results, 'evaluation_comparison.png'
            )
            
            # 打印评估摘要
            print_evaluation_summary(self.baseline_results, self.tft_results)
            
            # 保存评估报告
            save_evaluation_report(self.baseline_results, self.tft_results, 'evaluation_report.txt')
            
            print("✅ 评估和可视化完成")
            return True
            
        except Exception as e:
            print(f"❌ 评估失败: {str(e)}")
            return False
    
    def run_complete_pipeline(self):
        """运行完整的预测管道"""
        print("🚀 开始销量预测完整流程")
        print("="*80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        success_steps = 0
        total_steps = 4
        
        # 步骤1: 数据分析
        if self.step1_data_analysis():
            success_steps += 1
        else:
            print("❌ 管道在步骤1失败，停止执行")
            return False
        
        # 步骤2: 基线模型
        if self.step2_baseline_model():
            success_steps += 1
        else:
            print("❌ 管道在步骤2失败，停止执行")
            return False
        
        # 步骤3: TFT模型
        if self.step3_tft_model():
            success_steps += 1
        else:
            print("❌ 管道在步骤3失败，停止执行")
            return False
        
        # 步骤4: 评估和可视化
        if self.step4_evaluation():
            success_steps += 1
        else:
            print("❌ 管道在步骤4失败，停止执行")
            return False
        
        # 完成总结
        print("\n" + "="*80)
        print("🎉 销量预测项目完成!")
        print("="*80)
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"成功步骤: {success_steps}/{total_steps}")
        
        print("\n📋 生成的主要文件:")
        print("   📊 data_summary.png - 数据概览图")
        print("   📊 evaluation_comparison.png - 模型评估对比图")
        print("   📄 evaluation_report.txt - 详细评估报告")
        
        print("\n💡 项目总结:")
        print("   1. 数据分析: 完成数据加载和基础分析")
        print("   2. 基线模型: 基于去年同期销量的简单预测")
        print("   3. TFT模型: 基于深度学习的复杂时间序列预测")
        print("   4. 评估对比: 全面评估两个模型的性能")
        
        return True

def main():
    """主函数"""
    print("🚀 销量预测项目 - 主预测管道")
    
    # 创建管道实例
    pipeline = SalesPredictionPipeline('model_data_top10percent.csv')
    
    # 运行完整管道
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎊 项目成功完成！")
    else:
        print("\n😞 项目执行未完成，请检查错误信息")

if __name__ == "__main__":
    main() 