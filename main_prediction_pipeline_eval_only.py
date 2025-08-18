#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
销量预测项目 - 仅评估主预测管道（跳过TFT训练）
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_analysis import load_and_analyze_data, create_data_summary_plot
from baseline_model import BaselineModel, evaluate_baseline_model
from tft_model import TFTModel
from evaluation_utils import create_evaluation_visualization, print_evaluation_summary, save_evaluation_report

class SalesPredictionEvalOnlyPipeline:
    """销量预测自动化管道（仅评估TFT，不重新训练）"""
    def __init__(self, data_file='model_data_top10percent.csv'):
        self.data_file = data_file
        self.df = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.baseline_model = None
        self.tft_model = None
        self.baseline_results = {}
        self.tft_results = {}

        print("🚀 仅评估管道初始化完成")
        print(f"📊 数据文件: {data_file}")

    def step1_data_analysis(self):
        print("\n" + "="*60)
        print("📖 步骤1: 数据分析")
        print("="*60)
        try:
            self.df, self.train_mask, self.val_mask, self.test_mask = load_and_analyze_data(self.data_file)
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
        print("\n" + "="*60)
        print("🔧 步骤2: 基线模型")
        print("="*60)
        try:
            self.baseline_model = BaselineModel()
            self.baseline_model.fit(self.df, self.train_mask)
            self.baseline_results = evaluate_baseline_model(
                self.baseline_model, self.df, self.val_mask, self.test_mask
            )
            print("✅ 基线模型完成")
            return True
        except Exception as e:
            print(f"❌ 基线模型失败: {str(e)}")
            return False

    def step3_tft_evaluate_only(self):
        print("\n" + "="*60)
        print("🤖 步骤3: TFT模型（仅评估）")
        print("="*60)
        try:
            # 只初始化TFTModel，不训练
            self.tft_model = TFTModel(
                prediction_length=30,
                encoder_length=90,
                learning_rate=0.0001,
                hidden_size=64,
                attention_head_size=8,
                dropout=0.2,
                hidden_continuous_size=32,
                batch_size=256,  # 减小batch_size避免NaN
                max_epochs=30,  
                patience=5,
                random_seed=42
            )
            # 跳过fit和train，直接加载数据和dataset
            df = self.tft_model.load_and_preprocess_data(self.data_file)
            feature_groups = self.tft_model.define_feature_groups()
            self.tft_model.create_timeseries_dataset(df, feature_groups)
            self.tft_model.create_model()  # 需要初始化模型结构以便后续加载权重

            # 直接评估（会自动加载最佳checkpoint）
            self.tft_results = self.tft_model.predict_and_evaluate(df)
            print("✅ TFT模型评估完成")
            return True
        except Exception as e:
            print(f"❌ TFT模型评估失败: {str(e)}")
            return False

    def step4_evaluation(self):
        print("\n" + "="*60)
        print("📊 步骤4: 评估和可视化")
        print("="*60)
        try:
            create_evaluation_visualization(
                self.baseline_results, self.tft_results, 'evaluation_comparison.png'
            )
            print_evaluation_summary(self.baseline_results, self.tft_results)
            save_evaluation_report(self.baseline_results, self.tft_results, 'evaluation_report.txt')
            print("✅ 评估和可视化完成")
            return True
        except Exception as e:
            print(f"❌ 评估失败: {str(e)}")
            return False

    def run_eval_only_pipeline(self):
        print("🚀 开始销量预测评估流程（跳过TFT训练）")
        print("="*80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        success_steps = 0
        total_steps = 4

        if self.step1_data_analysis():
            success_steps += 1
        else:
            print("❌ 管道在步骤1失败，停止执行")
            return False

        if self.step2_baseline_model():
            success_steps += 1
        else:
            print("❌ 管道在步骤2失败，停止执行")
            return False

        if self.step3_tft_evaluate_only():
            success_steps += 1
        else:
            print("❌ 管道在步骤3失败，停止执行")
            return False

        if self.step4_evaluation():
            success_steps += 1
        else:
            print("❌ 管道在步骤4失败，停止执行")
            return False

        print("\n" + "="*80)
        print("🎉 销量预测评估流程完成!")
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
        print("   3. TFT模型: 仅评估已训练好的模型")
        print("   4. 评估对比: 全面评估两个模型的性能")

        return True

def main():
    print("🚀 销量预测项目 - 仅评估主预测管道")
    pipeline = SalesPredictionEvalOnlyPipeline('model_data_top10percent.csv')
    success = pipeline.run_eval_only_pipeline()
    if success:
        print("\n🎊 项目成功完成！")
    else:
        print("\n😞 项目执行未完成，请检查错误信息")

if __name__ == "__main__":
    main()