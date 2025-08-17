#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
销量预测项目 - 快速启动脚本

一键运行完整的销量预测自动化管道
"""

import os
import sys
from datetime import datetime

def check_requirements():
    """检查依赖是否安装"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'torch', 
        'pytorch_lightning', 'pytorch_forecasting'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_data_file():
    """检查数据文件是否存在"""
    data_file = 'model_data_mini_shaping.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("请确保数据文件在当前目录下")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(data_file) / (1024 * 1024)  # MB
    print(f"✅ 数据文件存在: {data_file} ({file_size:.1f} MB)")
    
    return True

def main():
    """主函数"""
    print("🚀 销量预测项目 - 快速启动")
    print("=" * 60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 检查依赖
    print("\n📦 检查项目依赖...")
    if not check_requirements():
        return
    print("✅ 所有依赖已安装")
    
    # 检查数据文件
    print("\n📊 检查数据文件...")
    if not check_data_file():
        return
    
    # 导入并运行主管道
    print("\n🔧 启动预测管道...")
    try:
        from main_prediction_pipeline import SalesPredictionPipeline
        
        # 创建管道实例
        pipeline = SalesPredictionPipeline('model_data_mini_shaping.csv')
        
        # 运行完整管道
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("\n🎊 销量预测项目成功完成！")
            print("\n📋 生成的主要文件:")
            print("   📊 data_summary.png - 数据概览图")
            print("   📊 evaluation_comparison.png - 模型评估对比图")
            print("   📄 evaluation_report.txt - 详细评估报告")
            print("\n💡 建议:")
            print("   1. 查看数据概览图了解数据分布")
            print("   2. 查看评估对比图比较模型性能")
            print("   3. 查看详细报告了解具体指标")
        else:
            print("\n😞 项目执行未完成，请检查错误信息")
            
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保所有Python文件在当前目录下")
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        print("请检查错误信息并重试")

if __name__ == "__main__":
    main() 