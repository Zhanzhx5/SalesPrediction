import pandas as pd
import os

def filter_top10_percent_data():
    """
    从model_data.csv中筛选出store_id和item_id组合在top_10_percent_sales_combinations.csv中的所有行
    """
    print("正在读取数据文件...")
    
    # 获取项目根目录路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 读取数据文件（从根目录读取）
    model_data_path = os.path.join(root_dir, 'model_data.csv')
    top10_combinations_path = os.path.join(root_dir, 'top_10_percent_sales_combinations.csv')
    
    model_data = pd.read_csv(model_data_path)
    top10_combinations = pd.read_csv(top10_combinations_path)
    
    print(f"model_data.csv 总行数: {len(model_data)}")
    print(f"top_10_percent_sales_combinations.csv 总行数: {len(top10_combinations)}")
    
    # 创建筛选条件：store_id和item_id的组合在top10列表中
    # 使用merge
    filtered_data = model_data.merge(
        top10_combinations, 
        on=['store_id', 'item_id'], 
        how='inner'
    )
    
    print(f"筛选后的数据行数: {len(filtered_data)}")
    
    # 保存结果到根目录
    output_file = os.path.join(root_dir, 'model_data_top10percent.csv')
    filtered_data.to_csv(output_file, index=False)
    print(f"结果已保存到: {output_file}")
    
    # 显示一些统计信息
    print(f"\n筛选比例: {len(filtered_data)/len(model_data)*100:.2f}%")
    
    # 显示前几行数据作为验证
    print("\n筛选后的数据前5行:")
    print(filtered_data.head())
    
    return filtered_data

if __name__ == "__main__":
    filter_top10_percent_data()
