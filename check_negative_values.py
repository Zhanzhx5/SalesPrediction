import pandas as pd
import numpy as np

def check_data_quality(csv_file):
    """
    检查CSV文件中每一行每一列的数据质量问题：
    1. 负数（只检查数值型列）
    2. NaN值
    3. 无穷大值
    4. 其他缺失值
    """
    print("正在读取CSV文件...")
    df = pd.read_csv(csv_file)
    
    print(f"文件形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n" + "="*50)
    
    # 检查每一列的数据类型
    print("数据类型检查:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
    
    print("\n" + "="*50)
    
    # 只检查数值型列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    print(f"数值型列: {list(numeric_columns)}")
    
    issues_found = False
    
    # 1. 检查负数
    print("\n" + "="*50)
    print("1. 负数检查:")
    for col in numeric_columns:
        negative_mask = df[col] < 0
        negative_count = negative_mask.sum()
        
        if negative_count > 0:
            issues_found = True
            print(f"列 '{col}': 发现 {negative_count} 个负数")
        else:
            print(f"列 '{col}': 无负数")
    
    # 2. 检查NaN值
    print("\n" + "="*50)
    print("2. NaN值检查:")
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            issues_found = True
            print(f"列 '{col}': 发现 {nan_count} 个NaN值")
        else:
            print(f"列 '{col}': 无NaN值")
    
    # 3. 检查无穷大值（只检查数值型列）
    print("\n" + "="*50)
    print("3. 无穷大值检查:")
    for col in numeric_columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            issues_found = True
            print(f"列 '{col}': 发现 {inf_count} 个无穷大值")
        else:
            print(f"列 '{col}': 无无穷大值")
    
    # 4. 检查其他缺失值（空字符串、None等）
    print("\n" + "="*50)
    print("4. 其他缺失值检查:")
    for col in df.columns:
        # 检查空字符串
        empty_string_count = (df[col] == '').sum()
        # 检查None值（除了已经是NaN的）
        none_count = (df[col] == None).sum()
        # 检查字符串'nan'（小写）
        string_nan_count = (df[col].astype(str).str.lower() == 'nan').sum()
        # 检查字符串'null'
        string_null_count = (df[col].astype(str).str.lower() == 'null').sum()
        
        total_other_missing = empty_string_count + none_count + string_nan_count + string_null_count
        
        if total_other_missing > 0:
            issues_found = True
            print(f"列 '{col}': 发现 {total_other_missing} 个其他缺失值")
            if empty_string_count > 0:
                print(f"  - 空字符串: {empty_string_count} 个")
            if none_count > 0:
                print(f"  - None值: {none_count} 个")
            if string_nan_count > 0:
                print(f"  - 字符串'nan': {string_nan_count} 个")
            if string_null_count > 0:
                print(f"  - 字符串'null': {string_null_count} 个")
        else:
            print(f"列 '{col}': 无其他缺失值")
    
    # 5. 详细统计信息
    print("\n" + "="*50)
    print("5. 详细统计信息:")
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print(f"数值型列数: {len(numeric_columns)}")
    
    # 计算各种问题的总数
    total_negative = sum((df[col] < 0).sum() for col in numeric_columns)
    total_nan = df.isna().sum().sum()
    total_inf = sum(np.isinf(df[col]).sum() for col in numeric_columns)
    
    print(f"负数总数: {total_negative}")
    print(f"NaN值总数: {total_nan}")
    print(f"无穷大值总数: {total_inf}")
    
    print("\n" + "="*50)
    if issues_found:
        print("总结: 文件中发现了数据质量问题")
    else:
        print("总结: 文件中未发现数据质量问题")
    
    return issues_found

if __name__ == "__main__":
    csv_file = "model_data_mini_shaping.csv"
    check_data_quality(csv_file) 