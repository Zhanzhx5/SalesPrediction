import pandas as pd
import os
from datetime import datetime

print("开始提取销售信息表的指定字段...")
print(f"开始时间: {datetime.now()}")

# 输入输出文件
input_file = '销售信息.csv'
output_file = '销售信息_精简.csv'

# 需要提取的字段
required_columns = ['dtdate', 'sstoreno', 'sgoodsskc', 'fquantity', '在店库存']

# 检查输入文件是否存在
if not os.path.exists(input_file):
    print(f"错误: 文件 {input_file} 不存在!")
    exit(1)

# 获取文件大小
file_size = os.path.getsize(input_file) / (1024**3)  # GB
print(f"输入文件大小: {file_size:.2f} GB")

# 分块大小设置（根据内存调整，默认每次处理10万行）
chunk_size = 100000
print(f"分块大小: {chunk_size:,} 行")

try:
    # 初始化标志
    first_chunk = True
    total_rows = 0
    chunk_count = 0
    
    print("开始分块读取和写入...")
    
    # 分块读取CSV文件
    for chunk in pd.read_csv(input_file, 
                            usecols=required_columns,  # 只读取指定列
                            dtype=str,  # 统一为字符串类型，避免类型推断开销
                            encoding='utf-8-sig',
                            keep_default_na=False,
                            chunksize=chunk_size):
        
        chunk_count += 1
        chunk_rows = len(chunk)
        total_rows += chunk_rows
        
        print(f"处理第 {chunk_count} 块: {chunk_rows:,} 行 (累计: {total_rows:,} 行)")
        
        # 第一块：创建新文件并写入表头
        if first_chunk:
            chunk.to_csv(output_file, 
                        index=False, 
                        encoding='utf-8-sig',
                        mode='w')  # 写入模式
            first_chunk = False
        else:
            # 后续块：追加写入，不写表头
            chunk.to_csv(output_file, 
                        index=False, 
                        encoding='utf-8-sig',
                        mode='a',    # 追加模式
                        header=False)  # 不写表头
        
        # 每处理10块显示一次进度
        if chunk_count % 10 == 0:
            print(f"已处理 {chunk_count} 块，累计 {total_rows:,} 行")
    
    # 获取输出文件大小
    output_size = os.path.getsize(output_file) / (1024**3)  # GB
    compression_ratio = (1 - output_size / file_size) * 100
    
    print("=" * 50)
    print("提取完成!")
    print(f"结束时间: {datetime.now()}")
    print(f"总处理行数: {total_rows:,}")
    print(f"输出文件: {output_file}")
    print(f"输出文件大小: {output_size:.2f} GB")
    print(f"压缩比例: {compression_ratio:.1f}%")
    
    # 验证输出文件
    print("\n验证输出文件...")
    sample = pd.read_csv(output_file, nrows=5, encoding='utf-8-sig')
    print("前5行数据:")
    print(sample)
    print(f"输出文件列名: {list(sample.columns)}")

except Exception as e:
    print(f"处理过程中出现错误: {e}")
    # 如果出错，删除可能不完整的输出文件
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除不完整的输出文件: {output_file}") 