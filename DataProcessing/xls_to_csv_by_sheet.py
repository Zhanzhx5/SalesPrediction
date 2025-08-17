import pandas as pd
import os

# 文件名
xls_file = '不同类别商品关注的关键属性列表.xls'

# 读取所有sheet
sheets = pd.read_excel(xls_file, sheet_name=None)

for sheet_name, df in sheets.items():
    # 清理sheet名中的非法字符
    safe_name = sheet_name.replace('/', '_').replace('\\', '_').replace(' ', '').replace(':', '_')
    csv_name = f"{safe_name}属性列表.csv"
    df.to_csv(csv_name, index=False, encoding='utf-8-sig')
    print(f"已导出: {csv_name}") 