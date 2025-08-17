import os
import chardet
import shutil

# 支持的常见中文编码
COMMON_ENCODINGS = ['utf-8', 'gb18030', 'gbk', 'gb2312', 'big5', 'utf-16', 'utf-16le', 'utf-16be']

def detect_encoding(filepath, nbytes=100000):
    with open(filepath, 'rb') as f:
        raw = f.read(nbytes)
    result = chardet.detect(raw)
    return result['encoding']

def convert_file(filepath, target_encoding='utf-8'):
    enc = detect_encoding(filepath)
    if not enc:
        print(f"无法检测编码: {filepath}")
        return
    enc = enc.lower()
    if enc == target_encoding:
        print(f"已是utf-8: {filepath}")
        return
    try:
        with open(filepath, 'r', encoding=enc, errors='ignore') as f:
            content = f.read()
        # 备份原文件
        shutil.copy(filepath, filepath + '.bak')
        with open(filepath, 'w', encoding=target_encoding) as f:
            f.write(content)
        print(f"已转换: {filepath} ({enc} -> {target_encoding})")
    except Exception as e:
        print(f"转换失败: {filepath} ({enc})，错误: {e}")

def main():
    for fname in os.listdir('.'):
        if fname.lower().endswith('.csv'):
            convert_file(fname)

if __name__ == '__main__':
    main() 