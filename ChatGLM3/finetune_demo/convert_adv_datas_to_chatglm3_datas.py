import json
from typing import Union
from pathlib import Path    # 导入 Path 类,这是 Python 内置的一个用于处理文件路径的类

# 参数 path 可以是字符串或者 Path 对象,返回值是 Path 对象
def _resolve_path(path: Union[str, Path]) -> Path:
    # Path(path) 将path打包为Path对象
    # Path(path).expanduser().resolve() 将Path对象路径转化为绝对位置路径
    return Path(path).expanduser().resolve()


def _mkdir(dir_name: Union[str, Path]):
    dir_name = _resolve_path(dir_name)   # 先将输入路径转换为绝对路径
    if not dir_name.is_dir():            # 检查目录是否存在,使用 is_dir() 方法
        dir_name.mkdir(parents=True, exist_ok=False)   # parents=True 表示递归创建父目录,exist_ok=False 表示如果目录已存在则会报错

# 这个函数用于将 AdvertiseGen 数据集转换为对话格式
def convert_adgen(data_dir: Union[str, Path], save_dir: Union[str, Path]):
    # 定义一个内部函数 _convert,用于转换单个文件
    def _convert(in_file: Path, out_file: Path):
        _mkdir(out_file.parent)  # 先创建输出文件的父目录
        with open(in_file, encoding='utf-8') as fin:  # 打开输入文件,使用 utf-8 编码
            with open(out_file, 'wt', encoding='utf-8') as fout: # 打开输出文件,使用 utf-8 编码,'wt' 表示写入文本模式
                for line in fin:
                    dct = json.loads(line)   # 将每行 JSON 数据转换为 Python 字典
                    sample = {'conversations': [{'role': 'user', 'content': dct['content']},
                                                {'role': 'assistant', 'content': dct['summary']}]}
                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # 解析输入和输出目录路径
    data_dir = _resolve_path(data_dir)
    save_dir = _resolve_path(save_dir)

     # 处理训练集文件
    train_file = data_dir / 'train.json'   # Path 对象构建训练集文件路径
    if train_file.is_file():   # 检查训练集文件是否存在
        out_file = save_dir / train_file.relative_to(data_dir)  # 构建输出文件路径,保持与输入文件的相对路径
        _convert(train_file, out_file)  # 调用 _convert 函数转换训练集文件
    else:
        raise ValueError("训练集文件不存在!")  

    dev_file = data_dir / 'dev.json'   # 构建验证集文件路径
    if dev_file.is_file():             # 检查验证集文件是否存在
        # 构建输出文件路径,保持与输入文件的相对路径
        out_file = save_dir / dev_file.relative_to(data_dir)
        _convert(dev_file, out_file)  # 调用 _convert 函数转换验证集文件
    else:
        raise ValueError("验证集文件不存在!")

if __name__ == '__main__':
    # 调用 convert_adgen 函数,将 'data/AdvertiseGen' 目录下的数据转换为对话格式,并保存到 'data/AdvertiseGen_fix' 目录
    convert_adgen('~/Documents/NLP_Develop/July_project/datas/AdvertiseGen', '~/Documents/NLP_Develop/July_project/datas/AdvertiseGen_fix')