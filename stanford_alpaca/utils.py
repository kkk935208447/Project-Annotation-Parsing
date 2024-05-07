import dataclasses
import logging
import math
import os
import io
import sys
import time
import json

"""
好的,我来详细介绍一下Sequence和Union这两个类型提示。
Sequence
Sequence是Python中的一个泛型别名,用于表示序列类型,例如列表(list)、元组(tuple)和范围(range)等。它是由typing模块定义的,可以用于类型注解,以提高代码的可读性和类型安全性。
示例:
from typing import Sequence
def concatenate(strings: Sequence[str]) -> str:
    return ''.join(strings)
concatenate(['a', 'b', 'c'])  # 返回 'abc'
concatenate(('x', 'y', 'z'))  # 返回 'xyz'

Union
Union也是由typing模块定义的,它用于表示一个值可以是多种类型中的任意一种。这在处理heterogeneous(异构)数据时非常有用。
示例:
from typing import Union
def format_value(value: Union[int, float, str]) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    else:
        return str(value)
format_value(42)     # 返回 "42.00"
format_value(3.14159)  # 返回 "3.14"
format_value("hello")  # 返回 "hello"
"""
from typing import Optional, Sequence, Union
# openai是一个Python库,提供了与OpenAI API交互的接口
import openai
import tqdm
# 从openai库中导入openai_object模块
from openai import openai_object
import copy

# 1. 定义了一个类型别名StrOrOpenAIObject
# 它是str或openai_object.OpenAIObject类型的联合类型
# 用于指示某个参数可以是字符串或OpenAI对象
StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

# 从环境变量中获取OPENAI_ORG的值,表示OpenAI组织的名称
openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    # 如果环境变量OPENAI_ORG被设置,则将其赋值给openai.organization
    openai.organization = openai_org
    # 打印一条警告日志,表示切换到指定的OpenAI组织
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")

# 使用dataclass装饰器定义了一个数据类OpenAIDecodingArguments
@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800    # 表示生成文本的最大长度,默认值为1800
    temperature: float = 0.2   # 表示采样时的温度参数,默认值为0.2
    top_p: float = 1.0        # top_p是一个浮点数字段,表示进行nucleus sampling时的概率阈值,默认值为1.0
    n: int = 1                # 表示生成多少个独立的序列,默认值为1
    stream: bool = False       # 表示是否实时流式输出生成的文本,默认值为False
    stop: Optional[Sequence[str]] = None  # 表示生成文本的停止条件,如果遇到这些字符串就停止生成
    presence_penalty: float = 0.0   # 表示已生成token的存在惩罚,用于增加生成文本的多样性
    frequency_penalty: float = 0.0   # 表示惩罚过于频繁使用的token
    suffix: Optional[str] = None       # 表示在生成的文本后追加的字符串
    logprobs: Optional[int] = None     # logprobs是一个可选的整数字段,表示输出每个token的对数概率值
    echo: bool = False       # echo是一个布尔字段,表示在生成的文本前是否先输出提示


def openai_completion(
    # prompts是输入的提示(prompt),可以是字符串、字符串列表、字典列表或字典
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,  # decoding_args是一个OpenAIDecodingArguments实例,包含解码参数
    model_name="text-davinci-003",           # model_name是模型名称,默认为"text-davinci-003"
    sleep_time=2,                           # sleep_time是一个整数,表示在达到请求速率限制时的睡眠时间(秒)
    batch_size=1,                           # batch_size是一个整数,表示每批次发送的提示数量
    max_instances=sys.maxsize,              # max_instances是一个整数,表示最大实例数(提示数),默认为sys.maxsize
    max_batches=sys.maxsize,                # max_batches是一个整数,表示最大批次数,将在未来被弃用
    return_text=False,                     # return_text是一个布尔值,表示是否返回纯文本而不是完整的completion对象
    **decoding_kwargs,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    # 判断是否为单个提示,如果prompts是字符串或字典,则为单个提示
    is_single_prompt = isinstance(prompts, (str, dict))
    # 如果是单个提示,则将其转换为列表
    if is_single_prompt:
        prompts = [prompts]

    # 如果max_batches被设置,则给出一个警告,并将max_instances设置为max_batches * batch_size
    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size
    # 截断prompts列表,只保留前max_instances个提示
    prompts = prompts[:max_instances]

    # 获取提示的总数
    num_prompts = len(prompts)
    # 将prompts分割为多个批次,每个批次包含batch_size个提示
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    # 初始化一个空列表,用于存储生成的completion
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args,# 克隆解码参数,避免修改原始参数

        while True:
            try:
                # 构造调用OpenAI API的参数字典
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                # 调用OpenAI API生成completion
                completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                # 获取生成的多个选择(choices)
                choices = completion_batch.choices
                # 为每个choice添加total_tokens字段,表示生成文本的总token数
                for choice in choices:
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                # 将生成的choices添加到completions列表中
                completions.extend(choices)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):  # 如果提示过长,则减小max_tokens的值并重试
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests. # 如果达到请求速率限制,则睡眠一段时间后重试

    if return_text:  # 如果return_text为True,则只保留completion的文本部分
        completions = [completion.text for completion in completions]
    # 如果decoding_args.n大于1,表示为每个提示生成多个completion
    # 将completions重新排列为一个嵌套列表,每个子列表包含n个连续的completion
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:  # 如果只有一个提示,则返回一个非元组的completion
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions  


# _make_w_io_base 函数的作用是:
# 1. 检查传入的 f 是否为文件对象(io.IOBase 类型)
# 2. 如果不是文件对象,先创建必要的目录(如果目录不存在)
# 3. 以指定的模式(mode)打开文件,返回文件对象
# 这个函数通常用于写入操作,因为它会先创建必要的目录
def _make_w_io_base(f, mode: str):
    # 如果传入的 f 不是 io.IOBase 类型(文件对象)
    if not isinstance(f, io.IOBase):
        # 获取文件路径的目录部分
        f_dirname = os.path.dirname(f)
         # 如果目录部分不为空
        if f_dirname != "":
            # 创建目录,exist_ok=True 表示如果目录已存在也不会引发异常
            os.makedirs(f_dirname, exist_ok=True)
        # 以指定模式(mode)打开文件,返回文件对象
        f = open(f, mode=mode)
    # 返回文件对象
    return f

# _make_r_io_base 函数的作用是:
# 1. 检查传入的 f 是否为文件对象(io.IOBase 类型)
# 2. 如果不是文件对象,直接以指定的模式(mode)打开文件
# 3. 返回文件对象
# 这个函数通常用于读取操作,不需要创建目录
def _make_r_io_base(f, mode: str):
    # 如果传入的 f 不是 io.IOBase 类型(文件对象)
    if not isinstance(f, io.IOBase):
        # 直接以指定模式(mode)打开文件,返回文件对象
        f = open(f, mode=mode)
    # 返回文件对象
    return f


# jdump 函数的作用是:
# 1. 将一个字典、列表或字符串对象写入指定文件
# 2. 如果对象是字典或列表,使用 json.dump 将其序列化为 JSON 格式写入文件
# 3. 如果对象是字符串,直接将字符串写入文件
# 4. 如果对象不是字典、列表或字符串,引发 ValueError
# 5. 支持自定义缩进级别(indent)和处理非序列化对象的函数(default)
def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    # 调用 _make_w_io_base 函数获取文件对象
    f = _make_w_io_base(f, mode)
    # 如果 obj 是字典或列表
    if isinstance(obj, (dict, list)):
        # 使用 json.dump 将对象写入文件
        # indent 控制缩进级别,default 指定处理非序列化对象的函数(默认为 str)
        json.dump(obj, f, indent=indent, default=default)
    # 如果 obj 是字符串
    elif isinstance(obj, str):
        # 直接将字符串写入文件
        f.write(obj)
    else:
        # 如果 obj 不是字典、列表或字符串,引发 ValueError
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close() # 关闭文件


# jload 函数的作用是:
# 1. 从指定文件中加载 JSON 对象
# 2. 使用 json.load 将 JSON 数据反序列化为 Python 字典对象
# 3. 返回加载的字典对象
def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    # 调用 _make_r_io_base 函数获取文件对象
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)  # 使用 json.load 从文件中加载 JSON 对象
    f.close()  # 关闭文件
    return jdict   # 返回加载的字典对象
