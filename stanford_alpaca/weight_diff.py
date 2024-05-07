#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Optional

import fire
import torch
import tqdm
import transformers
from train import smart_tokenizer_and_embedding_resize


# 这段代码的主要功能是计算一个微调后的语言模型与原始预训练模型之间的权重差异,并将这个差异保存到指定路径。代码执行了以下关键步骤:
# 使用@torch.inference_mode()装饰器,在函数内部启用PyTorch的推理模式,提高性能。
# 从指定路径加载微调后的模型(model_tuned)和原始预训练模型(model_raw)。
# 加载对应的Tokenizer(tokenizer_tuned和tokenizer_raw)。
# 检查原始Tokenizer是否缺少Pad Token,如果缺少则使用smart_tokenizer_and_embedding_resize()函数添加并调整Embedding层大小。
# 获取微调后模型和原始模型的state_dict,分别表示为state_dict_tuned和state_dict_raw。
# 对于每个权重矩阵或向量,计算微调后权重与原始权重之间的差值,并将差值存储在state_dict_tuned中,即state_dict_tuned[key] = state_dict_tuned[key] - state_dict_raw[key]。
# 使用tqdm.tqdm显示进度条,方便监控长时间运行的任务。
# 最后,将计算得到的权重差异(state_dict_tuned)和对应的Tokenizer保存到指定路径(path_diff)。
# 这个函数的作用是为了提供完全透明的方式展示如何计算语言模型微调前后的权重差异。通过计算和保存这个差异,可以实现对预训练模型的高效微调,避免从头开始训练,从而节省大量计算资源和时间。同时,保存差异也方便了模型的部署和分发。
# 需要注意的是,这个函数使用了PyTorch的推理模式,因此不应在训练过程中使用。它主要用于计算已经微调完成的模型与原始模型之间的权重差异。

# 1. 使用@torch.inference_mode()装饰器
# 该装饰器用于在函数内部启用PyTorch的推理模式(evaluation mode)
# 在推理模式下,PyTorch会禁用一些不必要的计算(如梯度计算),从而提高性能
@torch.inference_mode()
def make_diff(
    path_raw: str, path_tuned: str, path_diff: str, device="cpu",  # "cuda" or "cpu"
):
    """Make the weight diff.

    This function is given to present full transparency of how the weight diff was created.

    Run:
        python weight_diff.py make_diff --path_raw <your_path_raw> --path_tuned <your_path_tuned> --path_diff <your_path_diff>
    """
    # 2. 加载微调后的模型
    # transformers.AutoModelForCausalLM.from_pretrained()自动选择合适的模型类并从指定路径加载
    # device_map={"": torch.device(device)}将模型加载到指定的设备(CPU或GPU)上
    # torch_dtype=torch.float32使用32位浮点数精度加载
    # low_cpu_mem_usage=True启用低CPU内存使用模式,适用于大型模型
    model_tuned: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_tuned,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # 3. 加载原始(未微调)的模型
    # 加载方式与微调后模型相同
    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # 4. 加载微调后模型的Tokenizer
    # transformers.AutoTokenizer.from_pretrained()自动选择合适的Tokenizer类并从指定路径加载
    tokenizer_tuned: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_tuned
    )
    # 5. 加载原始模型的Tokenizer
    # 加载方式与微调后Tokenizer相同
    tokenizer_raw: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_raw
    )
    # 6. 检查原始Tokenizer是否缺少Pad Token
    # 如果缺少,则使用smart_tokenizer_and_embedding_resize()函数添加Pad Token并调整Embedding层大小
    if tokenizer_raw.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=model_raw,
            tokenizer=tokenizer_raw,
        )

    # 7. 获取微调后模型的state_dict(包含所有权重和缓冲区的字典)
    state_dict_tuned = model_tuned.state_dict()
    # 8. 获取原始模型的state_dict
    state_dict_raw = model_raw.state_dict()
    # 9. 计算微调后模型权重和原始模型权重之间的差异
    # 对于每个键(权重矩阵或向量),计算微调后权重减去原始权重的差值
    # tqdm.tqdm用于显示进度条,方便监控长时间运行的任务
    for key in tqdm.tqdm(state_dict_tuned):
        state_dict_tuned[key].add_(-state_dict_raw[key])
    # 10. 将计算得到的权重差异保存到指定路径
    # model_tuned.save_pretrained()将模型权重保存到指定路径
    # tokenizer_tuned.save_pretrained()将Tokenizer也保存到同一路径
    model_tuned.save_pretrained(path_diff)
    tokenizer_tuned.save_pretrained(path_diff)



# 这段代码的主要功能是从给定的原始模型权重和权重差异文件中恢复出原始模型的权重,并可选择保存恢复后的模型。它还提供了一个简单的推理测试功能,用于验证恢复后的模型是否正常工作。具体步骤如下:

# 从 path_raw 路径加载原始预训练模型和 Tokenizer。
# 从 path_diff 路径加载权重差异模型和 Tokenizer。
# 如果原始 Tokenizer 没有 pad_token,则添加 "[PAD]" 作为 pad_token。
# 将原始模型的权重添加到权重差异模型中,得到恢复后的模型权重。
# 执行一个简单的完整性检查,计算所有权重张量元素之和,检查是否与预期值接近。
# 如果指定了 path_tuned 路径,则将合并后的模型和 Tokenizer 保存到该路径。
# 如果 test_inference 为 True,则执行推理测试,生成对给定输入文本的响应,并打印输入和输出。
# 返回合并后的模型和 Tokenizer。
@torch.inference_mode()
def recover(
    path_raw,
    path_diff,
    path_tuned: Optional[str] = None,
    device="cpu",
    test_inference=True,
    check_integrity_naively=True,
):
    """Recover the original weights from the released weight diff.

    This function is given for you to run.

    Things to do before running this:
        1. Convert Meta's released weights into huggingface format. Follow this guide:
            https://huggingface.co/docs/transformers/main/model_doc/llama
        2. Make sure you cloned the released weight diff into your local machine. The weight diff is located at:
            https://huggingface.co/tatsu-lab/alpaca-7b/tree/main
        3. Run this function with the correct paths. E.g.,
            python weight_diff.py recover --path_raw <path_to_step_1_dir> --path_diff <path_to_step_2_dir>

    Additional notes:
        - If things run too slowly, and you have an 80G GPU lying around, let GPU go brrr by setting `--device "cuda"`.
        - If you want to save the recovered weights, set `--path_tuned <your_path_tuned>`.
            Next time you can load the recovered weights directly from `<your_path_tuned>`.
    """
    # 1. 从 path_raw 路径加载原始模型
    # transformers.AutoModelForCausalLM.from_pretrained() 方法自动检测模型类型并加载预训练权重
    # device_map={"": torch.device(device)} 指定将模型加载到指定设备(CPU 或 GPU)
    # torch_dtype=torch.float32 指定使用 32 位浮点数加载模型
    # low_cpu_mem_usage=True 启用低 CPU 内存使用模式,以减少内存占用
    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # 2. 从 path_diff 路径加载权重差异模型
    # 加载方式与原始模型相同
    model_recovered: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_diff,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # 3. 从 path_raw 路径加载原始 Tokenizer
    # transformers.AutoTokenizer.from_pretrained() 方法自动检测 Tokenizer 类型并加载
    tokenizer_raw: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_raw
    )
    # 4. 如果原始 Tokenizer 没有 pad_token,则添加 "[PAD]" 作为 pad_token
    # smart_tokenizer_and_embedding_resize() 函数用于调整 Tokenizer 和模型嵌入层的大小
    if tokenizer_raw.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=model_raw,
            tokenizer=tokenizer_raw,
        )
    # 5. 从 path_diff 路径加载权重差异 Tokenizer
    tokenizer_recovered: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_diff
    )
    # 6. 将原始模型的权重添加到权重差异模型中
    # 通过遍历权重差异模型的 state_dict 中的每个键值对
    # 将原始模型对应键的权重值加到差异模型的权重上
    state_dict_recovered = model_recovered.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_recovered):
        state_dict_recovered[key].add_(state_dict_raw[key])
    # 7. 执行简单的完整性检查
    # 计算所有权重张量元素之和,检查是否与预期值接近
    # 这种检查并不是加密强度的完整性检查,只是一个简单的近似检查
    if check_integrity_naively:
        # This is not a rigorous, cryptographically strong integrity check :)
        allsum = sum(state_dict_recovered[key].sum() for key in state_dict_recovered)
        assert torch.allclose(
            allsum, torch.full_like(allsum, fill_value=50637.1836), atol=1e-2, rtol=0
        ), "Naive integrity check failed. This could imply that some of the checkpoint files are corrupted."
    # 8. 如果指定了 path_tuned 路径,则将合并后的模型和 Tokenizer 保存到该路径
    if path_tuned is not None:
        model_recovered.save_pretrained(path_tuned)
        tokenizer_recovered.save_pretrained(path_tuned)
    # 9. 如果 test_inference 为 True,则执行推理测试
    # 生成对给定输入文本的响应,并打印输入和输出
    if test_inference:
        input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\r\n\r\n"
            "### Instruction:\r\nList three technologies that make life easier.\r\n\r\n### Response:"
        )
        inputs = tokenizer_recovered(input_text, return_tensors="pt")
        out = model_recovered.generate(inputs=inputs.input_ids, max_new_tokens=100)
        output_text = tokenizer_recovered.batch_decode(out, skip_special_tokens=True)[0]
        output_text = output_text[len(input_text) :]
        print(f"Input: {input_text}\nCompletion: {output_text}")
    # 10. 返回合并后的模型和 Tokenizer
    return model_recovered, tokenizer_recovered


def main(task, **kwargs):
    # 1. 这是一个函数,接受两个参数:
    #    - task: 表示要执行的任务名称,是一个字符串
    #    - **kwargs: 表示任意数量的关键字参数,将被传递给要执行的任务函数

    # 2. globals() 函数返回一个表示当前全局符号表的字典
    #    在这里,globals()[task] 获取以 task 为键的全局变量(应该是一个函数)

    # 3. (**kwargs) 是一种解包操作,将 kwargs 字典中的键值对作为单独的关键字参数传递给函数
    #    因此,globals()[task](**kwargs) 的作用是:
    #    - 找到以 task 为名称的全局函数
    #    - 将 kwargs 字典中的键值对解包为单独的关键字参数
    #    - 调用该函数,并传入解包后的关键字参数

    # 4. 这种设计允许在运行时动态决定要执行的函数,并传入任意数量的参数
    #    它提供了一种灵活的方式来调用不同的函数,而无需预先知道函数名称和参数列表
    #    常用于命令行工具或框架中,用户可以指定要执行的任务和相应的参数
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
