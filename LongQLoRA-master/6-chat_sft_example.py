import json
# TextIteratorStreamer 用于实现文本的流式生成和输出,使用户能够及时看到生成的文本
from transformers import AutoTokenizer, AutoConfig, TextIteratorStreamer
import torch
import math
# threading 模块提供了对多线程的低级支持
# 在这个代码中,使用线程来实现文本生成的异步执行,避免主线程被阻塞
from threading import Thread

import sys
sys.path.append("../../")
from component.utils import ModelUtils


def main():
    # context_size = 12288  # # #
    context_size = 16384
    # 使用合并后的模型进行推理
    # model_name_or_path = '/root/autodl-tmp/output_merge/merge_llama2'  # # #
    # adapter_name_or_path = None

    # 使用base model和adapter进行推理
    model_name_or_path = '/workspace/Llama-2-7b-chat-hf'
    adapter_name_or_path = '/workspace/output/llama2-7b-sft-zero/checkpoint-200'

    # 加载数据
    with open("datas/7-paper_review_data_longqlora_infer.json", 'r', encoding='utf8') as f:
        data_list = f.readlines()
    data_example = data_list[0]
    data_example = json.loads(data_example)
    ip = data_example["input"].strip()  # 读取示例数据中的输入文本
    # op = data_example["output"].strip()
    del data_list

    # template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: {input}\nASSISTANT: "
    # 定义提示模板,包含指令和输入占位符
    template = ("Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{{instruction}}\n\n### Input:\n{{input}}\n\n### Response:\n")  # # #
    instruction = """You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteria: ** importance and novelty **, ** potential reasons for acceptance **, ** potential reasons for rejection **, and ** suggestions for improvement **. The "Input" is the given paper, and the "Response" is your review that you need to provide.\n\n\n""".strip()
    template = template.replace("{{instruction}}", instruction)
    


    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    # load_in_4bit = False
    load_in_4bit = False

    # 生成超参配置
    gen_kwargs = {
        'max_new_tokens': 900, # 生成文本的最大新 token 数
        'top_p': 0.9,  # 使用 top-p 采样策略,只考虑概率最高的一部分 token
        'temperature': 0.35,  # 控制生成文本的随机性,温度越高,随机性越大
        # 'repetition_penalty': 1.0,  # 重复惩罚系数,防止生成过于重复的文本
        'repetition_penalty': 1.0,
        'do_sample': True  # 是否进行采样,即根据概率分布随机生成 token
    }

    # Set RoPE scaling factor
    config = AutoConfig.from_pretrained(model_name_or_path)
    # 从预训练模型配置中获取原始上下文长度
    # 如果设置的上下文窗口大小超过了原始长度,则需要计算 RoPE 缩放因子
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    # 加载模型
    model = ModelUtils.load_model(
        model_name_or_path,
        config=config,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )

    # 创建 TextIteratorStreamer 实例,用于实现流式文本生成
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0)

    # 设置文本生成的结束 token ID 和 streamer
    gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
    gen_kwargs["streamer"] = streamer
    """
    # 推理在这部分代码中,主线程从 streamer 中获取流式生成的文本,并将其打印到控制台。streamer 是一个迭代器,它会不断从新线程中获取生成的文本片段,因此主线程可以实时显示生成的结果,而不会被阻塞。
    使用线程的目的是实现异步执行,避免主线程被阻塞。如果我们在主线程中直接调用 model.generate,那么主线程将被阻塞,直到文本生成完成才能继续执行后面的代码。这会导致交互式体验变差,用户无法实时看到生成的结果。
    通过使用线程,我们将计算密集型的文本生成操作offload到一个新线程中执行,而主线程可以继续响应用户输入、打印输出等操作,从而提供更流畅的交互体验。
    需要注意的是,线程并不是完全独立的执行单元,多个线程之间仍然共享同一个进程的内存空间,因此在多线程编程时需要注意线程安全问题。但在这个特定的场景下,由于文本生成是一个纯计算操作,不涉及共享状态的修改,因此使用线程是安全的。
    """
    text = ip
    while True:
        text = text.strip()
        text = template.replace("{{input}}", text) # 将输入文本替换到提示模板中

        # 将文本编码为模型可以处理的输入 tensor
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        gen_kwargs["input_ids"] = input_ids  # input_ids 放进去

        with torch.no_grad():
            # 创建一个新线程执行模型的 generate 方法,实现异步生成
            # 这里,我们创建了一个新线程,它的目标函数是 model.generate 方法,并将 gen_kwargs 字典作为关键字参数传递给该方法。model.generate 是预训练语言模型进行文本生成的方法,通常是一个计算密集型操作,可能会阻塞主线程。
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            # 接下来,我们启动这个新线程:
            thread.start()
            print('Output:')

            # 这个操作会让新线程开始执行它的目标函数,即 model.generate。但是,主线程并不会等待新线程执行完毕,而是继续执行后面的代码。
            response = []
            # 从 streamer 中获取流式生成的文本
            for new_text in streamer:
                print(new_text, end='', flush=True)
                response.append(new_text)
        print()
        text = input('Input：')


if __name__ == '__main__':
    main()
