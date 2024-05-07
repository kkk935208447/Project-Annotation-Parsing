import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, AutoTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# 1. 尝试检测是否可以使用 Apple 硅芯片的加速
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# 2. 定义主函数
def main(
    load_8bit: bool = False,       # 是否以 8 位精度加载模型,节省内存但可能影响性能
    base_model: str = "",          # 基础模型的路径,例如 "huggyllama/llama-7b"
    lora_weights: str = "tloen/alpaca-lora-7b",       # LoRA 权重的路径,用于微调模型
    prompt_template: str = "",  # The prompt template to use, will default to alpaca. # 使用的提示模板,默认为 alpaca 模板
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.   # 用于监听所有接口
    share_gradio: bool = True,   # 是否与他人共享 Gradio 界面,True表示便于公网访问
):
    # 3. 如果 base_model 参数为空,则尝试从环境变量中获取
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    # 4. 初始化 Prompter 类,用于生成和处理提示
    prompter = Prompter(prompt_template)
    # 5. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # 6. 根据设备类型加载模型
    if device == "cuda":
        # 6.1 如果设备为 CUDA(GPU),则执行以下操作:
        # 加载基础模型,并指定相关参数
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,     # 是否以 8 位精度加载
            torch_dtype=torch.float16,   # 使用半精度(16位浮点数)加载
            device_map="auto",          # 自动将模型放置在合适的设备上
        )
        # 加载 LoRA 权重,并指定相关参数
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,      # 使用半精度加载 LoRA 权重
        )
    # 6.2 如果设备为 Apple 硅芯片(MPS),则执行以下操作:
    # 加载基础模型,并指定相关参数
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},    # 将模型加载到 MPS 设备上
            torch_dtype=torch.float16,  # 使用半精度加载
        )
        # 加载 LoRA 权重,并指定相关参数
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},      # 将 LoRA 权重加载到 MPS 设备上
            torch_dtype=torch.float16,     # 使用半精度加载 LoRA 权重
        )
    # 6.3 如果设备为 CPU,则执行以下操作:
        # 加载基础模型,并指定相关参数
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        # 加载 LoRA 权重,并指定相关参数
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    # 7. 修复 decapoda-research 配置中的一些错误
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    # 8. 如果不是 8 位精度加载,则将模型转换为半精度
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    # 9. 将模型设置为评估模式
    model.eval()

    # # 10. 如果 PyTorch 版本在 2.0 以上且不是 Windows 系统,则启用模型编译, 提升性能
    # # TODO: 删除原代码,torch.compile 与 peft（0.9.0版本）目前似乎不兼容，开启此代码会导致lora权重文件保存的是空字典，推理时加载lora权重会报错
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    """
    这部分代码定义了一个名为 evaluate 的函数,用于根据给定的指令、输入和一系列生成参数来生成响应序列。它支持两种输出模式:流式输出和非流式输出。
    流式输出模式:当 stream_output 参数为 True 时,代码会使用一种基于迭代器和停止条件的技巧,逐个 token 生成并输出响应序列。这种模式可以实时观察生成过程,但可能会消耗更多的计算资源。
    非流式输出模式:当 stream_output 参数为 False 时,代码会一次性生成整个响应序列,然后将其解码并输出。这种模式计算效率更高,但无法实时观察生成过程。
    在生成过程中,代码会根据指定的参数(如 temperature、top_p、top_k、num_beams 等)创建生成配置对象,并将其传递给模型的 generate 方法。这些参数控制了生成序列的随机性、多样性和质量。
    最后,代码会使用 Gradio 库创建一个 Web 界面,允许用户输入指令、上下文和生成参数,并查看生成的响应。该界面提供了文本框、滑动条和复选框等控件,方便用户与模型进行交互。
    """
    def evaluate(
        instruction,    # 指令,是一个字符串,表示要执行的任务或问题
        input=None,     # 输入,是一个字符串,表示与指令相关的上下文或补充信息,默认为 None
        temperature=0.1,   # 温度,是一个浮点数,控制生成的随机性,值越高,生成的结果越随机,# 较低的温度会产生更确定的、重复的输出,而较高的温度会产生更多样化的输出
        top_p=0.75,        # 核采样(Nucleus Sampling)的 Top-p 值,是一个浮点数,范围在 0 到 1 之间
        top_k=40,           # 核采样的 Top-k 值
        num_beams=4,        # 束搜索的束大小
        max_new_tokens=128,  # 最大生成 token 数,是一个整数,表示生成序列的最大长度
        stream_output=False,   # 是否流式输出,是一个布尔值,如果为 True,则会逐个 token 生成并输出,否则会一次性生成整个序列
        **kwargs,              # 其他参数
    ):
        # 12. 生成提示
        # prompter.generate_prompt 是一个函数,用于根据指令和输入生成提示字符串
        prompt = prompter.generate_prompt(instruction, input)
        # 13. 将提示转换为 Tensor
        inputs = tokenizer(prompt, return_tensors="pt")
        # to(device) 将 Tensor 移动到指定的设备(如 GPU、CPU)上,以便后续计算
        input_ids = inputs["input_ids"].to(device)
        
        # 14. 创建生成配置对象
        # GenerationConfig 是 Transformers 库中的一个类,用于配置生成序列时的各种参数
        # temperature、top_p、top_k、num_beams 是上面定义的参数
        # **kwargs 允许传入其他生成配置参数
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        
        # 15. 设置生成参数
        # generate_params 是一个字典,包含了调用 model.generate() 所需的参数
        generate_params = {
            "input_ids": input_ids,         # 输入的 token 序列 Tensor
            "generation_config": generation_config,    # 生成配置对象
            "return_dict_in_generate": True,            # 以字典形式返回生成结果
            "output_scores": True,                     # 输出每个 token 的概率分数
            "max_new_tokens": max_new_tokens,          # 最大生成 token 数
        }

        # 16. 如果需要流式输出,则执行以下操作:
        if stream_output:
            # 这部分代码用于实现流式输出,即逐个 token 生成并输出
            # 它基于 Transformers 库中的 StoppingCriteria 和一些技巧实现了迭代器
            # 参考自 https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243

            # 16.1 定义 generate_with_callback 函数,用于生成并回调
            # callback 是一个可选参数,表示在生成每个 token 时调用的回调函数
            def generate_with_callback(callback=None, **kwargs):
                # 设置停止条件列表,如果不存在则创建一个新的列表
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                # 添加一个 Stream 对象到停止条件列表中
                # Stream 对象会在生成每个 token 时调用回调函数
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                # 在无梯度环境中调用 model.generate()
                with torch.no_grad():
                    model.generate(**kwargs)

            # 16.2 定义 generate_with_streaming 函数,用于生成并流式输出
            # 它使用 Iteratorize 将 generate_with_callback 函数转换为一个迭代器
            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            # 16.3 使用 generate_with_streaming 进行流式生成
            # 通过 with 语句创建一个迭代器对象 generator
            with generate_with_streaming(**generate_params) as generator:
                # 遍历迭代器,逐个输出生成的 token
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    # 使用 tokenizer.decode() 将 token 序列解码为字符串
                    decoded_output = tokenizer.decode(output)

                    # 如果输出为结束标记,则结束生成
                    if output[-1] in [tokenizer.eos_token_id]:
                        break
                        
                    # 使用 prompter.get_response() 获取响应字符串并输出
                    yield prompter.get_response(decoded_output)
            # 如果进入流式输出分支,则提前返回,避免执行无流式输出的代码
            return  # early return for stream_output

        # Without streaming
         # 17. 如果不需要流式输出,则执行以下操作:
        with torch.no_grad():
            # 在无梯度环境中调用 model.generate()
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        # 从生成结果中获取第一个序列
        s = generation_output.sequences[0]
        # 使用 tokenizer.decode() 将 token 序列解码为字符串
        output = tokenizer.decode(s)
        # 使用 prompter.get_response() 获取响应字符串并输出
        yield prompter.get_response(output)

    # 这部分代码使用了 Gradio 库创建了一个交互式界面,用于与 Alpaca-LoRA 模型进行交互。具体来说:
    # gr.Interface 函数用于创建 Gradio 界面。
    # fn=evaluate 指定了界面的回调函数为 evaluate。
    # inputs 参数指定了界面的输入组件:
    # 第一个 Textbox 用于输入指令,显示 2 行,标签为 "Instruction",占位符为 "Tell me about alpacas."。
    # 第二个 Textbox 用于输入文本,显示 2 行,标签为 "Input",占位符为 "none"。
    # 后面五个 Slider 用于调整生成参数:
    # Temperature: 温度,控制生成的随机性,范围为 0 到 1,默认值为 0.1。
    # Top p: 核采样的 Top-p 值,范围为 0 到 1,默认值为 0.75。
    # Top k: 核采样的 Top-k 值,范围为 0 到 100,步长为 1,默认值为 40。
    # Beams: 束搜索的束大小,范围为 1 到 4,步长为 1,默认值为 4。
    # Max tokens: 最大生成 token 数,范围为 1 到 2000,步长为 1,默认值为 128。
    # 最后一个 Checkbox 用于选择是否启用流式输出。
    # outputs 参数指定了界面的输出组件,为一个显示 5 行的 Textbox,用于显示模型生成的输出,标签为 "Output"。
    # title 参数设置界面标题为 "🦙🌲 Alpaca-LoRA"。
    # description 参数设置界面描述,介绍了 Alpaca-LoRA 模型的基本信息和来源。
    # 最后,调用 queue() 和 launch(server_name="0.0.0.0", share=share_gradio) 启动 Gradio 界面,其中 server_name="0.0.0.0" 表示监听所有网络接口,share=share_gradio 指定是否与他人共享界面。
    gr.Interface(
        fn=evaluate,         # 指定回调函数为 evaluate
        inputs=[
            gr.components.Textbox(
                lines=2,           # 文本框显示 2 行
                label="Instruction",        # 文本框标签为 "Instruction"
                placeholder="Tell me about alpacas.",       # 占位符文本为 "Tell me about alpacas."
            ),
            gr.components.Textbox(
                lines=2,           # 文本框显示 2 行
                label="Input",     # 文本框标签为 "Input"
                placeholder="none"  # 占位符文本为 "none"
                ),
            gr.components.Slider(
                minimum=0,            # 滑动条最小值为 0
                maximum=1,            # 滑动条最大值为 1
                value=0.1,            # 滑动条默认值为 0.1
                label="Temperature"   # 滑动条标签为 "Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1,     # 滑动条最小值为 1
                maximum=2000,    # 滑动条最大值为 2000
                step=1,          # 滑动条步长为 1
                value=128,        # 滑动条默认值为 128
                label="Max tokens"      # 滑动条标签为 "Max tokens"
            ),
            gr.components.Checkbox(
                label="Stream output"     # 复选框标签为 "Stream output"
                ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,          # 文本框显示 5 行
                label="Output",    # 文本框标签为 "Output"
            )
        ],
        title="🦙🌲 Alpaca-LoRA",     # 界面标题为 "🦙🌲 Alpaca-LoRA"
        # 描述
        description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    # 最后,调用 queue() 和 launch(server_name="0.0.0.0", share=share_gradio) 启动 Gradio 界面,
    # 其中 server_name="0.0.0.0" 表示监听所有网络接口,share=share_gradio 指定是否与他人共享界面。
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)
    
    # Old testing code follows.

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """


if __name__ == "__main__":
    fire.Fire(main)
