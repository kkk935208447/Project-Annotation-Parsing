import os, sys
# Gradio 是一个用于快速创建机器学习 UI 界面的开源 Python 库
# 它允许开发者快速将机器学习模型部署为 Web 应用程序,提供了丰富的 UI 组件和自定义功能
import gradio as gr
# mdtex2html 是一个用于将 Markdown 和 LaTeX 格式的文本转换为 HTML 的工具
# 它可以解析 Markdown 和 LaTeX 语法,并将其转换为对应的 HTML 标记
import mdtex2html

import torch
import transformers
# 从 Transformers 库中导入相关的类和函数
# AutoConfig 用于自动加载模型配置
# AutoModel 用于自动加载预训练模型
# AutoTokenizer 用于自动加载对应的 Tokenizer
# DataCollatorForSeq2Seq 用于序列到序列任务的数据collator
# HfArgumentParser 用于解析命令行参数
# Seq2SeqTrainingArguments 用于设置序列到序列任务的训练参数
# set_seed 用于设置随机种子,确保实验可重复性
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

# 从 arguments.py 文件中导入 ModelArguments 和 DataTrainingArguments 类
# 这些类用于定义模型和数据相关的参数
from arguments import ModelArguments, DataTrainingArguments

# 初始化全局变量
model = None
tokenizer = None

"""Override Chatbot.postprocess"""
# 1. 重写 Gradio Chatbot 组件的 postprocess 方法
# 该方法用于在渲染消息之前对其进行后处理
# 在这里,我们将 Markdown 和 LaTeX 转换为 HTML 格式
# 这种做法可以确保在 Web 界面中正确显示格式化的文本
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

# 2. 将重写的 postprocess 方法赋值给 Gradio Chatbot 组件
gr.Chatbot.postprocess = postprocess

# 3. 定义 parse_text 函数,用于解析和格式化文本
# 该函数主要用于处理代码块,将其转换为 HTML 格式
# 同时还对一些特殊字符进行转义,以确保它们在 HTML 中正确显示
# 这种做法可以确保在 Web 界面中正确显示代码块和特殊字符
def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

# 4. 定义 predict 函数,用于生成模型的响应
# 该函数将用户输入传递给模型,并通过流式输出的方式获取响应
# 同时,它还会更新 Gradio Chatbot 组件显示的对话历史
# 这种做法可以实现实时的响应生成和界面更新,提高用户体验
def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
    # 将用户输入添加到对话历史中
    chatbot.append((parse_text(input), ""))
    # 使用模型生成响应
    for response, history, past_key_values in model.stream_chat(tokenizer, input, 
                                                                history, 
                                                                past_key_values=past_key_values,
                                                                return_past_key_values=True,
                                                                max_length=max_length, 
                                                                top_p=top_p,
                                                                temperature=temperature):
        # 更新 Gradio Chatbot 组件显示的对话历史
        chatbot[-1] = (parse_text(input), parse_text(response))
        # 使用 yield 关键字,将更新后的对话历史、历史状态和过去的键值对逐步返回
        yield chatbot, history, past_key_values

# 5. 定义 reset_user_input 函数,用于清空用户输入框
# 这个函数可以在用户提交输入后自动清空输入框,为下一次输入做准备
def reset_user_input():
    return gr.update(value='')

# 6. 定义 reset_state 函数,用于重置对话状态
# 包括清空对话历史、历史状态和过去的键值对
# 这个函数可以在用户需要开始新的对话时使用,确保对话状态清空
def reset_state():
    return [], [], None

# 7. 使用 Gradio 创建 UI 界面
with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM2-6B</h1>""")

    # 创建 Chatbot 组件,用于显示对话历史
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                # 创建用户输入框,允许用户输入多行文本
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                # 创建提交按钮,用于提交用户输入
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            # 创建清空历史按钮,用于清空对话历史
            emptyBtn = gr.Button("Clear History")
            # 创建最大长度滑块,用于控制生成响应的最大长度
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            # 创建 Top-P 滑块,用于控制生成响应时的 Top-P 抽样策略
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            # 创建温度滑块,用于控制生成响应时的温度参数
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    # 创建用于存储对话历史和过去的键值对的状态变量
    history = gr.State([])
    past_key_values = gr.State(None)
    # 将提交按钮的点击事件绑定到 predict 函数
    # 当用户点击提交按钮时,predict 函数会被调用,生成模型响应并更新对话历史
    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)
    # 将提交按钮的点击事件绑定到 reset_user_input 函数,用于清空用户输入框
    submitBtn.click(reset_user_input, [], [user_input])
    # 将清空历史按钮的点击事件绑定到 reset_state 函数,用于重置对话状态
    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)


def main():
    global model, tokenizer
    # 8. 解析命令行参数
    parser = HfArgumentParser((
        ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # 如果命令行参数只有一个 JSON 文件路径,从该文件中加载参数
        model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        # 否则,从命令行参数中解析参数
        model_args = parser.parse_args_into_dataclasses()[0]

    # 9. 加载 Tokenizer 和模型配置
    # AutoTokenizer 可以自动加载与模型对应的 Tokenizer
    # trust_remote_code=True 允许从远程下载并执行代码
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    # AutoConfig 可以自动加载模型配置
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    # 10. 设置模型配置
    # 根据命令行参数设置模型的 pre_seq_len 和 prefix_projection 配置
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection
    # 11. 加载预训练模型或进行预训练模型微调
    if model_args.ptuning_checkpoint is not None:
        # 从指定路径加载预训练模型
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        # 加载 Prefix Tuning 模型权重
        print(f"Loading prefix_encoder weight from {model_args.ptuning_checkpoint}")
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        print(f"Loaded prefix_encoder weight from {model_args.ptuning_checkpoint}")
    else:
        # 直接从指定路径加载预训练模型
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    # 12. 如果指定了量化位数,对模型进行量化
    # 量化是一种将浮点数转换为定点数的技术,可以减小模型大小并提高推理速度
    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    # 13. 将模型移动到 GPU 上
    model = model.cuda()
    # 14. 如果指定了 pre_seq_len,对模型的前缀编码器进行浮点类型转换
    # 这是 P-Tuning v2 技术的一部分,用于改进模型的生成质量
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model.transformer.prefix_encoder.float()
    # 15. 将模型设置为评估模式
    # 在评估模式下,模型不会计算梯度和更新权重,可以提高推理速度
    model = model.eval()
    # 16. 启动 Gradio 界面
    # 启动 Gradio 应用程序,并在浏览器中打开界面
    # share=False 表示共享应用程序链接
    # inbrowser=True 表示在浏览器中打开界面
    demo.queue().launch(share=True, inbrowser=True)



if __name__ == "__main__":
    main()