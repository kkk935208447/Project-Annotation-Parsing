from transformers import AutoModel, AutoTokenizer
import gradio as gr
# 3. 导入 mdtex2html 模块,用于将 Markdown 和 LaTeX 格式的文本转换为 HTML
import mdtex2html
from utils import load_model_on_gpus   # 4. 从 utils.py 文件中导入 load_model_on_gpus 函数,用于在多个 GPU 上加载模型

# 5. 使用 AutoTokenizer.from_pretrained() 方法从指定路径加载 Tokenizer, trust_remote_code=True 允许从远程下载并执行代码
tokenizer = AutoTokenizer.from_pretrained("/workspace/chatglm2-6b", trust_remote_code=True)
# 6. 使用 AutoModel.from_pretrained() 方法从指定路径加载预训练模型, trust_remote_code=True 允许从远程下载并执行代码, cuda() 方法将模型移动到 GPU 上
model = AutoModel.from_pretrained("/workspace/chatglm2-6b", trust_remote_code=True).cuda()

# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)

# 8. 将模型设置为评估模式,在推理时不需要计算梯度和更新权重
model = model.eval()

"""Override Chatbot.postprocess"""
# 9. 重写 Gradio Chatbot 组件的 postprocess 方法
# 该方法用于在渲染消息之前对其进行后处理
# 在这里,我们将 Markdown 和 LaTeX 转换为 HTML 格式
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess  # 10. 将重写的 postprocess 方法赋值给 Gradio Chatbot 组件

# 11. 定义 parse_text 函数,用于解析和格式化文本
# 该函数主要用于处理代码块,将其转换为 HTML 格式
# 同时还对一些特殊字符进行转义,以确保它们在 HTML 中正确显示
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


def predict(input, 
            chatbot, 
            max_length, 
            top_p, 
            temperature, 
            history, 
            past_key_values):
    # 12. 将用户输入添加到对话历史中,同时使用 parse_text 函数对输入进行格式化
    chatbot.append((parse_text(input), ""))
    # stream_chat 方法支持以流式方式生成响应,可以实时更新界面
    for response, history, past_key_values in model.stream_chat(tokenizer, input, 
                                                                history, 
                                                                past_key_values=past_key_values,  # return_past_key_values=True 表示返回过去的键值对,用于下一轮生成
                                                                return_past_key_values=True,      
                                                                max_length=max_length,             # max_length 参数控制生成响应的最大长度
                                                                top_p=top_p,   # top_p 参数控制 Top-P 抽样策略
                                                                temperature=temperature):
        # 14. 使用 parse_text 函数对生成的响应进行格式化,并更新 Gradio Chatbot 组件显示的对话历史
        chatbot[-1] = (parse_text(input), parse_text(response))
        # 15. 使用 yield 关键字,将更新后的对话历史、历史状态和过去的键值对逐步返回
        yield chatbot, history, past_key_values

# 16. 定义 reset_user_input 函数,用于清空用户输入框
def reset_user_input():
    return gr.update(value='')

# 17. 定义 reset_state 函数,用于重置对话状态,包括清空对话历史、历史状态和过去的键值对
def reset_state():
    return [], [], None


with gr.Blocks() as demo:
    # 18. 使用 Gradio 创建 UI 界面,并在页面中心显示标题
    gr.HTML("""<h1 align="center">ChatGLM2-6B</h1>""")

    chatbot = gr.Chatbot()  # 19. 创建 Chatbot 组件,用于显示对话历史
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                # 20. 创建用户输入框,允许用户输入多行文本
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                # 21. 创建提交按钮,用于提交用户输入
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            # 22. 创建清空历史按钮,用于清空对话历史
            emptyBtn = gr.Button("Clear History")
             # 23. 创建最大长度滑块,用于控制生成响应的最大长度
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            # 24. 创建 Top-P 滑块,用于控制生成响应时的 Top-P 抽样策略
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            # 25. 创建温度滑块,用于控制生成响应时的温度参数
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])   # 26. 创建用于存储对话历史的状态变量
    past_key_values = gr.State(None)    # 27. 创建用于存储过去的键值对的状态变量

    # 28. 将提交按钮的点击事件绑定到 predict 函数
    # 当用户点击提交按钮时,predict 函数会被调用,生成模型响应并更新对话历史
    # show_progress=True 表示显示进度条
    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)
    # 29. 将提交按钮的点击事件绑定到 reset_user_input 函数,用于清空用户输入框
    submitBtn.click(reset_user_input, [], [user_input])
    # 30. 将清空历史按钮的点击事件绑定到 reset_state 函数,用于重置对话状态
    # show_progress=True 表示显示进度条
    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

# 31. 启动 Gradio 界面,并在浏览器中打开
# share=True 表示共享应用程序链接
# inbrowser=True 表示在浏览器中打开界面
demo.queue().launch(share=True, inbrowser=True)
