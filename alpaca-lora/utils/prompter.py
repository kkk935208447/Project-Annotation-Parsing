"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union

# 1. Prompter 类负责生成对话提示(prompt)和解析对话响应(response)。
# 它使用了 JSON 配置文件中定义的模板,以及用户提供的指令、输入和标签信息,生成完整的对话提示。
# 同时,它还提供了解析对话响应的功能,从响应中提取有效信息。
class Prompter(object):
    # 1.1. __slots__ 属性用于限制类实例能够添加的属性,提高内存利用率。
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        # 1.2. 保存 verbose 模式,用于控制是否打印详细信息。
        self._verbose = verbose
        # 1.3. 如果未指定 template_name,则使用默认的 "alpaca" 模板。
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        # 1.4. 根据模板名称构建文件路径,并检查文件是否存在。
        # 如果文件不存在,则抛出 ValueError 异常。
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        # 1.5. 读取 JSON 格式的模板文件,并将其保存在 self.template 属性中。
        with open(file_name) as fp:
            self.template = json.load(fp)
        # 1.6. 如果处于 verbose 模式,则打印模板的描述信息。
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
    # 2. generate_prompt 方法用于根据用户提供的指令、输入和标签,生成完整的对话提示。
    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        # 2.1. 如果提供了输入,则使用 prompt_input 模板格式化提示;
        # 否则使用 prompt_no_input 模板格式化提示。
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        # 2.2. 如果提供了标签,则将其附加到提示的末尾。
        if label:
            res = f"{res}{label}"
        # 2.3. 如果处于 verbose 模式,则打印生成的完整提示。
        if self._verbose:
            print(res)
        return res
    # 3. get_response 方法用于从对话响应中提取有效信息。
    # 它根据模板中定义的 response_split 字段,将响应分割并返回第二部分(即实际的响应内容)。
    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
