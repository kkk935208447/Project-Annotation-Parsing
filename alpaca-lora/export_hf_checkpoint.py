import os

import torch
import transformers
# PeftModel是PEFT(Parameter-Efficient Fine-Tuning)库提供的一个类
# 它支持使用LoRA(Low-Rank Adaptation)等技术进行高效微调
from peft import PeftModel
from transformers import LlamaForCausalLM, AutoTokenizer  # noqa: F402

"""
这段代码的主要目的是将基于Llama-7B预训练模型训练的Alpaca LoRA权重合并到基础模型中,并将合并后的模型保存到磁盘。代码执行了以下关键步骤:

加载Llama Tokenizer和Llama ForCausalLM基础模型。
备份基础模型的第一层权重,用于后续比较。
加载Alpaca LoRA权重,并验证LoRA权重是否被正确加载(不影响基础模型权重)。
使用PeftModel提供的merge_and_unload()方法将LoRA权重合并到基础模型中。
验证权重合并是否成功(基础模型权重发生改变)。
准备保存模型:获取state_dict,去除不需要的LoRA键值对。
使用LlamaForCausalLM.save_pretrained()方法将最终模型保存到指定路径。
整个过程利用了LoRA(Low-Rank Adaptation)技术,通过添加少量可训练参数(LoRA权重)来微调大型语言模型。这种方法可以显著减少微调所需的计算资源和存储空间,同时保持较高的性能。最后,将LoRA权重合并到基础模型中,得到一个能够直接部署的全新语言模型
"""
# 1. 从环境变量获取基础模型路径
# os.environ.get()方法用于获取环境变量的值
# 如果环境变量不存在,会返回None
BASE_MODEL = os.environ.get("BASE_MODEL", None)
# assert语句用于断言一个条件,如果条件为False,则会引发AssertionError
# 这里要求必须设置BASE_MODEL环境变量,否则无法执行后续代码
assert (BASE_MODEL), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

# 2. 加载Llama Tokenizer
# Tokenizer用于将文本转换为模型可以理解的token序列
# LlamaTokenizer.from_pretrained()方法从指定路径加载预训练好的Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# 3. 加载Llama ForCausalLM(语言模型)
# from_pretrained()方法从指定路径加载预训练好的模型权重
# load_in_8bit=False表示不使用8位精度加载模型(节省内存但可能影响性能)
# torch_dtype=torch.float16使用半精度(16位浮点数)加载,以节省内存
# device_map={"": "cpu"}将模型加载到CPU上,避免占用GPU内存
base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)
# 4. 备份基础模型的第一层权重
# base_model.model.layers[0]获取模型的第一层
# self_attn.q_proj.weight获取该层的查询向量(Query)投影权重张量
# 这些权重在后面会被LoRA权重修改,所以先克隆一个副本用于比较
first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

# 5. 加载Alpaca LoRA权重
# PeftModel用于高效地对大型语言模型进行微调
# 使用LoRA(Low-Rank Adaptation)技术,可以通过添加少量可训练参数来调整模型
# from_pretrained()方法从指定路径加载预训练好的LoRA权重
# "tloen/alpaca-lora-7b"表示针对Llama-7B模型训练好的Alpaca LoRA权重
# device_map={"": "cpu"}将LoRA权重加载到CPU上
# torch_dtype=torch.float16使用半精度加载,以节省内存
lora_model = PeftModel.from_pretrained(
    base_model,
    "tloen/alpaca-lora-7b",
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)
# 6. 检查LoRA权重是否被正确加载
# lora_model.base_model.model.model获取LoRA模型的内部基础模型
# layers[0]获取第一层
# self_attn.q_proj.weight获取该层的查询向量投影权重张量
lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight
# 使用torch.allclose()函数检查基础模型的第一层权重在加载LoRA后是否保持不变
# 如果权重发生变化,说明LoRA权重加载存在问题
assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
# 7. 合并LoRA权重到基础模型
# 通过PeftModel提供的merge_and_unload()方法
# 将LoRA权重合并到基础模型的参数张量中
# 这个过程会修改基础模型的权重
lora_model = lora_model.merge_and_unload()
# 8. 设置模型为评估模式
# 在推理(评估)阶段,不需要计算梯度和更新权重
# train(False)将模型设置为评估模式,可以提高性能并节省内存
lora_model.train(False)

# did we do anything?
# 9. 验证权重是否被成功合并
# 检查基础模型的第一层权重是否发生变化
# 如果权重没有变化,说明合并过程存在问题
assert not torch.allclose(first_weight_old, first_weight)

# 10. 准备保存模型
# 获取模型的state_dict(包含所有权重和缓冲区的字典)
lora_model_sd = lora_model.state_dict()
# 去除state_dict中不需要的键(以"lora"开头的键)
# 这些键对应LoRA权重,已经合并到主模型中,所以可以丢弃
# k.replace("base_model.model.", "")移除键名中的"base_model.model."前缀
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}
# 11. 将最终模型保存到"./hf_ckpt"路径
# LlamaForCausalLM.save_pretrained()方法用于保存模型
# base_model是原始的基础模型对象
# "./hf_ckpt"是要保存模型的路径
# state_dict=deloreanized_sd将处理后的state_dict作为模型权重保存
# max_shard_size="400MB"控制模型分片的大小,避免单个文件过大
LlamaForCausalLM.save_pretrained(
    base_model, "./hf_ckpt", state_dict=deloreanized_sd, max_shard_size="400MB"
)
