import json
import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, AutoTokenizer  # noqa: E402

"""
这段代码的主要目的是将Alpaca LoRA权重合并到Llama-7B基础模型中,并将合并后的模型权重和参数保存到磁盘。代码执行了以下关键步骤:

加载Llama Tokenizer和Llama ForCausalLM基础模型。
加载Alpaca LoRA权重。
设置LoRA权重合并标志,指定要合并的权重。
设置模型为评估模式。
获取LoRA模型的state_dict。
定义模型参数,包括隐藏层维度、层数、头数等。
初始化旋转位置嵌入(Rotary Positional Embedding)相关参数。
定义权重排列和解排列函数,用于处理Llama模型的多头注意力权重。
定义state_dict键转换函数,将LoRA模型的键名转换为与Llama模型相兼容的格式。
构建新的state_dict,包含转换后的键名和适当排列的权重。
创建保存目录。
使用torch.save()函数将新的state_dict保存为PyTorch模型文件。
将模型参数保存为JSON文件。
整个过程利用了LoRA(Low-Rank Adaptation)技术,通过添加少量可训练参数(LoRA权重)来微调大型语言模型。这种方法可以显著减少微调所需的计算资源和存储空间,同时保持较高的性能。最后,将LoRA权重合并到基础模型中,得到一个能够直接部署的全新语言模型。

与前一段代码相比,这段代码进行了更多的处理和转换,包括设置LoRA权重合并标志、定义模型参数、初始化旋转位置嵌入、处理多头注意力权重的排列和解排列、转换state_dict键名等。这些步骤是为了将LoRA权重与Llama模型的架构相匹配,并最终得到一个可以直接加载和使用的模型权重文件。
"""

# 1. 从环境变量获取基础模型路径
# os.environ.get()方法用于获取环境变量的值
# 如果环境变量不存在,会返回None
BASE_MODEL = os.environ.get("BASE_MODEL", None)
# 如果BASE_MODEL环境变量未设置,会引发AssertionError
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# 3. 加载Llama ForCausalLM(语言模型)
# LlamaForCausalLM是一个基于Transformer的自回归语言模型
# 可用于文本生成、机器翻译等任务
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
# 4. 加载Alpaca LoRA权重
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

# merge weights
# 5. 设置LoRA权重合并标志
# 遍历LoRA模型的每一层
for layer in lora_model.base_model.model.model.layers:
    # 对每一层的Self-Attention中的查询(Q)和值(V)投影权重设置merge_weights=True
    # 这表示在权重合并时,LoRA权重将被合并到这些投影权重中
    layer.self_attn.q_proj.merge_weights = True
    layer.self_attn.v_proj.merge_weights = True

# 6. 设置模型为评估模式
# 在推理(评估)阶段,不需要计算梯度和更新权重
# train(False)将模型设置为评估模式,可以提高性能并节省内存
lora_model.train(False)

# 7. 获取LoRA模型的state_dict
# state_dict是一个包含模型所有权重和缓冲区的字典
lora_model_sd = lora_model.state_dict()
# 8. 定义模型参数
# 这些参数描述了Llama模型的架构,将用于重新构建模型权重
params = {
    "dim": 4096,
    "multiple_of": 256,
    "n_heads": 32,
    "n_layers": 32,
    "norm_eps": 1e-06,
    "vocab_size": -1,
}
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
dims_per_head = dim // n_heads

# 9. 初始化旋转位置嵌入(Rotary Positional Embedding)
# 旋转位置嵌入是一种在Transformer模型中编码序列位置信息的方法
base = 10000.0  # 一个常数,用于控制位置编码的频率
inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))  # inv_freq是一个向量,存储不同位置编码的频率

# 10. 定义权重排列和解排列函数
# 这些函数用于处理Llama模型中的多头注意力权重
def permute(w):
     # 将权重张量w重新排列,以匹配Llama模型的格式
    return (
        w.view(n_heads, dim // n_heads // 2, 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )


def unpermute(w):
    # 将权重张量w从Llama模型的格式还原为标准格式
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )

# 11. 定义state_dict键转换函数
# 这个函数将LoRA模型的state_dict键名转换为与Llama模型相兼容的格式
def translate_state_dict_key(k):  # noqa: C901
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError

# 12. 构建新的state_dict
# 新的state_dict将包含经过转换的键名和适当排列的权重
new_state_dict = {}
for k, v in lora_model_sd.items():
    new_k = translate_state_dict_key(k)
    if new_k is not None:
        if "wq" in new_k or "wk" in new_k:
            # 对查询(Q)和键(K)投影权重进行解排列
            new_state_dict[new_k] = unpermute(v)
        else:
            new_state_dict[new_k] = v

# 13. 创建保存目录
os.makedirs("./ckpt", exist_ok=True)
# 14. 保存模型权重
# 使用torch.save()函数将新的state_dict保存为PyTorch模型文件
torch.save(new_state_dict, "./ckpt/consolidated.00.pth")
# 15. 保存模型参数
# 将模型参数保存为JSON文件
with open("./ckpt/params.json", "w") as f:
    json.dump(params, f)
