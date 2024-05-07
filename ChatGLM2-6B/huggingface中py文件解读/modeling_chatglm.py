""" PyTorch ChatGLM model. """

import math
import copy
import warnings
import re
import sys

import torch
 # PyTorch的检查点(checkpoint)工具,用于节省GPU内存
import torch.utils.checkpoint    
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn import CrossEntropyLoss, LayerNorm, MSELoss, BCEWithLogitsLoss
# 用于跳过某些层的初始化
from torch.nn.utils import skip_init          
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
# model.generate 生成相关的参数
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

from .configuration_chatglm import ChatGLMConfig

# flags required to enable jit fusion kernels
# 6. 设置PyTorch JIT融合内核标志(仅在非macOS系统上生效)
# 这些标志用于启用JIT(Just-In-Time)融合内核,以提高模型推理的性能
# 通过编译和融合模型的计算图,JIT可以显著提升推理速度
if sys.platform != 'darwin':    # 判断当前系统是否为macOS
    # 禁用JIT的profiling模式和执行器,使其只用于推理加速
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    # 覆盖JIT的默认设置,强制允许在CPU和GPU上进行融合
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)

# 3. 获取日志记录器
logger = logging.get_logger(__name__)
# 4. 设置用于文档的预训练模型检查点和配置
_CHECKPOINT_FOR_DOC = "THUDM/ChatGLM2-6B"
_CONFIG_FOR_DOC = "ChatGLM6BConfig"
# 5. 列出可用的ChatGLM-6B预训练模型
CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm2-6b",
    # See all ChatGLM models at https://huggingface.co/models?filter=chatglm
]

# 10. 定义默认初始化函数
# 这个函数将在初始化模型时使用,如果没有指定其他初始化函数
def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)

# 11. 定义InvalidScoreLogitsProcessor类,继承自LogitsProcessor
# 这个类用于处理生成过程中的无效分数(NaN或Inf)
class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 13. 检查分数张量中是否包含NaN或Inf
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            # 14. 如果包含,则将全部分数置零
            scores.zero_()
            # 15. 并将分数[..., 5]设置为5e4
            # 这可能是为了在生成过程中避免无效分数导致的异常或不合理的输出
            scores[..., 5] = 5e4
        return scores


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    # 1. PrefixEncoder 是一个继承自 torch.nn.Module 的 PyTorch 模型类,用于编码前缀(prefix)。
    # 2. 输入形状为 (batch-size, prefix-length),输出形状为 (batch-size, prefix-length, 2*layers*hidden)。
    """

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.prefix_projection = config.prefix_projection  # 4. 从配置中获取 prefix_projection 参数,指示是否使用投影层。
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            # 5. 如果启用前缀投影,则使用一个两层的 MLP (多层感知器) 来编码前缀。
            # 6. 计算 kv_size,即 key-value 向量的大小,其取决于层数、通道数和查询组数。
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            # 7. 创建一个 Embedding 层,将前缀token映射到 kv_size 维度的向量。
            self.embedding = torch.nn.Embedding(config.pre_seq_len, kv_size)
            # 8. 创建一个两层的 MLP,第一层将 kv_size 维度映射到 hidden_size,
            #    中间使用 Tanh 激活函数,第二层再将 hidden_size 维度映射回 kv_size。
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(kv_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, kv_size)
            )
        else:
            # 9. 如果不启用前缀投影,则直接创建一个 Embedding 层,
            #    将前缀token映射到 (config.num_layers * config.kv_channels * config.multi_query_group_num * 2) 维度。
            self.embedding = torch.nn.Embedding(config.pre_seq_len,
                                                config.num_layers * config.kv_channels * config.multi_query_group_num * 2)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            # 11. 如果启用前缀投影,则先通过 Embedding 层获取前缀token的向量表示,
            #     然后将这些向量输入到 MLP 中,得到编码后的 past_key_values。
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            # 12. 如果不启用前缀投影,则直接通过 Embedding 层获取编码后的 past_key_values。
            past_key_values = self.embedding(prefix)
        return past_key_values


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    
    # 14. split_tensor_along_last_dim 函数用于沿着最后一维切分张量。
    # 15. 参数:
    #     1) tensor: 输入张量。
    #     2) num_partitions: 切分的分区数量。
    #     3) contiguous_split_chunks: 若为 True,则每个切分块在内存中是连续的。
    # 16. 返回值:
    #     一个包含切分后张量的列表。
    """
    # Get the size and dimension. # 17. 获取最后一维的大小和张量的维数。
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split. 沿着最后一维切分张量,每个切分块的大小为 last_dim_size。
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default. torch.split 默认不会创建内存连续的张量块,
    #     如果 contiguous_split_chunks 为 True,则使每个张量块在内存中连续。
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    
    return tensor_list

# 1. RotaryEmbedding 是一个继承自 nn.Module 的 PyTorch 模型类,用于实现旋转位置编码(Rotary Position Embedding)。
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        #    1) dim: 旋转位置编码的维度。
        #    2) original_impl: 是否使用原始实现方式,默认为 False。
        #    3) device: 指定使用的设备(CPU 或 GPU),默认为 None。
        #    4) dtype: 指定数据类型,默认为 None。
        super().__init__()
        # 3. 计算逆频率 inv_freq,用于生成旋转位置编码。
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        # 4. 使用 register_buffer 方法将 inv_freq 注册为缓冲区(buffer),以便在模型中持久化。
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim     # 5. 保存旋转位置编码的维度。
        self.original_impl = original_impl    # 6. 保存是否使用原始实现方式的标志。

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.

        # 7. forward_impl 方法用于实现旋转位置编码的前向传播。
        # 8. 该方法是基于论文 "Enhanced Transformer with Rotary Position Embedding" 中的实现。
        # 9. 参数:
        #    1) seq_len: 序列长度。
        #    2) n_elem: 旋转位置编码的维度。
        #    3) dtype: 数据类型。
        #    4) device: 设备(CPU 或 GPU)。
        #    5) base: 用于计算旋转位置编码的基数,默认为 10000。
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$ , 计算 theta,用于生成旋转位置编码。
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]` , 创建位置索引序列 [0, 1, ..., seq_len - 1]。
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

        # Calculate the product of position index and $\theta_i$ , 计算位置索引和 theta 的外积,得到 idx_theta。
        idx_theta = torch.outer(seq_idx, theta).float()
        # 13. 计算 cos(idx_theta) 和 sin(idx_theta),并将它们组合成一个张量 cache。
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        # 14. 根据数据类型进行类型转换,以模拟 complex32 的行为,否则可能会得到不同的结果。
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        # 15. forward 方法是模型的前向传播入口,接受以下参数:
        #     1) max_seq_len: 最大序列长度。  默认 32768
        #     2) offset: 偏移量,默认为 0。
        # 16. 调用 forward_impl 方法来实现旋转位置编码的前向传播。
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )

# 17. 使用 @torch.jit.script 装饰器,将 apply_rotary_pos_emb 函数标记为可脚本化的函数,
#     以便在模型推理时通过 PyTorch JIT 编译器进行优化和加速
@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # 18. apply_rotary_pos_emb 函数用于将旋转位置编码应用到输入张量 x。
    #     1) x: 输入张量,形状为 [seq_len, batch_size, num_heads, hidden_size]。
    #     2) rope_cache: 预计算的旋转位置编码缓存。


    # x: [sq, b, np, hn], 获取输入张量 x 的形状信息。
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2    # 20. 计算旋转位置编码的维度。
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]         # 21. 将输入张量 x 分割成两部分。
    # truncate to support variable sizes, 截断 rope_cache,使其长度与输入序列长度相同。
    rope_cache = rope_cache[:sq]
    # 23. 对输入张量 x 和 rope_cache 进行形状转换,以方便后续计算。
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    # 24. 应用旋转位置编码到输入张量 x,得到 x_out2。
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    # 25. 将应用了旋转位置编码的张量 x_out2 与未编码的部分 x_pass 拼接在一起,得到最终的输出张量。
    return torch.cat((x_out2, x_pass), dim=-1)

# 26. RMSNorm 是一个继承自 torch.nn.Module 的 PyTorch 模型类,用于实现 RMS Norm 层。
class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        #     1) normalized_shape: 需要归一化的形状。
        #     2) eps: 一个小的常数,用于避免除以零。
        #     3) device: 指定使用的设备(CPU 或 GPU),默认为 None。
        #     4) dtype: 指定数据类型,默认为 None。
        #     5) **kwargs: 其他可选参数。
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype         # 31. 保存输入张量的数据类型。
        # 32. 计算输入张量的方差,并将输入张量归一化。
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        # 33. 将归一化后的张量与权重张量相乘,并将结果转换回原始数据类型。
        return (self.weight * hidden_states).to(input_dtype)

# 1. CoreAttention 是一个继承自 torch.nn.Module 的 PyTorch 模型类,用于实现 Transformer 的注意力机制。
class CoreAttention(torch.nn.Module):
    def __init__(self, config: ChatGLMConfig, layer_number):
        super(CoreAttention, self).__init__()
        # 3. 从配置对象中获取是否应用查询-键层缩放(apply_query_key_layer_scaling)和注意力软最大值计算精度(attention_softmax_in_fp32)的设置。
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        # 4. 如果应用查询-键层缩放,则强制使用 fp32 精度计算注意力软最大值。
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        # 5. 确保层编号不小于 1。
        self.layer_number = max(1, layer_number)
        # 6. 计算投影大小(projection_size),即每个注意力头的隐藏维度乘以注意力头数量。
        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        # 7. 计算每个分区(partition)的隐藏大小(hidden_size_per_partition),等于投影大小。
        self.hidden_size_per_partition = projection_size
        # 8. 计算每个注意力头的隐藏大小(hidden_size_per_attention_head),等于投影大小除以注意力头数量。
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        # 9. 获取每个分区的注意力头数量(num_attention_heads_per_partition),等于配置中的注意力头数量。
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None  # 10. 初始化缩放因子 coeff。
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head) # 11. 计算归一化因子(norm_factor),等于每个注意力头的隐藏大小的平方根。
        # 12. 如果应用查询-键层缩放,则将归一化因子乘以当前层编号。
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff
        # 13. 创建一个dropout层,用于注意力权重的dropout操作。
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, 
                query_layer,  # 查询向量。  [sq, b, np, hn], sq 序列长度, b 批大小, np 注意力头数, hn 查询向量的维度。
                key_layer,    # 键向量。    [sq, b, np, hn]
                value_layer,  # 值向量。    [sq, b, np, hn]
                attention_mask  # 注意力掩码张量。 
            ):
        pytorch_major_version = int(torch.__version__.split('.')[0])  # 15. 判断 PyTorch 的主版本号。

        # TODO 疑问: torch2.0 版本实现attention的问题
        # TODO 1. self.norm_factor 没有传入, 没有加入层的数目的归一化, 后续可否选择加入? 
        # TODO 2. 问: 为什么在不传入att mask 时, 使用下三角矩阵, 没有考虑序列中pad的mask, 答: 输出时可以根据每个序列最长的真实序列去除pad部分的loss(但在计算时用下三角矩阵，pad部分会被计算也没关系)
        #         但是这种方式不太适用前pad的情况, 仅适用于后pad
        if pytorch_major_version >= 2:  # 16. 如果 PyTorch 版本大于或等于 2.0,则应用 permute 操作对输入张量进行转置。
            # [sq, b, np, hn] -> [b, np, sq, hn]
            query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]

            # 17. 如果没有注意力掩码,且查询和键的序列长度相同,则使用 scaled_dot_product_attention 函数计算注意力权重和上下文向量。
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                # is_causal=True, chatglm还是encoder架构，但是到了chatglm2 变成了decoder only的架构（这点很少有资料会提及到)
                context_layer = torch.nn.functional.scaled_dot_product_attention(
                                                    query_layer, key_layer, value_layer,
                                                        is_causal=True         # 启用因果注意力
                                                        )
            else:   # 18. 否则,如果有注意力掩码,则对其进行逻辑非操作。
                if attention_mask is not None:
                    attention_mask = ~attention_mask
                # 19. 使用 scaled_dot_product_attention 函数计算注意力权重和上下文向量,并传入注意力掩码。
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 attention_mask)
            # 20. 对上下文向量进行转置,并将其形状调整为 (sequence_length, batch_size, num_partitions, hidden_size_per_partition)。
            context_layer = context_layer.permute(2, 0, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            # [sq, b, np, hn] --> [sq, b, hp]
            context_layer = context_layer.reshape(*new_context_layer_shape)
        else:
            # 21. 如果 PyTorch 版本低于 2.0,则使用旧的注意力计算方式。
            # 22. 计算输出大小。
            # Raw attention scores

            # [b, np, sq, sk]
            output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

            # preallocting input tensor: [b * np, sq, sk], # 24. 预分配一个输入张量,用于后续计算。
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
                device=query_layer.device
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            if self.attention_softmax_in_fp32:
                attention_scores = attention_scores.float()
            if self.coeff is not None:
                attention_scores = attention_scores * self.coeff

            # 初始化掩码
            if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
                attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                            device=attention_scores.device, dtype=torch.bool)
                attention_mask.tril_()
                attention_mask = ~attention_mask

            if attention_mask is not None:
                # 掩码
                attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))

            # 28. 计算注意力概率。
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.type_as(value_layer)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.  # 对注意力概率应用dropout操作。
            attention_probs = self.attention_dropout(attention_probs)
            # =========================
            # Context layer. [sq, b, hp],  计算上下文向量。
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))
            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.

    # 1. SelfAttention 是一个继承自 torch.nn.Module 的 PyTorch 模型类,用于实现 Transformer 的自注意力(self-attention)层。
    # 2. 自注意力层接受形状为 [sequence_length, batch_size, hidden_size] 的输入,并输出相同形状的结果。
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)    # # 4. 确保层编号不小于 1。

        self.projection_size = config.kv_channels * config.num_attention_heads  # 5. 计算投影大小(projection_size),即每个注意力头的隐藏维度乘以注意力头数量。

        # Per attention head and per partition values.
        # 6. 计算每个注意力头的隐藏大小(hidden_size_per_attention_head),等于投影大小除以注意力头数量。
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        # 7. 获取每个分区的注意力头数量(num_attention_heads_per_partition),等于配置中的注意力头数量。
        self.num_attention_heads_per_partition = config.num_attention_heads
        # 8. 从配置对象中获取是否启用多查询注意力(multi_query_attention)的设置。
        self.multi_query_attention = config.multi_query_attention
        # 9. 计算查询-键-值(QKV)的隐藏大小(qkv_hidden_size)。
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            # 10. 如果启用多查询注意力,则获取每个分区的多查询组数量(num_multi_query_groups_per_partition),
            #     并重新计算 QKV 的隐藏大小。
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        # 11. 创建一个线性层,用于从隐藏状态计算 QKV。
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias,
                                         device=device, **_config_to_kwargs(config)
                                         )
        # 12. 创建一个 CoreAttention 实例,用于计算注意力。
        self.core_attention = CoreAttention(config, self.layer_number)

        # Output. 创建一个 CoreAttention 实例,用于计算注意力。
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                               device=device, **_config_to_kwargs(config)
                               )

    # 14. _allocate_memory 方法用于为推理过程中的键-值缓存分配内存。
    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            # 15. 如果启用多查询注意力,则使用多查询组数量作为注意力头数量。
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            # 16. 否则,使用每个分区的注意力头数量。
            num_attention_heads = self.num_attention_heads_per_partition
        # 17. 返回一个预分配的张量,用于存储键-值缓存。
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # 18. forward 方法是模型的前向传播入口,接受以下参数:
        #     1) hidden_states: 输入的隐藏状态张量,形状为 [sequence_length, batch_size, hidden_size]。
        #     2) attention_mask: 注意力掩码张量。
        #     3) rotary_pos_emb: 旋转位置编码张量。
        #     4) kv_cache: 用于推理的键-值缓存,默认为 None。
        #     5) use_cache: 是否使用键-值缓存,默认为 True。

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        # 19. 计算 QKV。
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            # 20. 如果启用多查询注意力,则从 mixed_x_layer 中分割出查询、键和值向量。
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            # 21. 对查询、键和值向量进行形状转换,以便后续计算。
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            # 22. 如果不启用多查询注意力,则将 mixed_x_layer 重新调整形状,以便分割出查询、键和值向量。
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn], 使用 split_tensor_along_last_dim 函数沿着最后一维将混合张量分割成查询、键和值向量。
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding), 如果提供了旋转位置编码,则将其应用到查询和键向量上。
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference, 将 p-tuning 的前缀与 kv 拼接在一起
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        # 26. 如果需要使用键-值缓存,则将当前的键和值向量保存为缓存。
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            # 27. 如果启用多查询注意力,则对键和值向量进行扩展和形状转换,以便与查询向量相匹配。
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )

        # ==================================
        # core attention computation
        # ==================================
        # 28. 使用 CoreAttention 实例计算注意力输出。
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================
        # 29. 使用线性层从注意力输出计算最终的输出向量。
        output = self.dense(context_layer)

        return output, kv_cache   # 30. 返回输出向量和键-值缓存。


def _config_to_kwargs(args):
    # 1. _config_to_kwargs 是一个辅助函数,用于从配置对象中提取一些公共的关键字参数。
    common_kwargs = {
        # 2. 从配置对象中获取 torch_dtype 参数,并将其作为关键字参数返回。
        #    torch_dtype 参数用于指定张量的数据类型,例如 torch.float16 或 torch.float32。
        "dtype": args.torch_dtype,
    }
    return common_kwargs


class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.

    # 3. MLP 是一个继承自 torch.nn.Module 的 PyTorch 模型类,用于实现多层感知器(Multi-Layer Perceptron)。
    # 4. MLP 将接受隐藏状态作为输入,将其投影到 4 倍的隐藏维度,执行非线性变换,然后再将状态投影回原始的隐藏维度。
    """

    def __init__(self, config: ChatGLMConfig, device=None):
        #    1) config: 一个 ChatGLMConfig 对象,包含模型配置信息。
        #    2) device: 指定使用的设备(CPU 或 GPU),默认为 None。
        super(MLP, self).__init__()
        # 6. 从配置对象中获取是否添加偏置(add_bias_linear)的设置。
        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        # 7. 创建一个线性层,用于将隐藏状态投影到 4 倍的隐藏维度。
        #    如果使用 SwiGLU 激活函数,则输出宽度需要再乘以 2,详见 https://arxiv.org/pdf/2002.05202.pdf。
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
            # 8. swiglu 是一个自定义的激活函数,它将输入张量沿最后一维拆分为两部分,
            #    对第一部分应用 SiLU 激活函数,然后与第二部分相乘。
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]
        # 9. 将 swiglu 函数作为激活函数赋值给 self.activation_func。
        self.activation_func = swiglu

        # Project back to h. , 创建一个线性层,用于将激活函数的输出投影回原始的隐藏维度。
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # 13. 对投影后的张量应用 SwiGLU 激活函数。
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    # 1. GLMBlock 是一个继承自 torch.nn.Module 的 PyTorch 模型类,用于实现 Transformer 的单层块。
    # 2. Transformer 层接受形状为 [sequence_length, batch_size, hidden_size] 的输入,并输出相同形状的结果。
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number    # 4. 保存当前层的编号。
        # 5. 从配置对象中获取是否在层归一化之后应用残差连接(apply_residual_connection_post_layernorm)的设置。
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        # 6. 从配置对象中获取是否使用 fp32 精度进行残差连接(fp32_residual_connection)的设置。
        self.fp32_residual_connection = config.fp32_residual_connection
        # 7. 根据配置中是否使用 RMSNorm,选择使用 RMSNorm 或 LayerNorm 作为层归一化函数。
        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data. 创建一个层归一化层,用于对输入数据进行归一化。
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                             dtype=config.torch_dtype)

        # Self attention. 创建一个 SelfAttention 实例,用于计算自注意力。
        self.self_attention = SelfAttention(config, layer_number, device=device)
        # 从配置对象中获取隐藏层 dropout 的比率。
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output, 创建一个层归一化层,用于对注意力输出进行归一化。
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                      dtype=config.torch_dtype)

        # MLP. 创建一个 MLP 实例,用于计算前馈网络。
        self.mlp = MLP(config, device=device)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        # 13. forward 方法是模型的前向传播入口,接受以下参数:
        #     1) hidden_states: 输入的隐藏状态张量,形状为 [sequence_length, batch_size, hidden_size]。
        #     2) attention_mask: 注意力掩码张量。
        #     3) rotary_pos_emb: 旋转位置编码张量。
        #     4) kv_cache: p-tuning 前缀
        #     5) use_cache: 是否使用键-值缓存,默认为 True。

        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer. 对输入的隐藏状态张量进行层归一化。
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention. 计算自注意力输出和键-值缓存。
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # Residual connection. 根据配置中是否在层归一化之后应用残差连接,选择残差连接的输入。
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # 17. 对注意力输出应用 dropout,并与残差连接相加。
        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention. 对注意力输出后的残差连接结果进行层归一化。
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP. 计算 MLP 输出。
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection. 根据配置中是否在层归一化之后应用残差连接,选择残差连接的输入。
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input
        # 21. 对 MLP 输出应用 dropout,并与残差连接相加。
        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache       # 22. 返回最终的输出张量和键-值缓存。


class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig, device=None):
        #    1) config: 一个 ChatGLMConfig 对象,包含模型配置信息。
        #    2) device: 指定使用的设备(CPU 或 GPU),默认为 None。
        super(GLMTransformer, self).__init__()

        # 3. 从配置对象中获取是否使用 fp32 精度进行残差连接(fp32_residual_connection)的设置。
        self.fp32_residual_connection = config.fp32_residual_connection

         # 4. 从配置对象中获取是否在最后一层之后进行层归一化(post_layer_norm)的设置。
        self.post_layer_norm = config.post_layer_norm

        # Number of layers. 从配置对象中获取 Transformer 层的数量。
        self.num_layers = config.num_layers

        # Transformer layers. 创建一个 ModuleList,用于存储所有 Transformer 层。
        def build_layer(layer_number):
            # build_layer 是一个辅助函数,用于创建 GLMBlock 实例。
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            # 如果配置中指定在最后一层之后进行层归一化,则创建一个层归一化层。
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output. 使用指定的层归一化函数创建最后一层的层归一化层。
            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                 dtype=config.torch_dtype)
        # 10. 设置是否使用梯度检查点(gradient_checkpointing)的标志,默认为 False。
        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        # _get_layer 是一个辅助函数,用于获取指定编号的 Transformer 层。
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        #     1) hidden_states: 输入的隐藏状态张量,形状为 [sequence_length, batch_size, hidden_size]。
        #     2) attention_mask: 注意力掩码张量。
        #     3) rotary_pos_emb: 旋转位置编码张量。
        #     4) kv_caches: p-tuning 前缀
        #     5) use_cache: 是否使用键-值缓存,默认为 True。
        #     6) output_hidden_states: 是否输出所有层的隐藏状态,默认为 False。

        if not kv_caches:
            # 13. 如果没有提供键-值缓存,则创建一个空列表,长度等于 Transformer 层数量。
            kv_caches = [None for _ in range(self.num_layers)]
        # 14. 如果使用键-值缓存,则将其存储在 presents 元组中,否则设置为 None。
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                # 15. 如果同时使用梯度检查点和键-值缓存,则会发出警告并禁用键-值缓存。
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 16. 初始化存储所有层的隐藏状态的列表,如果不需要输出则设置为 None。
        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                # 17. 如果需要输出所有层的隐藏状态,则将当前层的隐藏状态添加到 all_hidden_states 列表中。
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 18. 获取当前层的 GLMBlock 实例。
            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                # 19. 如果使用梯度检查点,则通过 checkpoint 函数计算当前层的输出。
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                # 20. 否则直接调用当前层的前向传播函数计算输出。
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            # 21. 解包当前层的输出,包括隐藏状态和键-值缓存。
            hidden_states, kv_cache = layer_ret
            if use_cache:
                # 22. 如果使用键-值缓存,则将当前层的键-值缓存添加到 presents 元组中。
                presents = presents + (kv_cache,)

        if output_hidden_states:
            # 23. 如果需要输出所有层的隐藏状态,则将最后一层的隐藏状态添加到 all_hidden_states 列表中。
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm. 如果配置中指定在最后一层之后进行层归一化,则对最后一层的隐藏状态进行层归一化。
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
        # 返回最终的隐藏状态张量、键-值缓存列表、所有层的隐藏状态列表和注意力输出(目前为 None)。
        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.

    # 1. ChatGLMPreTrainedModel 是一个抽象类,继承自 PreTrainedModel,用于处理权重初始化和加载预训练模型。
    """

    is_parallelizable = False                  # 2. 设置是否可以并行化的标志,默认为 False。
    supports_gradient_checkpointing = True      # 3. 设置是否支持梯度检查点的标志,默认为 True。
    config_class = ChatGLMConfig               # 4. 指定配置类为 ChatGLMConfig。
    base_model_prefix = "transformer"         # 5. 指定基础模型前缀为 "transformer"。
    _no_split_modules = ["GLMBlock"]           # 6. 指定不应该被分割的模块列表,包括 "GLMBlock"。

    def _init_weights(self, module: nn.Module):
        """
        # 7. _init_weights 是一个用于初始化权重的方法,接受一个 nn.Module 实例作为参数。
        """
        # 8. 在该实现中,该方法没有任何操作。
        return

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        #    1) input_ids: 输入序列的 ID 张量。
        #    2) past_key_values: p-tuning 中的前缀key value向量。
        #    3) padding_mask: 填充掩码张量,默认为 None。

        # 10. 获取批次大小和序列长度。
        batch_size, seq_length = input_ids.shape
        # 11. 创建一个全注意力掩码张量,初始值为 1。
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        # 12. 使用 tril_ 函数将上三角部分设置为 0,形成一个下三角掩码。
        full_attention_mask.tril_()
        past_length = 0  # 13. 初始化 p-tuning 中的前缀key value向量
        if past_key_values:  # p-tuning 中的前缀key value向量序列长度。
            past_length = past_key_values[0][0].shape[0]   

        if past_length:   # 15. 如果p-tuning 中的前缀key value向量长度不为 0,沿着最后一维进行拼接。
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
            
        if padding_mask is not None:  # 16. 如果提供了填充掩码,则将注意力掩码与填充掩码相乘。
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        # TODO 疑问:为什么这里减去 1  答: 为了能够把前pad的数据掩盖掉pad的信息
        if not past_length and padding_mask is not None: 
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        # 18. 将注意力掩码转换为布尔张量。
        full_attention_mask = (full_attention_mask < 0.5).bool()
        # 19. 在注意力掩码的第二维添加一个单一维度。
        full_attention_mask.unsqueeze_(1)
        
        return full_attention_mask  # 20. 返回生成的注意力掩码。

    def get_position_ids(self, input_ids, device):
        # 21. get_position_ids 是一个方法,用于生成位置 ID, 用于计算旋转位置编码
        #     1) input_ids: 输入序列的 ID 张量。
        #     2) device: 设备信息。

        # 22. 获取批次大小和序列长度。
        batch_size, seq_length = input_ids.shape
        # 23. 创建一个位置 ID 张量,形状为 (batch_size, seq_length)。
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids   # 24. 返回生成的位置 ID 张量。

    def _set_gradient_checkpointing(self, module, value=False):
        # _set_gradient_checkpointing 是一个方法,用于设置梯度检查点。
        #     1) module: 要设置的模块。
        #     2) value: 设置值,默认为 False。

        if isinstance(module, GLMTransformer):  # 26. 如果模块是 GLMTransformer 实例,则设置其 gradient_checkpointing 属性。
            module.gradient_checkpointing = value


class Embedding(torch.nn.Module):
    # 1. Embedding 是一个继承自 torch.nn.Module 的 PyTorch 模型类,用于实现语言模型的嵌入层。
    """Language model embeddings."""

    def __init__(self, config: ChatGLMConfig, device=None):
        #    1) config: 一个 ChatGLMConfig 对象,包含模型配置信息。
        #    2) device: 指定使用的设备(CPU 或 GPU),默认为 None。
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size   # 3. 从配置对象中获取隐藏层大小。
        # Word embeddings (parallel).
        # 4. 创建一个词嵌入层,用于将词汇 ID 映射到隐藏层向量。
        #    参数包括:
        #    1) config.padded_vocab_size: 词汇表大小,包括填充词。
        #    2) self.hidden_size: 嵌入向量的维度,即隐藏层大小。
        #    3) dtype: 使用的数据类型,来自配置对象。
        #    4) device: 使用的设备,来自函数参数。
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
            device=device
        )
        # 5. 从配置对象中获取是否使用 fp32 精度进行残差连接(fp32_residual_connection)的设置。
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)  # 7. 通过词嵌入层获取词嵌入向量。
        embeddings = words_embeddings                       # 8. 将词嵌入向量赋值给 embeddings 变量。
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        # 9. 将嵌入向量的维度从 [batch_size, sequence_length, hidden_size]
        #    转换为 [sequence_length, batch_size, hidden_size]。
        #    这种数据格式更适合后续的 Transformer 计算。
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        # 10. 如果配置中指定使用 fp32 精度进行残差连接,则将嵌入向量转换为浮点数。
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class ChatGLMModel(ChatGLMPreTrainedModel):
    # 1. ChatGLMModel 是一个继承自 ChatGLMPreTrainedModel 的 PyTorch 模型类,用于实现 ChatGLM 模型。
    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        #    1) config: 一个 ChatGLMConfig 对象,包含模型配置信息。
        #    2) device: 指定使用的设备(CPU 或 GPU),默认为 None。
        #    3) empty_init: 指定是否使用空初始化,默认为 True。
        super().__init__(config)
        if empty_init:
            # 3. 如果 empty_init 为 True,则使用 skip_init 函数进行初始化。
            init_method = skip_init
        else:
            # 4. 否则使用 default_init 函数进行初始化。
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            # 5. 如果指定了使用设备,则将其作为参数传递给初始化函数。
            init_kwargs["device"] = device
        # 6. 使用指定的初始化方法和参数初始化 Embedding 层。
        self.embedding = init_method(Embedding, config, **init_kwargs)
        # 7. 从配置对象中获取 Transformer 层的数量。
        self.num_layers = config.num_layers
        # 8. 从配置对象中获取多查询组数量。
        self.multi_query_group_num = config.multi_query_group_num
        # 9. 从配置对象中获取键-值通道数量。
        self.kv_channels = config.kv_channels

        # Rotary positional embeddings
        # 10. 从配置对象中获取序列长度。
        self.seq_length = config.seq_length
        # 11. 计算旋转位置编码的维度。
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        # 12. 创建旋转位置编码层。
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope, device=device,
                                              dtype=config.torch_dtype)
        # 13. 使用指定的初始化方法和参数初始化 GLMTransformer 层。
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        # 14. 创建输出层,用于将隐藏状态映射到词汇表。
        self.output_layer = init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
                                        dtype=config.torch_dtype, **init_kwargs)
        # 15. 从配置对象中获取前缀序列长度。
        self.pre_seq_len = config.pre_seq_len
        # 16. 从配置对象中获取前缀投影设置。
        self.prefix_projection = config.prefix_projection
        if self.pre_seq_len is not None:
            # 17. 如果设置了前缀序列长度,则将所有参数设置为不可训练。
            for param in self.parameters():
                param.requires_grad = False
            # 18. 创建一个前缀标记张量。
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            # 19. 创建前缀编码器。
            self.prefix_encoder = PrefixEncoder(config)
            # 20. 创建 Dropout 层。
            self.dropout = torch.nn.Dropout(0.1)

    def get_input_embeddings(self):
        # 21. get_input_embeddings 方法返回模型的输入嵌入层。
        return self.embedding.word_embeddings

    def get_prompt(self, batch_size, device, dtype=torch.half):
        # 22. get_prompt 方法用于生成前缀的键-值缓存。
        #     它接受批次大小、设备和数据类型作为参数。
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.multi_query_group_num,
            self.kv_channels
        )
        # b, seq_len, num_layers * 2,  nh,  kv hidden_size. 对前缀的键-值缓存应用 Dropout。
        past_key_values = self.dropout(past_key_values)
        # 对前缀的键-值缓存进行维度重排,并分割为键和值。[num_layers * 2, seq_len, b, nh,  kv hidden_size ]
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        # 返回前缀的键-值缓存。
        return past_key_values

    def forward(
            self,
            #  输入序列的 ID 张量 (B,T)
            input_ids,
            # 位置 ID 张量, 用于计算旋转位置编码
            position_ids: Optional[torch.Tensor] = None,
            # 注意力掩码张量
            attention_mask: Optional[torch.BoolTensor] = None,
            # 加入了p-tuning 中的前缀key value向量后的全注意力掩码张量
            full_attention_mask: Optional[torch.BoolTensor] = None,
            # 在这里表示 p-tuning 中的前缀key value向量
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            # 是否使用k v缓存, 加速推理过程
            use_cache: Optional[bool] = None,
            # 是否输出所有层的隐藏状态
            output_hidden_states: Optional[bool] = None,
            # 是否返回字典格式的输出
            return_dict: Optional[bool] = None,
    ):

        # 27. 根据配置设置输出所有层的隐藏状态的标志。
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 28. 根据配置设置使用键-值缓存的标志。
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 29. 根据配置设置返回字典格式的标志。
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 30. 获取批次大小和序列长度。
        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            # 31. 如果没有提供输入嵌入,则通过嵌入层计算。
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None: # 判断是否使用 p-tuning
            if past_key_values is None:
                # 计算p-tuning 中的前缀键-值缓存。
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                # p-tuning 中的前缀的attention与输入的attention进行拼接。
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                # 计算全注意力掩码(加入 p-tuning 中的前缀的attention, attention mask, casul attention)。
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        # 35. 计算旋转位置编码。
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            # 36. 如果提供了位置 ID,则使用位置 ID 索引旋转位置编码。
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            # 37. 否则使用序列长度索引旋转位置编码。
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        # 38. 对旋转位置编码进行维度重排。
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        # 39. 通过 Transformer 编码器进行计算。
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            # 40. 如果不返回字典格式的输出,则返回一个元组。
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        # 41. 否则返回一个包含最终隐藏状态、键-值缓存、所有层的隐藏状态和注意力输出的 BaseModelOutputWithPast 对象。
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def quantize(self, weight_bit_width: int):
        # 42. quantize 方法用于量化模型权重,接受权重位宽作为参数。
        from .quantization import quantize
        # 43. 调用量化函数对编码器进行量化。
        quantize(self.encoder, weight_bit_width)
        return self  # 44. 返回量化后的模型实例。


class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    # 1. ChatGLMForConditionalGeneration 是一个继承自 ChatGLMPreTrainedModel 的 PyTorch 模型类,用于实现条件文本生成任务,例如对话系统、机器翻译等。
    #    它结合了 ChatGLMModel 和一个输出层,能够根据输入序列生成相应的条件输出序列。
    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        #    1) config: 一个 ChatGLMConfig 对象,包含模型的配置信息,如词汇表大小、隐藏层大小、层数等。
        #    2) empty_init: 指定是否使用空初始化,即不初始化模型参数,默认为 True。
        #    3) device: 指定使用的设备(CPU 或 GPU),默认为 None,表示使用默认设备。
        super().__init__(config)
        # 3. 从配置对象中获取最大序列长度,并将其赋值给实例属性 max_sequence_length。
        #    这个值用于限制输入和输出序列的最大长度,防止过长的序列导致内存不足。
        self.max_sequence_length = config.max_length
        # 4. 创建 ChatGLMModel 实例,用于实现基本的 Transformer 模型。
        #    ChatGLMModel 包含了嵌入层、编码器层和输出层
        #    传递的参数包括配置对象、是否使用空初始化和设备。
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)
        self.config = config
        # 6. 初始化一个布尔标志 quantized,用于指示模型是否已经进行了量化。
        #    量化是一种压缩模型的技术,可以减小模型的大小和内存占用,但可能会导致一定程度的精度损失。
        self.quantized = False

        if self.config.quantization_bit:
            # 7. 如果配置对象中指定了量化位宽(quantization_bit),则调用 quantize 方法对模型进行量化。
            #    量化位宽越小,模型压缩程度越高,但精度损失也越大。
            #    该方法将对 Transformer 编码器进行量化,量化后的模型会替换当前实例。
            self.quantize(self.config.quantization_bit, empty_init=True)

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # 8. _update_model_kwargs_for_generation 方法用于在每个生成步骤中更新模型的关键字参数。
        #    它接受以下参数:
        #    1) outputs: 模型的前一步输出,包含隐藏状态、注意力权重等信息。
        #    2) model_kwargs: 当前的模型关键字参数字典,包含输入 ID、注意力掩码、位置 ID 等信息。
        #    3) is_encoder_decoder: 是否为编码器-解码器模型,默认为 False。
        #    4) standardize_cache_format: 是否标准化缓存格式,默认为 False。


        # 9. 更新 past_key_values 参数,用于缓存之前的键-值对。_extract_past_from_model_output 方法从模型输出中提取键-值对,并按照标准格式进行排列。
        #    这些键-值对将用于下一步的计算,以提高效率。
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        # 10. 如果存在注意力掩码,则将其扩展一位。
        #     这是因为在生成过程中,每一步都会生成一个新的标记,因此需要为新的标记提供对应的注意力掩码。
        #     该操作通过在原有注意力掩码的末尾追加一个 1 来实现,表示新的标记不受任何掩码的影响。
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        # 11. 如果存在位置 ID,则将其扩展一位。
        #     这是因为在生成过程中,每一步都会生成一个新的标记,因此需要为新的标记提供对应的位置 ID。
        #     该操作通过复制最后一个位置 ID,并将其值加 1 来实现,表示新的标记位于序列的最后一个位置。
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        # 12. 设置 is_first_forward 参数为 False。
        #     这个参数用于指示当前步骤是否为生成过程的第一步。在第一步之后,该参数将被设置为 False。
        model_kwargs["is_first_forward"] = False
        return model_kwargs  # 13. 返回更新后的模型关键字参数字典。

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:
        # 14. prepare_inputs_for_generation 方法用于准备生成过程中所需的输入数据。
        #     它接受以下参数:
        #     1) input_ids: 输入序列的 ID 张量。
        #     2) past_key_values: 用于推理的前一个键-值缓存,默认为 None。
        #     3) attention_mask: 注意力掩码张量,默认为 None。
        #     4) position_ids: 位置 ID 张量,默认为 None。
        #     5) use_cache: 是否使用键-值缓存,默认为 None。
        #     6) is_first_forward: 是否为第一次前向传播,默认为 True。
        #     7) **kwargs: 其他关键字参数。

        # 15. 如果未提供位置 ID,则计算位置 ID。
        #     get_position_ids 方法根据输入序列的长度生成对应的位置 ID 张量。
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)
        if not is_first_forward:
            # 16. 如果不是第一次前向传播,且存在前一个键-值缓存,则只保留最后一个位置的 ID 和输入 ID。
            #     这是因为在生成过程中,每一步只需要关注当前位置的输入和之前的缓存,而不需要考虑之前所有位置的输入。
            #     这种做法可以节省内存和计算资源。
            if past_key_values is not None:
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]

        # 17. 返回一个字典,包含输入 ID、键-值缓存、位置 ID、注意力掩码、是否返回最后一个 logit 和是否使用缓存。
        #     这些数据将被用作模型的输入,进行文本生成。
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "use_cache": use_cache
        }

    def forward(
            self,
            # 输入序列的 ID 张量, (B,T)
            input_ids: Optional[torch.Tensor] = None,
            # 位置 ID 张量, 用于计算选装位置编码。
            position_ids: Optional[torch.Tensor] = None,
            # 正常的注意力掩码张量
            attention_mask: Optional[torch.Tensor] = None,
            # 在这里指的是p-tuning 中的前缀key value向量
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            # 标签张量
            labels: Optional[torch.Tensor] = None,
            # 是否使用k v缓存, 加速推理的速度
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            # 是否输出所有层的隐藏状态
            output_hidden_states: Optional[bool] = None,
            # 是否返回字典格式的输出
            return_dict: Optional[bool] = None,
            # 是否仅返回最后一个logit
            return_last_logit: Optional[bool] = False,
    ):

        # 19. 根据配置设置使用键-值缓存和返回字典格式的标志。
        #     如果参数为 None,则使用配置对象中的默认值。
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 20. 通过 ChatGLMModel 进行前向传播计算,获取 Transformer 输出。
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 21. 从 Transformer 输出中获取最后一层的隐藏状态。
        hidden_states = transformer_outputs[0]
        if return_last_logit:
            # 22. 如果返回最后一个 logit，则只返回最后一个token的隐藏状态, 因为 chatglm是 left pad, 所以这里取最后一个就是真实的最后一个
            hidden_states = hidden_states[-1:]

        # 23. 通过输出层将隐藏状态映射到 logit。
        #     输出层是一个线性层,将隐藏状态的维度从隐藏层大小映射到词汇表大小。
        lm_logits = self.transformer.output_layer(hidden_states)

        # 24. 对 logit 进行维度重排。
        #     原始的 logit 维度为 [batch_size, sequence_length, vocab_size]。
        #     通过维度重排,将其变为 [sequence_length, batch_size, vocab_size],以便更好地进行后续处理。
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None  # 25. 初始化损失为 None。
        if labels is not None:
            # 26. 如果提供了标签,则计算损失。
            #     首先将 logit 转换为 float32 精度,以便计算交叉熵损失。
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            # 27. 将 logit 和标签进行位移,以便让当前位置的 logit 预测下一个位置的标签。
            #     这是因为语言模型通常在时间步 t 预测时间步 t+1 的标记。
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # 29. 将 logit 和损失转换为与隐藏状态相同的数据类型。
            #     这是为了确保在后续计算中不会出现数据类型不匹配的问题。
            #     通常隐藏状态使用较低精度(如 float16)以节省内存,而 logit 和损失计算使用较高精度(如 float32)以保证精度。
            #     计算完成后,需要将它们转换回隐藏状态的数据类型。
            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            # 30. 如果不返回字典格式的输出,则返回一个元组。
            #     元组中包含 logit、键-值缓存、所有层的隐藏状态和注意力权重。
            #     如果计算了损失,则在元组的开头添加损失值。
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        # 31. 否则返回一个包含 logit、损失、键-值缓存、所有层的隐藏状态和注意力权重的 CausalLMOutputWithPast 对象。
        #     CausalLMOutputWithPast 是一个 PyTorch 的命名元组,用于方便访问各个输出值。
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        """
        # 32. _reorder_cache 是一个静态方法,用于在 beam search 或 beam sample 时重新排序缓存的键-值对。
        #     beam search 和 beam sample 是两种常用的文本生成策略,通过保留多个候选序列并进行评分,从而获得更好的生成结果。
        #     在这个过程中,由于每个候选序列对应不同的缓存,因此需要根据每个步骤的得分对缓存进行重新排序,以匹配正确的 beam_idx。
        #     
        #     该方法接受以下参数:
        #     1) past: 一个元组,包含每一层的键-值对缓存。每个元素是一对(键张量,值张量)。
        #     2) beam_idx: 一个 LongTensor,表示当前步骤中每个候选序列对应的 beam 索引。
        #     
        #     输出与 `past` 共享内存存储,但是重新排列了缓存的顺序,以匹配 beam_idx。
        """
        return tuple(
            (   # 对于每一层的键张量和值张量,根据 beam_idx 重新索引它们的第二个维度(batch 维度)。
                # index_select(dim, index) 函数根据指定维度和索引张量,从原始张量中选择出相应的切片。
                # 这样可以确保每个候选序列的缓存正确对应于其 beam 索引。
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    def process_response(self, response):
        # 33. process_response 方法用于对生成的响应进行后处理。
        #     这里的后处理操作包括:
        #     1) 去除响应两端的空白字符。
        #     2) 将特殊标记 "[[训练时间]]" 替换为 "2023年"。
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        return response

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        # 34. build_inputs 方法用于构建模型输入。
        #     它接受以下参数:
        #     1) tokenizer: 用于对文本进行tokenize的tokenizer对象。
        #     2) query: 查询字符串,即当前的输入文本。
        #     3) history: 对话历史记录,是一个列表,每个元素是一个(query, response)对。
        #     
        #     该方法首先使用 tokenizer.build_prompt 方法构建输入的提示(prompt),
        #     这通常会将当前的查询和历史记录组合在一起,形成一个完整的输入序列。
        #     然后,它使用 tokenizer 对提示进行tokenize,并将结果转换为 PyTorch 张量。
        #     最后,将输入张量移动到当前设备上。
        prompt = tokenizer.build_prompt(query, history=history)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def build_stream_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        # 35. build_stream_inputs 方法用于构建流式输入。
        #     它接受以下参数:
        #     1) tokenizer: 用于对文本进行tokenize的tokenizer对象。
        #     2) query: 查询字符串,即当前的输入文本。
        #     3) history: 对话历史记录,是一个列表,每个元素是一个(query, response)对。
        #     
        #     该方法的作用是为流式生成输入做准备。它会根据历史记录的长度,构建一个特定格式的提示序列。
        #     如果存在历史记录,则提示序列的格式为:
        #     "\n\n[Round {round_num}]\n\n问：{query}\n\n答："
        #     如果没有历史记录,则提示序列的格式为:
        #     "[Round {round_num}]\n\n问：{query}\n\n答："
        #     
        #     提示序列中包含了当前回合的编号和查询内容。然后,该方法使用 tokenizer 对提示进行tokenize,并将结果转换为 PyTorch 张量。
        #     最后,将输入张量移动到当前设备上。
        if history:
            prompt = "\n\n[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = input_ids[1:]
            inputs = tokenizer.batch_encode_plus([(input_ids, None)], return_tensors="pt", add_special_tokens=False)
        else:
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    # 这段代码使用了以下一些关键技术和原因:
    # PyTorch 推理模式: 通过 @torch.inference_mode() 装饰器,可以在推理阶段关闭梯度计算和自动微分,提高性能和内存利用率。
    # LogitsProcessorList: 用于管理多个 logits 处理器,这里添加了 InvalidScoreLogitsProcessor 来处理无效的分数,确保生成的输出更加合理。
    # 生成参数: 通过设置 max_length、num_beams、do_sample、top_p、temperature 等参数,可以控制文本生成的过程,例如限制最大长度、调整采样策略等。
    # 输入构建: 根据当前 query 和对话历史 history 构建模型的输入序列,以便模型理解上下文信息。
    # 输出处理: 将生成的输出序列转换为文本,并调用 process_response 方法进一步处理,确保响应更加合理和人性化。
    # 对话历史更新: 将当前对话加入到对话历史中,以便后续的对话能够建立在前文的基础上。
    @torch.inference_mode()
    def chat(self, tokenizer,          # 用于tokenize的tokenizer对象。
                query: str,        # query: 查询字符串,即当前的输入文本
                history: List[Tuple[str, str]] = None,  # history: 对话历史记录,是一个列表,每个元素是一个(query, response)对,默认为 None。
                max_length: int = 8192,       # max_length: 生成序列的最大长度,默认为 8192。
                num_beams=1,         # num_beams: beam search 的beam数量,默认为 1(表示不使用beam search)。
                do_sample=True,      # do_sample: 是否进行随机采样,默认为 True。
                top_p=0.8, temperature=0.8,  # top_p: 对候选标记进行 top-p 采样的阈值,默认为 0.8。
                logits_processor=None,       # 用于处理 logit 的处理器,默认为 None。
                **kwargs):
        # 1. 如果 history 为 None,则初始化为空列表
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        # 添加 InvalidScoreLogitsProcessor,用于处理无效的分数
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, 
                      "logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs(tokenizer, query, history=history)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history

    # PyTorch 推理模式: 通过 @torch.inference_mode() 装饰器,可以在推理阶段关闭梯度计算和自动微分,提高性能和内存利用率。
    @torch.inference_mode()
    def stream_chat(self, 
                tokenizer,  # 用于tokenize的tokenizer对象。
                query: str,   # query: 查询字符串,即当前的输入文本。
                history: List[Tuple[str, str]] = None,  # 对话历史记录,是一个列表,每个元素是一个(query, response)对,默认为 None。
                past_key_values=None,  # past_key_values: 用于推理的前一个键-值缓存,默认为 None。
                max_length: int = 8192,  # max_length: 生成序列的最大长度,默认为 8192。
                do_sample=True, top_p=0.8, temperature=0.8, 
                logits_processor=None,  # logits_processor: 用于处理 logit 的处理器,默认为 None。
                return_past_key_values=False,  # 是否返回键-值缓存,默认为 False。
                **kwargs):

        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        # 3. 向 logits_processor 中添加 InvalidScoreLogitsProcessor
        # 用于处理无效的分数,防止生成不合适的输出
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, 
                      "logits_processor": logits_processor, **kwargs}
        # 5. 根据 past_key_values 是否存在,构建不同的输入
        if past_key_values is None and not return_past_key_values:
            inputs = self.build_inputs(tokenizer, query, history=history)
        else:
            inputs = self.build_stream_inputs(tokenizer, query, history=history)
        # 6. 如果存在 past_key_values,则更新 inputs 中的 position_ids 和 attention_mask
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[0]
            if self.transformer.pre_seq_len is not None:
                past_length -= self.transformer.pre_seq_len
            inputs.position_ids += past_length
            attention_mask = inputs.attention_mask
            attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
            inputs['attention_mask'] = attention_mask
         # 7. 通过流式生成方式获取输出
        for outputs in self.stream_generate(**inputs, 
                                            past_key_values=past_key_values,
                                            return_past_key_values=return_past_key_values, 
                                            **gen_kwargs):
            # 如果需要返回 past_key_values,则将其从输出中分离出来
            if return_past_key_values:
                outputs, past_key_values = outputs
            # 将生成的输出序列转换为文本
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            # 对响应进行进一步处理,确保输出合理
            if response and response[-1] != "�":
                response = self.process_response(response)
                new_history = history + [(query, response)] # 更新对话历史
                if return_past_key_values: # 根据是否需要返回 past_key_values,产生不同的输出
                    yield response, new_history, past_key_values
                else:
                    yield response, new_history

    # PyTorch 推理模式: 通过 @torch.inference_mode() 装饰器,可以在推理阶段关闭梯度计算和自动微分,提高性能和内存利用率。
    @torch.inference_mode()
    def stream_generate(
            self,
            input_ids,  # 输入序列的 ID 张量。
            generation_config: Optional[GenerationConfig] = None,  # 生成配置对象,默认为 None。如果为 None,则使用模型的默认配置。
            logits_processor: Optional[LogitsProcessorList] = None, # 用于处理 logit 的处理器列表,默认为 None。
            stopping_criteria: Optional[StoppingCriteriaList] = None,  # 用于判断何时停止生成的停止条件列表,默认为 None。
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None, # 一个函数,用于指定在特定位置允许生成的标记,默认为 None。
            return_past_key_values=False,  # 是否返回键-值缓存,默认为 False。
            **kwargs,  # 其他关键字参数,将被传递给 generate 方法。
    ):

        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
        # 39. 如果没有提供 generation_config,则使用模型的默认配置。
        if generation_config is None:
            generation_config = self.generation_config
        # 40. 创建 generation_config 的深拷贝,以避免修改原始配置。
        generation_config = copy.deepcopy(generation_config)
        # 41. 根据传入的关键字参数更新生成配置。
        model_kwargs = generation_config.update(**kwargs)
        # 42. 设置是否使用键-值缓存的标志。
        model_kwargs["use_cache"] = generation_config.use_cache
        # 43. 获取开始标记和结束标记的 ID。
        bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id
        # 44. 如果 eos_token_id 是一个整数,则将其转换为列表,以便后续处理。
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # 45. 检查是否使用了默认的 max_length 参数,以及是否设置了 max_new_tokens 参数。
        #     如果使用了默认的 max_length 且没有设置 max_new_tokens,则发出警告,因为这种行为在未来版本中可能会被弃用。
        #     如果设置了 max_new_tokens,则使用它和输入序列长度的总和作为 max_length。
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        # 46. 如果输入序列的长度已经超过了 max_length,则发出警告,因为这可能会导致意外的行为。
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        # 47. 如果没有提供 logits_processor,则创建一个空的 LogitsProcessorList。
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        # 48. 如果没有提供 stopping_criteria,则创建一个空的 StoppingCriteriaList。
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        # 49. 获取用于处理 logit 的处理器。
        #     _get_logits_processor 方法根据生成配置、输入序列长度、编码器输入 ID 和前缀允许标记函数,
        #     构建合适的 logits_processor。
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 50. 获取用于判断是否应该停止生成的停止条件。
        #     _get_stopping_criteria 方法根据生成配置和提供的 stopping_criteria 列表,
        #     构建合适的停止条件。
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 51. 获取用于对 logit 进行处理的 warper。
        #     _get_logits_warper 方法根据生成配置,返回合适的 logits_warper。
        logits_warper = self._get_logits_warper(generation_config)
        # 52. 创建一个布尔张量,用于跟踪哪些序列已经完成生成。
        #     初始时,所有序列都被标记为未完成。
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None  # 53. 初始化分数张量为 None。

        # 54. 开始流式生成的主循环。
        while True:
            # 55. 准备模型输入。
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            # 56. 进行前向传播,获取输出。
            #     返回的输出包括 logit、键-值缓存、隐藏状态和注意力权重。
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            # 57. 从输出中获取当前时间步的 logit。
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            # 58. 对 logit 进行预处理。
            #     logits_processor 中包含一系列处理器,用于对 logit 进行修改,
            #     例如执行 top-k 或 top-p 采样、应用禁止词列表等。
            next_token_scores = logits_processor(input_ids, next_token_logits)
            # 59. 对预处理后的 logit 应用 logits_warper。
            #     logits_warper 可以进一步修改 logit,例如应用温度值。
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            # 60. 从修改后的 logit 中采样下一个标记。
            #     如果 generation_config.do_sample 为 True,则进行随机采样;
            #     否则,选择具有最大 logit 值的标记。
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            # 61. 更新输入 ID、模型关键字参数和未完成序列的标记。
            #     将新生成的标记添加到输入 ID 的末尾。
            #     调用 _update_model_kwargs_for_generation 方法更新模型关键字参数。
            #     如果新生成的标记是结束标记,则将对应序列标记为已完成。
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())
            # 62. 如果需要返回键-值缓存,则将其与输入 ID 一起返回。
            #     这在流式生成中很有用,因为我们可以在下一次迭代中重用缓存,从而提高效率。
            if return_past_key_values:
                yield input_ids, outputs.past_key_values
            else:
                yield input_ids
            # stop when each sentence is finished, or if we exceed the maximum length
            # 63. 如果所有序列都已完成生成,或者满足了停止条件,则退出循环。
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

    def quantize(self, bits: int, empty_init=False, device=None, **kwargs):
        # 1. quantize 方法用于对模型进行量化,即将模型的浮点数参数转换为定点数表示。
        #    量化可以有效减小模型的大小,从而节省存储空间和内存占用,同时还能提高推理速度。
        #    该方法接受以下参数:
        #    1) bits: 一个整数,表示量化的位宽。更小的位宽可以获得更高的压缩率,但也会导致精度损失。
        #    2) empty_init: 一个布尔值,指示是否使用空初始化。如果为 True,则不初始化模型参数。
        #    3) device: 设备(如 CPU 或 GPU)的字符串标识符。如果为 None,则使用默认设备。
        #    4) **kwargs: 其他关键字参数,将被传递给量化函数。
        if bits == 0:
            return

        from .quantization import quantize

        if self.quantized:
            # 4. 如果模型已经被量化,则打印日志信息并直接返回。
            logger.info("Already quantized.")
            return self

        self.quantized = True  # 5. 将 quantized 标志设置为 True,表示模型已经被量化。

        self.config.quantization_bit = bits  # 6. 将量化位宽存储在模型配置对象中。
        # 7. 调用 quantize 函数对模型的编码器部分进行量化。
        #    quantize 函数接受以下参数:
        #    1) self.transformer.encoder: 要量化的编码器模块。
        #    2) bits: 量化位宽。
        #    3) empty_init: 是否使用空初始化。
        #    4) device: 设备标识符。
        #    5) **kwargs: 其他关键字参数。
        #    quantize 函数会返回一个量化后的编码器模块,并将其赋值给 self.transformer.encoder。
        self.transformer.encoder = quantize(self.transformer.encoder, bits, empty_init=empty_init, device=device,
                                            **kwargs)
        return self


class ChatGLMForSequenceClassification(ChatGLMPreTrainedModel):
    # 1. 定义一个名为 ChatGLMForSequenceClassification 的类,继承自 ChatGLMPreTrainedModel。
    #    这个类用于执行序列分类任务,如文本分类、情感分析等。
    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        super().__init__(config)

        self.num_labels = config.num_labels  # 4. 从模型配置中获取分类标签的数量,并将其存储在 num_labels 属性中。
        # 5. 初始化一个 ChatGLMModel 对象,作为模型的编码器部分。
        #    ChatGLMModel 是一个预训练的语言模型,用于编码输入序列。
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)
        # 6. 初始化一个全连接层,作为分类头。
        #    1) 全连接层的输入维度为模型的隐藏大小。
        #    2) 输出维度为标签数量。
        #    3) 使用半精度浮点数(torch.half)进行计算,以节省内存。
        self.classifier_head = nn.Linear(config.hidden_size, config.num_labels, bias=True, dtype=torch.half)
        if config.classifier_dropout is not None:
            # 7. 根据模型配置中的 classifier_dropout 参数,决定是否初始化一个 Dropout 层。
            #    Dropout 层用于防止过拟合,通过随机丢弃一些神经元的输出来实现。
            self.dropout = nn.Dropout(config.classifier_dropout)
        else:
            self.dropout = None
        self.config = config

        if self.config.quantization_bit:
            # 9. 如果模型配置中指定了量化位宽(quantization_bit),则调用 quantize 方法对模型进行量化。
            #    量化可以减小模型大小,提高推理速度,但会带来一定精度损失。
            self.quantize(self.config.quantization_bit, empty_init=True)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            full_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutputWithPast]:
        #     该函数接受以下参数:
        #     1) input_ids: 输入序列的标记 ID。
        #     2) position_ids: 输入序列的位置 ID。
        #     3) attention_mask: 用于计算注意力权重的掩码张量。
        #     4) full_attention_mask: 用于计算全注意力的掩码张量。
        #     5) past_key_values: 用于序列生成的前一个键-值缓存。
        #     6) inputs_embeds: 输入序列的嵌入表示。
        #     7) labels: 输入序列的标签。
        #     8) use_cache: 是否使用缓存。
        #     9) output_hidden_states: 是否输出隐藏状态。
        #     10) return_dict: 是否以字典形式返回输出。
        #     该函数返回一个元组或 SequenceClassifierOutputWithPast 对象,包含分类的 logit 和可选的损失值、过去的键-值缓存、隐藏状态和注意力权重。

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 12. 调用 ChatGLMModel 的前向传播,获取编码器的输出。
        #     transformer_outputs 是一个包含隐藏状态、键-值缓存和注意力权重的元组或命名元组。
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            full_attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        pooled_hidden_states = hidden_states[-1]  

        if self.dropout is not None:
            # 14. 如果初始化了 Dropout 层,则对隐藏状态应用 Dropout 正则化,防止过拟合。
            pooled_hidden_states = self.dropout(pooled_hidden_states)
        # 15. 将隐藏状态输入到分类头(全连接层),获得分类的 logit。
        logits = self.classifier_head(pooled_hidden_states)

        loss = None
        if labels is not None:
            # 16. 如果提供了标签,则计算损失。
            if self.config.problem_type is None:
                # 17. 如果模型配置中没有指定问题类型,则根据标签的数量和数据类型自动推断问题类型。
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                # 18. 如果是回归问题,使用均方误差损失函数(MSELoss)计算损失。
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze().float(), labels.squeeze())
                else:
                    loss = loss_fct(logits.float(), labels)
            elif self.config.problem_type == "single_label_classification":
                # 19. 如果是单标签分类问题,使用交叉熵损失函数(CrossEntropyLoss)计算损失。
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 20. 如果是多标签分类问题,使用二值交叉熵损失函数(BCEWithLogitsLoss)计算损失。
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.float(), labels.view(-1, self.num_labels))

        # 21. 如果不以字典形式返回输出,则返回一个元组,包含 logit、键-值缓存、隐藏状态和注意力权重。
        #     如果计算了损失,则在元组的开头添加损失值。
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 22. 否则,返回一个 SequenceClassifierOutputWithPast 对象,包含损失、logit、键-值缓存、隐藏状态和注意力权重。
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
