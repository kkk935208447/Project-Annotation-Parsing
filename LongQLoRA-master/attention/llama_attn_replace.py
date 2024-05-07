# Modified based on https://github.com/lm-sys/FastChat

import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import transformers
from einops import rearrange  # 1. einops 是一个用于高效操作多维数据(如 Tensor)的库,rearrange 函数可以方便地调整 Tensor 的维度和形状。
from flash_attn import __version__ as flash_attn_version
# 3. 从 flash_attn.bert_padding 模块导入 pad_input 和 unpad_input 函数。
    # pad_input 用于在输入数据(通常是 Tensor)的末尾添加填充元素,使其长度达到指定值。
    # unpad_input 则用于移除输入数据末尾的填充元素。
    # 在处理变长序列时,这两个函数可以确保输入数据具有相同长度,以满足模型的输入要求。
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,                  # flash_attn_func: 是 Flash Attention 的核心函数,用于计算注意力权重和加权和。
    flash_attn_varlen_kvpacked_func,  # flash_attn_varlen_kvpacked_func: 用于处理变长 key-value 对的 Flash Attention 计算。
    flash_attn_varlen_qkvpacked_func  # flash_attn_varlen_qkvpacked_func: 用于处理变长 query-key-value 的 Flash Attention 计算。
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,       # apply_rotary_pos_emb: 应用旋转位置编码(Rotary Position Embedding, RPE)。RPE 是一种位置编码方式,可以更好地捕获序列的位置信息。
    repeat_kv,                  # repeat_kv: 重复 key 和 value,使其维度与 query 相匹配。
    rotate_half                 # rotate_half: 对 value 的一半维度应用旋转操作,这是 RPE 的一部分。
)
# TODO 注释掉了官方代码, 疑似重复导入
# from flash_attn.bert_padding import unpad_input, pad_input
import math

group_size_ratio = 1/4  # 分组的比例, 组大小 T * group_size_ratio
# group_size_ratio = 1/8  
def forward_flashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    # 10. 如果不是训练模式,抛出异常,因为该函数只适用于训练. s2atten 训练时使用分组, 推理的时候用的standard attention, 具体看相关论文 
    if not self.training:
        raise ValueError("This function is only for training. For inference, please use forward_flashattn_inference.")
    # 11. 如果需要输出注意力权重,发出警告,因为这里不支持输出注意力权重。
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()  # 12. 获取输入张量的 batch size、序列长度和隐藏层维度。
    # 13. 计算 query、key 和 value 的状态。使用线性变换将输入编码为 query、key 和 value,并将其调整为 [bsz, nh, q_len, hd] 的形状,便于后续计算。
    # [bsz, q_len, nh, hd]  ---->   [bsz, nh, q_len, hd]
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    
    # 14. 计算 key-value 序列长度,用于 Rotary Position Embedding。
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    # 15. 如果存在 past_key_value,将其拼接到 key_states 和 value_states 中。past_key_value 是用于序列生成任务的先前计算过的 key 和 value 状态,在生成新的 token 时,需要将其与当前计算的 key 和 value 拼接起来。
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    # 16. 如果需要使用缓存,则将 key_states 和 value_states 作为 past_key_value 返回。在序列生成任务中,为了加速计算,我们可以缓存已经计算过的 key 和 value,避免重复计算。
    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    # 广播: 如果 num_key_value_heads 小于 num_heads,重复 key 和 value 以匹配 num_heads。在某些情况下,模型的 key 和 value 头数量可能小于 query 头数量,为了保持一致性,需要重复 key 和 value。
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention, 
    # 将 query、key 和 value 拼接成 qkv tensor,并调整形状。
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    # 由于禁用了 _prepare_decoder_attention_mask 函数,因此注意力掩码与 key_padding_mask 相同
    key_padding_mask = attention_mask.repeat(2, 1)
    nheads = qkv.shape[-2]
    # shift
    # 19. 根据 q_len 计算 group_size,用于 Flash Attention。group_size 决定了每个 group 的长度,是一个重要的超参数,影响计算效率和内存使用。
    group_size = int(q_len * group_size_ratio)
    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d." % (q_len, group_size))
    # 进行 shift 2 attention, [b, t, 3, 2, nh//2, d] -> [b, 2, t, 3, nh//2, d] -> [b * 2, t, 3, nh//2, d]
    qkv = qkv.reshape(bsz, q_len, 3, 2, self.num_heads // 2, self.head_dim).permute(0, 3, 1, 2, 4, 5).reshape(bsz * 2,
                                                                                                              q_len, 3,
                                                                                                              self.num_heads // 2,
                                                                                                              self.head_dim)
    # 将 qkv 张量从 [bsz, q_len, 3, nh, hd] 的形状重排为 [bsz, q_len, (3 * nh * hd)], 其中 3 表示 query、key 和 value 的维度,nh 表示注意力头数,hd 表示每个注意力头的维度, 这种重排是为了方便后续的计算和处理
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    # 调用 unpad_input 函数,去除输入张量中的填充部分, 输入为 x 和 key_padding_mask
    # 返回值包括: x_unpad: 去除填充后的张量, indices: 用于重新填充的索引张量, cu_q_lens: 每个序列的实际长度, max_s: 去填充后的最大序列长度
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    # 计算 cu_q_len_tmp,表示每个 group 的起始位置, 使用 torch.arange 生成从 0 到 max_s 的等差数列,步长为 group_size, 并将其在 GPU 上创建,数据类型与 cu_q_lens 相同
    cu_q_len_tmp = torch.arange(0, max_s, group_size, device=key_padding_mask.device, dtype=cu_q_lens.dtype)
    # 将 cu_q_len_tmp 分成两部分,一部分表示每个 group 的起始位置,另一部分表示每个 group 的中点位置, 然后在第一维度上重复 bsz 次,并将其与每个序列的实际长度相加,得到每个 group 的起始位置和中点位置
    cu_q_len_tmp = torch.stack([cu_q_len_tmp, cu_q_len_tmp + group_size // 2]).repeat(bsz, 1) + cu_q_lens[:-1].unsqueeze(-1)
    # 将 cu_q_len_tmp 与 cu_q_lens 拼接起来,得到每个 group 的起始位置、中点位置和每个序列的实际长度, 然后将结果展平,以方便后续的计算
    cu_q_lens = torch.cat([cu_q_len_tmp, cu_q_lens[1:].unsqueeze(-1)], dim=-1).view(-1)
    # 将 x_unpad 从 [nnz, (3 * nh * hd)] 的形状重排为 [nnz, 3, nh//2, hd], 其中 nnz 表示去填充后的非零元素数量,3 表示 query、key 和 value 的维度, nh//2 表示注意力头数被平均分成两半,hd 表示每个注意力头的维度
    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads // 2
    )
    # 调用 flash_attn_varlen_qkvpacked_func 函数,执行 Flash Attention 计算
    # 输入包括: x_unpad: 重排后的输入张量, cu_q_lens: 每个 group 的起始位置、中点位置和每个序列的实际长度, group_size: 每个 group 的大小, 0.0: 用于初始化 dropout 的值
        # softmax_scale: softmax 缩放因子,设置为 None 表示不使用缩放, causal=True: 表示使用因果注意力机制
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, group_size, 0.0, softmax_scale=None, causal=True
    )
    # 将 output_unpad 从 [nnz, nh//2, hd] 的形状重排为 [nnz, (nh//2 * hd)], 然后调用 pad_input 函数,将其填充回原始形状 [bsz * 2, q_len, (nh//2 * hd)]
        # 其中 bsz * 2 是因为之前进行了 shift 2 attention 操作,导致 batch size 扩大了一倍
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz * 2, q_len
        ),
        "b s (h d) -> b s h d",
        h=nheads // 2,
    )
    # 将 output 从 [bsz * 2, q_len, (nh//2 * hd)] 的形状调整为 [bsz, q_len, nh, hd], 首先将其重塑为 [bsz, 2, q_len, nh//2, hd]
        # 然后在第一和第二维度上进行转置,得到 [bsz, q_len, 2, nh//2, hd], 最后将第二和第四维度合并,得到 [bsz, q_len, nh, hd]
    output = output.reshape(bsz, 2, q_len, nheads // 2, self.head_dim).transpose(1, 2).reshape(bsz, q_len, nheads,
                                                                                               self.head_dim)
    # 调用 self.o_proj 函数,将 output 从 [bsz, q_len, nh, hd] 的形状映射回 [bsz, q_len, hidden_dim], 同时返回 None 作为注意力权重,以及 past_key_value
    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value

def forward_flashattn_full(
    self,
    hidden_states: torch.Tensor,  # 输入的隐藏状态张量
    attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量
    position_ids: Optional[torch.Tensor] = None,  # 可选的位置 ID 张量
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 可选的先前计算过的 key 和 value 状态
    output_attentions: bool = False,  # 是否输出注意力权重,默认为 False
    use_cache: bool = False,    # 是否使用缓存,默认为 False
    padding_mask: Optional[torch.LongTensor] = None,  # 可选的填充掩码张量
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    # 如果需要输出注意力权重,发出警告,因为这里不支持输出注意力权重
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )
    # 获取输入张量的 batch size、序列长度和隐藏层维度
    bsz, q_len, _ = hidden_states.size()
    # 计算 query、key 和 value 的状态
    # 使用线性变换将输入编码为 query、key 和 value,并将其调整为 [bsz, nh, q_len, hd] 的形状,便于后续计算
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    
    # 计算 key-value 序列长度,用于 Rotary Position Embedding
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    # 如果存在 past_key_value,将其拼接到 key_states 和 value_states 中, past_key_value 是用于序列生成任务的先前计算过的 key 和 value 状态, 在生成新的 token 时,需要将其与当前计算的 key 和 value 拼接起来
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    # 如果需要使用缓存,则将 key_states 和 value_states 作为 past_key_value 返回, 在序列生成任务中,为了加速计算,我们可以缓存已经计算过的 key 和 value,避免重复计算
    past_key_value = (key_states, value_states) if use_cache else None

    # 广播: 如果 num_key_value_heads 小于 num_heads,重复 key 和 value 以匹配 num_heads
    # 在某些情况下,模型的 key 和 value 头数量可能小于 query 头数量,为了保持一致性,需要重复 key 和 value
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
        # 将 query、key 和 value 拼接成 qkv tensor,并调整形状
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # 我们在 LlamaModel 中禁用了 _prepare_decoder_attention_mask 函数
    # the attention_mask should be the same as the key_padding_mask
    # 因此注意力掩码应该与 key_padding_mask 相同

    key_padding_mask = attention_mask
    nheads = qkv.shape[-2]
    # 将 qkv 张量从 [bsz, q_len, 3, nh, hd] 的形状重排为 [bsz, q_len, (3 * nh * hd)]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    # 调用 unpad_input 函数,去除输入张量中的填充部分, 输入为 x 和 key_padding_mask
        # 返回值包括: x_unpad: 去除填充后的张量, indices: 用于重新填充的索引张量, cu_q_lens: 每个序列的实际长度, max_s: 去填充后的最大序列长度
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    # 将 x_unpad 从 [nnz, (3 * nh * hd)] 的形状重排为 [nnz, 3, nh, hd], 其中 nnz 表示去填充后的非零元素数量
    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
    )
    # 调用 flash_attn_varlen_qkvpacked_func 函数,执行 Flash Attention 计算, 
    # 输入包括: x_unpad: 重排后的输入张量, cu_q_lens: 每个序列的实际长度, max_s: 去填充后的最大序列长度, 0.0: 用于初始化 dropout 的值, softmax_scale: softmax 缩放因子,设置为 None 表示不使用缩放
        # causal=True: 表示使用因果注意力机制
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )
    # 将 output_unpad 从 [nnz, nh, hd] 的形状重排为 [nnz, (nh * hd)], 然后调用 pad_input 函数,将其填充回原始形状 [bsz, q_len, (nh * hd)]
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    # 将 output 从 [bsz, q_len, (nh * hd)] 的形状调整为 [bsz, q_len, nh, hd]
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)
    # 调用 self.o_proj 函数,将 output 从 [bsz, q_len, nh, hd] 的形状映射回 [bsz, q_len, hidden_dim], 同时返回 None 作为注意力权重,以及 past_key_value
    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


def forward_noflashattn(
    self,
    hidden_states: torch.Tensor,   # 输入的隐藏状态张量
    attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量
    position_ids: Optional[torch.LongTensor] = None,  # 可选的位置 ID 张量
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 可选的先前计算过的 key 和 value 状态
    output_attentions: bool = False,  # 是否输出注意力权重,默认为 False
    use_cache: bool = False,   # 是否使用缓存,默认为 False
    padding_mask: Optional[torch.LongTensor] = None,  # 可选的填充掩码张量
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # 获取输入张量的 batch size、序列长度和隐藏层维度
    bsz, q_len, _ = hidden_states.size()
    # 计算 group_size,即每个组中包含的 token 数量,通过乘以 group_size_ratio 获得
    group_size = int(q_len * group_size_ratio)  
    # 检查序列长度是否能够被 group_size 整除,如果不能,抛出错误
    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
    num_group = q_len // group_size  # 计算总共有多少个组
    # 如果模型配置中 pretraining_tp 大于 1,即使用了张量并行,则需要对投影矩阵进行切分,并分别计算 query、key 和 value
    if self.config.pretraining_tp > 1:
        # 计算投影矩阵的切分大小
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        # 将 query 投影矩阵按列切分成 pretraining_tp 份
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        # 将 key 投影矩阵按列切分成 pretraining_tp 份
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        # 将 value 投影矩阵按列切分成 pretraining_tp 份
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
        # 分别计算 query、key 和 value,并在最后一个维度上拼接
        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:   # 如果不使用张量并行,直接计算 query、key 和 value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
    # 将 query、key 和 value 调整为 [bsz, nh, q_len, hd] 的形状,方便后续计算
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # 计算 key-value 序列长度,用于 Rotary Position Embedding
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # 如果存在 past_key_value,将其拼接到 key_states 和 value_states 中
    # past_key_value 是用于序列生成任务的先前计算过的 key 和 value 状态
    # 在生成新的 token 时,需要将其与当前计算的 key 和 value 拼接起来
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    # 如果需要使用缓存,则将 key_states 和 value_states 作为 past_key_value 返回
    # 在序列生成任务中,为了加速计算,我们可以缓存已经计算过的 key 和 value,避免重复计算
    past_key_value = (key_states, value_states) if use_cache else None

    # 广播: 如果 num_key_value_heads 小于 num_heads,重复 key 和 value 以匹配 num_heads
    # 在某些情况下,模型的 key 和 value 头数量可能小于 query 头数量,为了保持一致性,需要重复 key 和 value
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # shift 操作: 对 query、key 和 value 进行位移,实现 shift 2 attention
    def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
        # 对后一半的 query、key 和 value 进行位移
        # 这一步将 qkv 张量在第三个维度(即序列维度)上进行滚动。具体来说,对于每个 batch 和每个注意力头,后一半的序列被向左移动 group_size // 2 个位置。这种位移操作可以让每个 token 的注意力范围扩大到周围的一些 token,从而引入局部感受野。
        qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
        # 将 qkv 从 [bsz, nh, q_len, hd] 的形状重排为 [bsz * (q_len // group_size), group_size, nh, hd]
            # 首先,将 qkv 张量在第一和第二个维度上进行转置,变为 [bsz, q_len, nh, hd] 的形状。
            # 然后,将序列维度和 batch 维度合并,同时将序列长度除以 group_size,得到 [bsz * (q_len // group_size), group_size, nh, hd] 的形状。这种形状调整是为了方便在每个组内进行注意力计算。
            # 最后,再将第一和第二个维度进行转置,变为 [bsz * (q_len // group_size), nh, group_size, hd] 的形状,以便后续的注意力计算。
        qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
        return qkv
    # 分别对 query、key 和 value 进行 shift 操作
    query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    # 计算注意力权重
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    # 检查注意力权重的形状是否正确
    if attn_weights.size() != (bsz * num_group, self.num_heads, group_size, group_size):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_group, self.num_heads, group_size, group_size)}, but is"
            f" {attn_weights.size()}"
        )
    # 对注意力掩码进行扩展,并应用到注意力权重上
    # 注意力权重张量 attn_weights 的形状为 [bsz * num_group, self.num_heads, group_size, group_size]。通过扩展后的注意力掩码张量形状为 [bsz * num_group, 1, group_size, group_size],我们可以将其与注意力权重张量相加,从而实现对注意力权重的掩码操作。
        # 对原始的注意力掩码张量进行切片操作,只保留前 group_size 行和 group_size 列。
        # 这一步是对切片后的注意力掩码张量进行重复操作,将其在第一个维度上重复 num_group 次。num_group 表示将整个序列分成了多少个组,其值为 q_len // group_size。
    attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)
    if attention_mask is not None:
        if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
            raise ValueError(
                f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32, 对注意力权重应用 softmax 函数,并将其转换为 float32 精度
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # 计算注意力输出
    attn_output = torch.matmul(attn_weights, value_states)
    # 检查注意力输出的形状是否正确
    if attn_output.size() != (bsz * num_group, self.num_heads, group_size, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    # 将注意力输出从 [bsz * num_group, nh, group_size, hd] 的形状重排为 [bsz, q_len, nh, hd]
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    # shift back, 对注意力输出进行反向 shift 操作
    attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)
    # 将注意力输出从 [bsz, q_len, nh, hd] 的形状重排为 [bsz, q_len, hidden_size]
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    # 如果使用了张量并行,对输出进行切分和线性变换
    if self.config.pretraining_tp > 1:
        # 将输出在最后一个维度上切分成 pretraining_tp 份
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        # 将输出线性变换矩阵按行切分成 pretraining_tp 份
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        # 对每一份输出分别进行线性变换,然后将结果相加
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:  # 如果不使用张量并行,直接进行线性变换
        attn_output = self.o_proj(attn_output)
    
    if not output_attentions: # 如果不需要输出注意力权重,将其设置为 None
        attn_weights = None
    # 返回注意力输出、注意力权重和 past_key_value
    return attn_output, attn_weights, past_key_value

# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask

def apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids):
    """
    技术: Rotary Position Embedding(旋转位置嵌入)
    解决问题: 在 Transformer 中引入相对位置信息,提高模型对序列的建模能力。
    参数:
    q (torch.Tensor): Query 张量,形状为 [bsz, num_heads, seq_len, head_dim]
    k (torch.Tensor): Key 张量,形状为 [bsz, num_heads, seq_len, head_dim]
    cos_sin (tuple): 包含预计算的余弦值和正弦值,用于旋转位置嵌入计算
    position_ids (torch.Tensor): 位置 ID 张量,形状为 [bsz, seq_len]
    返回值:
    q (torch.Tensor): 经过旋转位置嵌入加性运算后的 Query 张量
    k (torch.Tensor): 经过旋转位置嵌入加性运算后的 Key 张量
    """
    # 从 position_ids 中获取索引,扩展维度以匹配 cos 和 sin 的形状
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    # 获取当前 batch size
    bsz = gather_indices.shape[0]
    # 从预计算的 cos 和 sin 中提取当前位置的值, 使用 gather 操作从预计算的张量中获取当前位置的值, 对 cos 和 sin 张量进行转置和重复,以匹配 gather_indices 的形状
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    # 将 Query 和 Key 张量与旋转位置嵌入进行加性运算
    # 对 Query 和 Key 张量进行旋转位置嵌入的加性运算
        # 使用辅助函数 rotate_half 进行半转旋转
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k


def forward_flashattn_inference(
    self,
    hidden_states: torch.Tensor,  # 输入的隐藏状态张量,形状为 [batch_size, sequence_length, hidden_dim]
    attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量,形状为 [batch_size, sequence_length],用于指定哪些位置应该被忽略
    position_ids: Optional[torch.Tensor] = None,  # 可选的位置 ID 张量,形状为 [batch_size, sequence_length],用于计算旋转位置嵌入
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 可选的先前计算过的 key 和 value 状态,用于序列生成任务,每个张量的形状为 [batch_size, num_heads, past_sequence_length, head_dim]
    output_attentions: bool = False,  # 是否输出注意力权重,默认为 False
    use_cache: bool = False,  # 是否使用缓存,默认为 False,如果为 True,则将当前计算过的 key 和 value 状态作为 past_key_value 返回
    padding_mask: Optional[torch.Tensor] = None,  # 可选的填充掩码张量,与 attention_mask 相同,形状为 [batch_size, sequence_length]
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    解决问题: 在推理(inference)过程中高效地计算注意力层的输出,同时引入相对位置信息。
    
    返回值:
    output (torch.Tensor): 注意力层的输出张量,形状为 [bsz, seq_len, hidden_dim]
    attentions (Optional[torch.Tensor]): 始终为 None,因为不支持输出注意力权重
    past_key_value (Optional[Tuple[torch.Tensor]]): 如果使用缓存,则返回当前计算过的 key 和 value 状态
    """
    # 如果需要输出注意力权重,发出警告,因为这里不支持输出注意力权重
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )
    bsz, q_len, _ = hidden_states.size()
    # 获取 key 和 value 头的数量,默认与 query 头数量相同
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)
    # 计算 query、key 和 value 的状态
    # 使用线性变换将输入编码为 query、key 和 value,并将其调整为 [batch_size, sequence_length, num_heads, head_dim] 的形状,便于后续计算
    # 对于 query,num_heads 为 self.num_heads;对于 key 和 value,num_heads 为 kv_heads
    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)   # [batch_size, sequence_length, num_heads, head_dim]
        for op, nh in (  
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    ) # shape: (b, s, num_heads, head_dim)

    # 计算 key-value 序列长度,用于 Rotary Position Embedding
    kv_seq_len = k.shape[1]  # key 的序列长度
    past_kv_len = 0
    # 如果存在 past_key_value,将其长度加到 kv_seq_len 上
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[2]  # past_key_value 中 key 的序列长度
        kv_seq_len += past_kv_len  # 总的 key-value 序列长度
    # 计算旋转位置嵌入的余弦和正弦值
    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    # 将 query 和 key 与旋转位置嵌入进行加性运算
    q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)
    # Past Key value support
        # 如果存在 past_key_value,将其拼接到 key 和 value 中
    if past_key_value is not None:
        # 确保 flash-attn 版本大于等于 2.1.0,以支持 past_key_value
        assert (
            flash_attn_version >= "2.1.0"
        ), "past_key_value support requires flash-attn >= 2.1.0"
        # 拼接 past_key_value 到 key 和 value 中,同时进行转置操作
        # past_key_value[0] 的形状: [batch_size, num_heads, past_sequence_length, head_dim]
        # k 的形状: [batch_size, sequence_length, num_heads, head_dim]
        # 拼接后的形状: [batch_size, num_heads, past_sequence_length + sequence_length, head_dim]
        k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
        v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)
    # 如果需要使用缓存,则将 key 和 value 作为 past_key_value 返回, 进行转置操作,使形状变为 [batch_size, num_heads, sequence_length, head_dim]
    past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

    # 计算注意力层的输出
    if attention_mask is None:
        # 如果没有注意力掩码,直接调用 flash_attn_func 函数计算注意力层输出
        # q、k、v 的形状: [batch_size, sequence_length, num_heads, head_dim]
        # output 的形状: [batch_size, sequence_length, num_heads * head_dim]
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len, -1
        )
    else:
        # 如果有注意力掩码,需要进行填充和去填充操作
        # 对 query 进行去填充操作, q 的形状: [batch_size, sequence_length, num_heads, head_dim]
        # attention_mask[:, -q_len:] 的形状: [batch_size, sequence_length]
        q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask[:, -q_len:])
        # We can skip concat and call unpad twice but seems better to call unpad only once.
            # q_unpad 的形状: [nnz, num_heads, head_dim]，其中 nnz 表示去填充后的非零元素数量
            # indices 是用于重新填充的索引张量
            # cu_q_lens 是每个序列的实际长度
            # max_s 是去填充后的最大序列长度

            # 对 key 和 value 进行去填充操作,并将它们堆叠在一起
            # k 和 v 的形状: [batch_size, sequence_length, num_heads, head_dim]
            # kv 的形状: [batch_size, sequence_length, num_heads * 2, head_dim]
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), attention_mask
        )
            # kv_unpad 的形状: [nnz, num_heads * 2, head_dim]
            # cu_k_lens 是每个序列的实际长度
            # max_k 是去填充后的最大序列长度

            # 调用 flash_attn_varlen_kvpacked_func 函数计算注意力层输出
            # q_unpad 的形状: [nnz, num_heads, head_dim]
            # kv_unpad 的形状: [nnz, num_heads * 2, head_dim]
            # output_unpad 的形状: [nnz, num_heads, head_dim]
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        # 调整输出张量的形状
        # output_unpad 的形状: [nnz, num_heads * head_dim]
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        # 将输出张量填充回原始形状
        # output 的形状: [batch_size, sequence_length, num_heads * head_dim]
        output = pad_input(output_unpad, indices, bsz, q_len)
    # 使用输出映射将注意力层输出映射回原始隐藏维度
    # output 的形状: [batch_size, sequence_length, hidden_dim]
    return self.o_proj(output), None, past_key_value

# 该函数用于在推理(inference)过程中准备解码器的注意力掩码, 注意力掩码用于指定序列中哪些位置应该被忽略,不参与注意力计算, 在解码器端,注意力掩码确保模型只关注当前位置之前的输出,以实现自回归(auto-regressive)特性
def _prepare_decoder_attention_mask_inference(
    self, 
    attention_mask, # 输入的注意力掩码张量,形状为 [batch_size, sequence_length],用于指定哪些位置应该被忽略
    input_shape, # 输入张量的形状,通常为 [batch_size, sequence_length, hidden_dim]
    inputs_embeds, # 输入的嵌入张量,形状为 [batch_size, sequence_length, embedding_dim]
    past_key_values_length # 过去的 key 和 value 状态的长度,用于序列生成任务
):
    # [bsz, seq_len]
    # 如果过去的 key 和 value 状态的长度大于 0,且提供了注意力掩码
    if past_key_values_length > 0 and attention_mask is not None:
        # 将过去的注意力掩码(全为 True,表示不进行掩码)与当前注意力掩码沿着序列维度拼接
        # 这样可以确保模型在生成新的令牌时,不会关注未来的令牌,同时也不会忽略过去已生成的令牌
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length), # 创建一个全 True 的张量,形状为 [batch_size, past_key_values_length]
                    True,
                    dtype=attention_mask.dtype,  # 与原注意力掩码数据类型相同
                    device=attention_mask.device,  # 与原注意力掩码设备相同
                ),
                attention_mask,  # 当前注意力掩码
            ),
            dim=-1,  # 沿着最后一个维度(序列维度)进行拼接
        )
    # 如果提供了注意力掩码,且所有位置都为 True(即没有进行掩码)
    if attention_mask is not None and torch.all(attention_mask):
        # 则返回 None,这样可以使用更快的计算方式,不需要进行掩码操作
        return None  # This uses the faster call when training with full samples
    # 否则返回准备好的注意力掩码
    return attention_mask

def replace_llama_attn(use_flash_attn=True, use_full=False, inference=False):
    if use_flash_attn:
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        if cuda_major < 8:
            warnings.warn(
                "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
                "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            )
        if inference:
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_inference
        else:
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
                _prepare_decoder_attention_mask
            )
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_full if use_full else forward_flashattn
    else:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_noflashattn
