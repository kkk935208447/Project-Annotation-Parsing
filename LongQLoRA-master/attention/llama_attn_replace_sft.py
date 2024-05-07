# Modified based on https://github.com/lm-sys/FastChat

import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import transformers
from einops import rearrange  # 1. einops 是一个用于高效操作多维数据(如 Tensor)的库,rearrange 函数可以方便地调整 Tensor 的维度和形状。
from flash_attn import __version__ as flash_attn_version  # 2. 导入 flash_attn 库的版本号。

# 3. 从 flash_attn.bert_padding 模块导入 pad_input 和 unpad_input 函数。
    # pad_input 用于在输入数据(通常是 Tensor)的末尾添加填充元素,使其长度达到指定值。
    # unpad_input 则用于移除输入数据末尾的填充元素。
    # 在处理变长序列时,这两个函数可以确保输入数据具有相同长度,以满足模型的输入要求。
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,         # flash_attn_func: 是 Flash Attention 的核心函数,用于计算注意力权重和加权和。
    flash_attn_varlen_kvpacked_func,     # flash_attn_varlen_kvpacked_func: 用于处理变长 key-value 对的 Flash Attention 计算。
    flash_attn_varlen_qkvpacked_func     # flash_attn_varlen_qkvpacked_func: 用于处理变长 query-key-value 的 Flash Attention 计算。
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,       # apply_rotary_pos_emb: 应用旋转位置编码(Rotary Position Embedding, RPE)。RPE 是一种位置编码方式,可以更好地捕获序列的位置信息。
    repeat_kv,                  # repeat_kv: 重复 key 和 value,使其维度与 query 相匹配。
    rotate_half                 # rotate_half: 对 value 的一半维度应用旋转操作,这是 RPE 的一部分。
)

# TODO 注释掉了官方代码, 疑似重复导入
# from flash_attn.bert_padding import unpad_input, pad_input
import math


group_size_ratio = 1/4   # s2attention group 大小比例
sft_group_size = 8192    # s2attention + Flash Attention 大小

# 9. 定义 forward_flashattn 函数,用于实现使用s2attention + Flash Attention 的前向传播过程。
# 调用 unpad_input 函数去除输入数据中的填充元素,获取真实序列长度和索引信息。。
# 重新排列去填充后的输入张量 x_unpad 的形状,以适应 flash_attn_varlen_qkvpacked_func 函数的输入格式。
# 调用 flash_attn_varlen_qkvpacked_func 函数计算注意力输出,该函数采用 CUDA 加速,可以极大提高计算效率。
# 调用 pad_input 函数将填充元素插回到注意力输出中,并对输出张量进行最后一步形状调整。
def forward_flashattn(
    self,
    hidden_states: torch.Tensor,    # hidden_states: 输入的隐藏状态张量,形状为 (batch_size, sequence_length, hidden_size)。
    attention_mask: Optional[torch.Tensor] = None,  # attention_mask: 注意力掩码,用于指定哪些位置应该被忽略,形状为 (batch_size, query_length)。
    position_ids: Optional[torch.Tensor] = None,    # position_ids: 位置编码的 ID,用于计算位置编码,形状为 (batch_size, sequence_length)。
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # past_key_value: 用于序列生成任务的先前计算过的 key 和 value 状态。
    output_attentions: bool = False,    # output_attentions: 一个布尔值,指示是否输出注意力权重。
    use_cache: bool = False,     # use_cache: 一个布尔值,指示是否使用缓存来加速序列生成。
    padding_mask: Optional[torch.LongTensor] = None,   # padding_mask: 填充掩码,用于指定序列中的填充位置。
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]

    # 该函数返回一个元组,包含 (output, attentions, new_cache)。
    # output 是注意力计算的输出张量,形状为 (batch_size, sequence_length, hidden_size)。
    # attentions 是注意力权重,在这里总是返回 None。
    # past_key_value 是用于序列生成任务的新计算过的 key 和 value 状态。
    """

    # 10. 如果不是训练模式,抛出异常,因为该函数只适用于训练. s2atten 训练时使用分组, 推理的时候用的standard attention, 具体看相关论文 
    if not self.training:
        raise ValueError("This function is only for training. For inference, please use forward_flashattn_inference.")

    # 11. 如果需要输出注意力权重,发出警告,因为这里不支持输出注意力权重。
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size() # 12. 获取输入张量的 batch size、序列长度和隐藏层维度。
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
    # shift 2 attention 
    # 19. 根据 q_len 计算 group_size,用于 Flash Attention。group_size 决定了每个 group 的长度,是一个重要的超参数,影响计算效率和内存使用。
    if q_len % 4096 == 0:
        group_size = int(q_len * group_size_ratio)
    else:
        group_size = sft_group_size

    # 进行 shift 2 attention, [b, t, 3, 2, nh//2, d] -> [b, 2, t, 3, nh//2, d] -> [b * 2, t, 3, nh//2, d]
    qkv = qkv.reshape(bsz, q_len, 3, 2, self.num_heads // 2, self.head_dim).permute(0, 3, 1, 2, 4, 5).reshape(bsz * 2,
                                                                                                              q_len, 3,
                                                                                                              self.num_heads // 2,
                                                                                                              self.head_dim)
    # 21. 使用 Flash Attention 函数计算注意力输出。
    # 这里使用的是 flash_attn_varlen_qkvpacked_func,适用于处理变长的 query、key 和 value 序列。
        # 将 qkv 张量重新排列形状,以适应 Flash Attention 函数的输入格式。新的形状为 [batch_size, sequence_length, 3 * num_heads * head_dim]。
    x = rearrange(qkv, "b s three h d -> b s (three h d)") 
    # 调用 unpad_input 函数,去除输入数据中的填充元素。x_unpad 是去除填充后的数据。indices 是一个索引张量,用于在后续恢复填充时定位原始数据。
        # cu_q_lens 是每个序列的真实长度。max_s 是最大序列长度。
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        # 计算每个 group 的起始位置。从 0 开始,步长为 group_size,生成一个等差数列。
    cu_q_len_tmp = torch.arange(0, max_s, group_size, device=key_padding_mask.device, dtype=cu_q_lens.dtype)
        # 计算每个 group 的中点位置,用于后续计算。
    cu_q_len_tmp2 = cu_q_len_tmp + group_size // 2  
        # 如果中点位置超出最大序列长度,将其设置为最小值。
    cu_q_len_tmp2[cu_q_len_tmp2 >= max_s] = torch.iinfo(cu_q_len_tmp2.dtype).min
        # 将起始位置和中点位置堆叠,并为每个批次重复,再加上每个序列的真实长度,这一步是为了处理变长序列的情况。
    cu_q_len_tmp = torch.stack([cu_q_len_tmp, cu_q_len_tmp2]).repeat(bsz, 1) + cu_q_lens[:-1].unsqueeze(-1)
        # 将上一步计算的结果和真实序列长度拼接,并展平为一维张量。
    cu_q_lens = torch.cat([cu_q_len_tmp, cu_q_lens[1:].unsqueeze(-1)], dim=-1).view(-1)
    cu_q_lens = cu_q_lens[cu_q_lens >= 0] # 去除无效的负数长度。

        # 重新排列 x_unpad 的形状,以适应 Flash Attention 函数的输入格式。新的形状为 [总有效长度, 3, num_heads // 2, head_dim]。
    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads // 2
    )

    # 调用 flash_attn_varlen_qkvpacked_func 函数计算注意力输出。
    # 参数说明:
    # x_unpad: 输入的 query、key 和 value 张量,形状为 [总有效长度, 3, num_heads // 2, head_dim]。
    # cu_q_lens: 每个序列的有效长度,一维张量。
    # group_size: Flash Attention 的 group 大小。
    # 0.0: dropout 比例,这里设置为 0。
    # softmax_scale: softmax 的缩放因子,这里设置为 None,表示使用默认值。
    # causal=True: 表示使用因果注意力(causial attention),适用于语言模型等序列生成任务。
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, group_size, 0.0, softmax_scale=None, causal=True
    )
    # 首先,调用 rearrange 函数将 output_unpad 的形状调整为 [总有效长度, num_heads // 2 * head_dim]。
    # 然后,调用 pad_input 函数,根据之前的 indices 索引,将填充元素插回到注意力输出中。
    # 最后,再次调用 rearrange 函数,将注意力输出的形状调整为 [batch_size, sequence_length, num_heads // 2, head_dim]。
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz * 2, q_len
        ),
        "b s (h d) -> b s h d",
        h=nheads // 2,
    )
    # 对注意力输出进行最后一步重塑,将其形状调整为 [batch_size, sequence_length, num_heads, head_dim]。
    output = output.reshape(bsz, 2, q_len, nheads // 2, self.head_dim).transpose(1, 2).reshape(bsz, q_len, nheads,
                                                                                               self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


# 这段代码实现了使用 Flash Attention 技术进行高效的注意力计算,主要包括以下几个方面:
# 导入所需的张量,如隐藏状态、注意力掩码、位置编码等,并进行相应的形状调整。
# 计算 query、key 和 value 的状态,并应用 Rotary Position Embedding (RPE)。RPE 是一种位置编码方式,可以更好地捕获序列的位置信息。
# 处理用于序列生成任务的 past_key_value,将其与当前计算的 key 和 value 拼接。
# 如果 key 和 value 头数量小于 query 头数量,重复 key 和 value 以匹配头数。
# 将 query、key 和 value 拼接成一个张量,并调整形状以适应 Flash Attention 函数的输入格式。
# 调用 unpad_input 函数去除输入数据中的填充元素,获取真实序列长度和索引信息。
# 调用 flash_attn_varlen_qkvpacked_func 函数计算注意力输出,该函数采用 CUDA 加速,可以极大提高计算效率,特别适用于处理变长序列和大规模数据。
# 调用 pad_input 函数将填充元素插回到注意力输出中,并对输出张量进行最后一步形状调整。
# 使用线性变换将注意力输出映射回原始维度,并返回注意力输出、注意力权重 (None) 和 past_key_value。
def forward_flashattn_full(
    self,
    hidden_states: torch.Tensor,         # 输入的隐藏状态张量,形状为 (batch_size, sequence_length, hidden_size)
    attention_mask: Optional[torch.Tensor] = None,   # 注意力掩码,用于指定哪些位置应该被忽略,形状为 (batch_size, query_length)
    position_ids: Optional[torch.Tensor] = None,     # 位置编码的 ID,用于计算位置编码,形状为 (batch_size, sequence_length)
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 用于序列生成任务的先前计算过的 key 和 value 状态
    output_attentions: bool = False,    # 一个布尔值,指示是否输出注意力权重
    use_cache: bool = False,            # 一个布尔值,指示是否使用缓存来加速序列生成
    padding_mask: Optional[torch.LongTensor] = None,    # 填充掩码,用于指定序列中的填充位置
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    # 如果需要输出注意力权重,发出警告,因为这里不支持输出注意力权重
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size() # 获得 B,T

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

    # 计算 key-value 序列长度,用于 Rotary Position Embedding
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    # 如果存在 past_key_value,将其拼接到 key_states 和 value_states 中, past_key_value 是用于序列生成任务的先前计算过的 key 和 value 状态,在生成新的 token 时,需要将其与当前计算的 key 和 value 拼接起来
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    # 如果需要使用缓存,则将 key_states 和 value_states 作为 past_key_value 返回, 在序列生成任务中,为了加速计算,我们可以缓存已经计算过的 key 和 value,避免重复计算
    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    # 广播: 如果 num_key_value_heads 小于 num_heads,重复 key 和 value 以匹配 num_heads, 在某些情况下,模型的 key 和 value 头数量可能小于 query 头数量,为了保持一致性,需要重复 key 和 value
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
    # the attention_mask should be the same as the key_padding_mask
    # 由于禁用了 _prepare_decoder_attention_mask 函数,因此注意力掩码与 key_padding_mask 相同
    key_padding_mask = attention_mask
    nheads = qkv.shape[-2]

    # 重塑 qkv 的形状,[bsz, q_len, 3, num_heads, head_dim] -> [bsz, q_len, 3 * num_heads * head_dim]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    # 调用 unpad_input 函数,去除输入数据中的填充元素,获取真实序列长度和索引信息
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    # 重新排列 x_unpad 的形状, [nnz, 3 * num_heads * head_dim] -> [nnz, 3, num_heads, head_dim]
    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
    )

    # 调用 flash_attn_varlen_qkvpacked_func 函数计算注意力输出,该函数采用 CUDA 加速,可以极大提高计算效率
    # 参数说明:
    # x_unpad: 输入的 query、key 和 value 张量,形状为 [总有效长度, 3, num_heads, head_dim]
    # cu_q_lens: 每个序列的有效长度,一维张量
    # max_s: 最大序列长度
    # 0.0: dropout 比例,这里设置为 0
    # softmax_scale: softmax 的缩放因子,这里设置为 None,表示使用默认值
    # causal=True: 表示使用因果注意力(causial attention),适用于语言模型等序列生成任务
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )

    # 调用 pad_input 函数,根据之前的 indices 索引,将填充元素插回到注意力输出中
    # 然后,再次调用 rearrange 函数,将注意力输出的形状调整为 [batch_size, sequence_length, num_heads, head_dim]
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    # 使用线性变换将输出映射回原始维度,并返回注意力输出、注意力权重 (None) 和 past_key_value
    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value

# s2attention 不使用 Flash Attention
def forward_noflashattn(
    self,
    hidden_states: torch.Tensor,   # 输入的隐藏状态张量,形状为 (batch_size, sequence_length, hidden_size)
    attention_mask: Optional[torch.Tensor] = None,   # 注意力掩码,用于指定哪些位置应该被忽略,形状为 (batch_size, query_length)
    position_ids: Optional[torch.LongTensor] = None,  # 位置编码的 ID,用于计算位置编码,形状为 (batch_size, sequence_length)
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 用于序列生成任务的先前计算过的 key 和 value 状态
    output_attentions: bool = False,  # 一个布尔值,指示是否输出注意力权重
    use_cache: bool = False,   # 一个布尔值,指示是否使用缓存来加速序列生成
    padding_mask: Optional[torch.LongTensor] = None,  # 填充掩码,用于指定序列中的填充位置
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size() # 获得 B,T

    group_size = int(q_len * group_size_ratio)  # group_size_ratio 分组后需要被整除, group_size 是 s2attention 分组大小

    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
    num_group = q_len // group_size  # s2attention 分组数

    # 如果使用了张量并行训练 (pretraining_tp > 1),则需要对权重矩阵进行切分
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
        
        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else: # 否则,使用普通计算
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    # [b, s, n_head, d_head] -> [b, n_head, s, d_head]
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
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # 如果需要使用缓存,则将 key_states 和 value_states 作为 past_key_value 返回
    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    # 如果 num_key_value_heads 小于 num_heads,重复 key 和 value 以匹配 num_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # 定义 shift 函数,用于在计算注意力时对 query、key 和 value 进行位移操作
    def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
        # 这一行代码对 qkv 张量的后半部分 (num_heads // 2 到最后) 进行了位移操作。
        # roll(-group_size // 2, dims=2) 表示沿着第三个维度 (即 seq_len 维度) 进行循环位移,向左移动 group_size // 2 个位置。
        # 例如,如果 seq_len=16, group_size=4, 那么每个 head 中的序列就会被分成 4 组,每组长度为 4。对于后半部分的 head,它们的序列将向左移动 2 个位置。这种操作可以让每个 head 关注不同的序列区域,从而实现局部注意力 (local attention)。
        qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
        # 首先,qkv.transpose(1, 2) 将 qkv 的形状变换为 [bsz, seq_len, num_heads, head_dim]。
        # 然后,reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim) 将 qkv 重新排列为 [bsz * (q_len // group_size), group_size, num_heads, head_dim] 的形状。
        # 这里,bsz * (q_len // group_size) 表示将 batch 维度和分组后的序列维度合并成一个新的维度。group_size 表示每个组的长度。之所以这样做,是为了方便在每个组内进行注意力计算。
        # 最后,transpose(1, 2) 将形状变换为 [bsz * (q_len // group_size), num_heads, group_size, head_dim]
        qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
        return qkv

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

    # 应用注意力掩码
    attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)
    if attention_mask is not None:
        if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
            raise ValueError(
                f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    # 对注意力权重应用 softmax 函数,得到归一化的注意力分数, 注意力权重先被提升到 fp32 精度,计算完成后再转换回原始数据类型
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # 计算注意力输出
    if attn_output.size() != (bsz * num_group, self.num_heads, group_size, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    
    # 检查注意力输出的形状是否正确
    attn_output = attn_output.transpose(1, 2).contiguous()
    # 调整注意力输出的形状
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    # shift back, 对注意力输出进行反向位移操作,与之前的 shift 操作相反
    attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)
    # 将注意力输出的形状调整为 (batch_size, sequence_length, hidden_size)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    # 如果使用了张量并行训练,则需要对注意力输出进行拆分和线性变换
    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else: # 如果不使用张量并行训练,则直接应用线性变换
        attn_output = self.o_proj(attn_output)
    # 如果不需要输出注意力权重,则将其设置为 None
    if not output_attentions:
        attn_weights = None
    # 返回注意力输出、注意力权重和 past_key_value
    return attn_output, attn_weights, past_key_value

# 这里禁用了 the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask

# 这个函数的作用是在推理(inference)阶段应用 Rotary Position Embedding
# RoPE 是一种位置编码方式,可以更好地捕获序列的位置信息,提高模型的性能
def apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids):
    # 从 position_ids 中获取索引,用于从 cos_sin 张量中收集相应的 cos 和 sin 值
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    # 重复 gather_indices,使其形状与 cos_sin 张量相匹配
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    bsz = gather_indices.shape[0]  # 获取 batch size
    # 从 cos_sin 张量中收集相应的 cos 和 sin 值
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    # 应用 RoPE 到 query 和 key 向量
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k


def forward_flashattn_inference(
    self,
    hidden_states: torch.Tensor,   # 输入的隐藏状态张量,形状为 (batch_size, sequence_length, hidden_size)
    attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码,用于指定哪些位置应该被忽略,形状为 (batch_size, query_length)
    position_ids: Optional[torch.Tensor] = None,    # 位置编码的 ID,用于计算位置编码,形状为 (batch_size, sequence_length)
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 用于序列生成任务的先前计算过的 key 和 value 状态
    output_attentions: bool = False,   # 一个布尔值,指示是否输出注意力权重
    use_cache: bool = False,    # 一个布尔值,指示是否使用缓存来加速序列生成
    padding_mask: Optional[torch.Tensor] = None,   # 填充掩码,用于指定序列中的填充位置
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # 如果需要输出注意力权重,发出警告,因为这里不支持输出注意力权重
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()  # B,T
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)  # 获取 key-value 头数量,如果没有指定,则默认与 query 头数量相同

    q, k, v = (    #  shape: (b, s, num_heads, head_dim)
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )  # shape: (b, s, num_heads, head_dim)

    # 计算 key-value 序列长度,用于 Rotary Position Embedding
    kv_seq_len = k.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[2]
        kv_seq_len += past_kv_len

    # 计算 Rotary Position Embedding
    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

    # 如果存在 past_key_value,将其拼接到 key 和 value 张量中
    if past_key_value is not None:
        assert (flash_attn_version >= "2.1.0"), "past_key_value support requires flash-attn >= 2.1.0"
        # reuse k, v
        k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
        v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)
    # 如果需要使用缓存,则将 key 和 value 作为 past_key_value 返回
    past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

    if attention_mask is None:
        # 如果没有提供注意力掩码,直接调用 flash_attn_func 计算注意力输出
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len, -1
        )
    else:
        # 如果提供了注意力掩码,需要进行额外的处理
        q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask[:, -q_len:])
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(torch.stack((k, v), dim=2), attention_mask)

        # 调用 flash_attn_varlen_kvpacked_func 计算注意力输出
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
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)
    # 使用线性变换将注意力输出映射回原始维度,并返回注意力输出、注意力权重 (None) 和 past_key_value
    return self.o_proj(output), None, past_key_value

def _prepare_decoder_attention_mask_inference(
                                            self, 
                                            attention_mask,  # 输入的注意力掩码,形状为 [bsz, seq_len]
                                            input_shape,     # 输入张量的形状
                                            inputs_embeds,   # 输入的嵌入向量
                                            past_key_values_length  # 先前计算的 key 和 value 的长度
                                        ):
    # [bsz, seq_len], 如果存在先前计算的 key 和 value,并且提供了注意力掩码,则需要将先前的掩码与当前掩码拼接
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(   # 创建一个全为 True 的掩码张量,表示先前计算的部分应该被完全关注
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,  # 将先前的掩码和当前掩码在最后一维(序列长度维度)上拼接
            ),
            dim=-1,
        )
    # 如果所有位置都应该被关注,则返回 None,这样可以使用更快的计算路径
    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask  # 否则,返回准备好的注意力掩码

def replace_llama_attn(use_flash_attn=True, use_full=False, inference=False):
    if use_flash_attn:
        cuda_major, cuda_minor = torch.cuda.get_device_capability()   # 获取 GPU 的计算能力
        # 如果 GPU 计算能力低于 8.0,则发出警告,因为训练时 Flash Attention 只支持 A100 或 H100 GPU
        if cuda_major < 8:
            warnings.warn(
                "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
                "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            )
        if inference:  # 如果是用于推理(inference)
            # 替换 LlamaModel 中的 _prepare_decoder_attention_mask 方法
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
            # 替换 LlamaAttention 中的 forward 方法
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_inference
        else:  # 如果是用于训练
            # 替换 LlamaModel 中的 _prepare_decoder_attention_mask 方法
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (_prepare_decoder_attention_mask)
            # TODO 修改源码
            # transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_full if use_full else forward_flashattn
            
            if use_full:  # 如果选择使用完整版本的 Flash Attention
                # 替换 LlamaAttention 中的 forward 方法为 forward_flashattn_full
                transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_full
            # 否则,使用非完整版本的 Flash Attention
            else:
                # 替换 LlamaAttention 中的 forward 方法为 forward_flashattn
                transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn
    else: # 如果不使用 Flash Attention
        # 替换 LlamaAttention 中的 forward 方法为 forward_noflashattn
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_noflashattn
