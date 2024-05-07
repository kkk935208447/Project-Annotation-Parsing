# Modified based on https://github.com/lm-sys/FastChat
import warnings,sys
from loguru import logger
# 重定义终端logger显示颜色
logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} |<cyan><lvl>{level:8}</></>| {name} : {module}:{line:4} | <cyan>mymodule</> | - <lvl>{message}</>",
        "colorize": True
    },
])
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import transformers
# 1. einops 是一个用于高效操作多维数据(如 Tensor)的库,rearrange 函数可以方便地调整 Tensor 的维度和形状。
from einops import rearrange  
from flash_attn import __version__ as flash_attn_version
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,                  
    flash_attn_varlen_kvpacked_func,  
    flash_attn_varlen_qkvpacked_func  
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,       
    repeat_kv,                  
    rotate_half           
)
from transformers.cache_utils import Cache
import math


# s2atten 左shift 过程, 滚动 group size 的一半
# 详情见: https://arxiv.org/pdf/2309.12307 中 伪代码
def shift_1(qkv:torch.Tensor,  # query/key/value 张量, 维度为 [b,nh,q_len,hd]
          bsz:int,  # batch size
          q_len:int, # seq length
          group_size:int, # 分组的大小
          num_heads:int, 
          head_dim:int
        ) -> torch.Tensor :
    # 维度检测
    if qkv.size() != (bsz, num_heads, q_len, head_dim):
        raise ValueError(
            f"qkv weights should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
            f" {qkv.size()}"
        )
    # 对后一半的 query、key 和 value 进行位移, 将张量在第三个维度(即序列维度)上进行向左滚动。具体来说,对于每个 batch 和每个注意力头,后一半的序列被向左移动 group_size // 2 个位置。这种位移操作可以让每个 token 的注意力范围扩大到周围的一些 token,从而引入局部感受野。
    # qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)   # 不使用 这种in-place 的方式, 改用 cat chunk函数处理, 保证梯度流传播
    qkv = torch.cat(
        (qkv.chunk(2, dim=1)[0], qkv.chunk(2, dim=1)[1].roll(-group_size//2, dims=2))  # dims = 2 表示在序列维度上进行左滚动, -group_size//2 表示左滚动的步数
        , dim=1) # 维度2 进行 cat
    # 开始 qkv.transpose(1, 2): [bsz, nh, q_len, head_dim] ---> [bsz, q_len, nh, head_dim]
    # 中间 .reshape: [bsz, q_len, nh, head_dim] ---> [bsz * (q_len // group_size), group_size, num_heads, head_dim]
    # 最后 .transpose(1, 2): [bsz * (q_len // group_size), group_size, num_heads, head_dim] ---> [bsz * (q_len // group_size), num_heads, group_size, head_dim]
    qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
    return qkv


# 与 shift_1 只是输入qkv的维度不同
def shift_2(qkv:torch.Tensor,  # query/key/value 张量, 维度为 [b,q_len,nh ,hd]
          bsz:int,  # batch size
          q_len:int, # seq length
          group_size:int, # 分组的大小
          num_heads:int, 
          head_dim:int
        ) -> torch.Tensor :
    # 维度检测
    if qkv.size() != (bsz, q_len, num_heads, head_dim):
        raise ValueError(
            f"qkv weights should be of size {(bsz, q_len, num_heads, head_dim)}, but is"
            f" {qkv.size()}"
        )
    # qkv[:,:, num_heads // 2:] = qkv[:,:, num_heads // 2:].roll(-group_size // 2, dims=1)  # 不使用 这种in-place 的方式, 改用 cat chunk函数处理, 保证梯度流传播
    qkv = torch.cat(
        (qkv.chunk(2, dim=2)[0], qkv.chunk(2, dim=2)[1].roll(-group_size//2, dims=1))  # dims = 1 表示在序列维度上进行左滚动, -group_size//2 表示左滚动的步数
        , dim=2)  # 维度2 进行 cat
    # [b,q_len,nh ,hd] --> [b * (q_len // group_size), group_size, nh, hd]
    qkv = qkv.reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim)
    return qkv


group_size_ratio = 1/4  # 分组的比例, 组大小 T * group_size_ratio 
# s2attention 使用 flash attention
def llama_forward_s2flashattn(self,
                        hidden_states: torch.Tensor,    # B,T,D 
                        # flash attention mask 维度有两种情况：
                        # 第一种：attention_mask为2维：[batch_size, sequence_length], 这种情况出现在存在任何 pad的情况下
                        # 第二种：当所有seq都没有pad, attention会被置为None,交给flash attention 处理即可
                        attention_mask: Optional[torch.LongTensor] = None, 
                        position_ids: Optional[torch.LongTensor] = None,  # B,T
                        past_key_value: Optional[Cache] = None,  
                        output_attentions: bool = False,
                        use_cache: bool = False,
                        cache_position: Optional[torch.LongTensor] = None,  # B,T
                        **kwargs,
                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len] or None

    # 该函数返回一个元组,包含 (output, attentions, new_cache)。
    # output 是注意力计算的输出张量,形状为 (batch_size, sequence_length, hidden_size)。
    # attentions 是注意力权重,在这里总是返回 None。
    # past_key_value 是用于序列生成任务的新计算过的 key 和 value 状态。
    """

    output_attentions = False
    # 11. 如果需要输出注意力权重,发出警告,因为这里不支持输出注意力权重。
    if output_attentions:
        raise ValueError("Output attentions is not supported.")

    bsz, q_len, _ = hidden_states.size() # B,T

    # TODO 分组
    group_size = int(q_len * group_size_ratio)  
    # 检查序列长度是否能够被 group_size 整除,如果不能,抛出错误
    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
    num_group = q_len // group_size  # 计算总共有多少个组

    query_states = self.q_proj(hidden_states)  # B,T,D --> B,T,nq * hd
    key_states = self.k_proj(hidden_states)    # B,T,D --> B,T,nk * hd
    value_states = self.v_proj(hidden_states)  # B,T,D --> B,T,nv * hd

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    # B,T,nq,hd ---> B,nq,T,hd
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # B,T,nk,hd ---> B,nk,T,hd
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # B,T,nv,hd ---> B,nv,T,hd
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # value_states: B,nv,T,hd       position_ids: B,T
    # cos: B,T,hd    sin: B,T,hd
    cos, sin = self.rotary_emb(value_states, position_ids)
    # query_states: B,nq,T,hd    key_states: B,nk,T,hd
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    # B,nq,T,hd --> B,T,nq,hd
    query_states = query_states.transpose(1, 2)
    # B,nk,T,hd --> B,T,nk,hd
    key_states = key_states.transpose(1, 2)
    # B,nv,T,hd --> B,T,nv,hd
    value_states = value_states.transpose(1, 2)
    # attention mask , 只有训练时开启
    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)


    # TODO 新增代码, shift 过程
        # q k v  shift and group,  分别对 query、key 和 value 进行 shift 操作并分组
    # query_states: [B,T,nq,hd] ---> [B*(T//group_size),group_size,nq,hd]
    query_states = shift_2(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    # key_states: [B,T,nk,hd] ---> [B*(T//group_size),group_size,nk,hd]
    key_states = shift_2(key_states, bsz, q_len, group_size, self.num_key_value_heads, self.head_dim)
    # value_states: [B,T,nv,hd] ---> [B*(T//group_size),group_size,nv,hd]
    value_states = shift_2(value_states, bsz, q_len, group_size, self.num_key_value_heads, self.head_dim)
        # attention mask group
    if attention_mask is not None: # attention_mask: [B,T], 而对于attention mask为None的情况, 交给flash attention 处理即可
        # attention_mask: [B,T] --> [B * num_group, group_size]
        attention_mask = attention_mask.reshape(bsz * num_group, group_size)
    
    # TODO 维度检测
    if (
        query_states.size() != (bsz * num_group, group_size, self.num_heads, self.head_dim) or 
        key_states.size() != (bsz * num_group, group_size, self.num_key_value_heads, self.head_dim) or
        value_states.size() != (bsz * num_group, group_size, self.num_key_value_heads, self.head_dim)
        ):
        raise ValueError(
            f"query_states or key_states or value_states size don't match"
        )
    if attention_mask is not None:
        if attention_mask.size() != (bsz * num_group, group_size):
            raise ValueError(
                f"Attention mask should be of size {(bsz * num_group, group_size)}, but is {attention_mask.size()}"
            )

    # 开始计算flash attention
    # query_states: B,T,nq,hd
    # key_states: B,T,nk,hd
    # value_states: B,T,nv,hd
    # attention_mask: 如果句子中没有pad, 此时attention为None,交给flash处理即可, 否则, 使用原本正常的 attention mask
    # attention dropout: 只有在训练时有值

    # attn_output: B,T,nq,hd
    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len//num_group, dropout=dropout_rate
    )

    # TODO 维度检测
    if attn_output.size() != (bsz * num_group, group_size, self.num_heads, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_group, group_size, self.num_heads, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    
    # [bsz * num_group, group_size, nh, hd] --> [bsz, q_len, nh, hd]
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
    # TODO 反向 shift 复原,这里使用 chunk cat处理不使用 in-place的方式, 保持梯度流畅, 公式详情见: https://arxiv.org/pdf/2309.12307
    # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)
    # chunk cat 沿着 2 维度分割或合并, roll 沿着 1 维度(序列维度) 向右滚动 group_size//2 复原
    attn_output = torch.cat(
        (attn_output.chunk(2, dim=2)[0], attn_output.chunk(2, dim=2)[1].roll(group_size//2, dims=1))
        , dim=2)

    # B,T,nq,hd --> B,T,nq * hd
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    # B,T,nq * hd --> B,T,D
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, None, past_key_value

# s2attention 不使用 flash attention
def llama_forward_s2noflashattn(self,
                        hidden_states: torch.Tensor,     # B,T,D
                        # attention_mask为4维：[batch_size, 1, sequence_length, target_length], 1的维度对应的是 num head
                        attention_mask: Optional[torch.Tensor] = None, 
                        position_ids: Optional[torch.LongTensor] = None, # b,t
                        past_key_value: Optional[Cache] = None,
                        output_attentions: bool = False,
                        use_cache: bool = False,
                        cache_position: Optional[torch.LongTensor] = None, # b,t
                        **kwargs,
                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # 11. 如果需要输出注意力权重,发出警告,因为这里不支持输出注意力权重。
    if output_attentions:
        raise ValueError("Output attentions is not supported.")

    bsz, q_len, _ = hidden_states.size()   # 获得B,T

    # TODO 分组
    group_size = int(q_len * group_size_ratio)  
    # 检查序列长度是否能够被 group_size 整除,如果不能,抛出错误
    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
    num_group = q_len // group_size  # 计算总共有多少个组

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

    else:
        query_states = self.q_proj(hidden_states) # B,T,D --> B,T,nq * hd
        key_states = self.k_proj(hidden_states)   # B,T,D --> B,T,nk * hd
        value_states = self.v_proj(hidden_states) # B,T,D --> B,T,nv * hd

    # B,T,nh,hd ---> B,nh,T,hd
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # B,T,nk,hd ---> B,nk,T,hd
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # B,T,nv,hd ---> B,nv,T,hd
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    past_key_value = getattr(self, "past_key_value", past_key_value)
    # cos: B,T,hd,  sin: B,T,hd
    cos, sin = self.rotary_emb(value_states, position_ids)
    # query_states: B,nh,T,hd     key_states:  B,nk,T,hd 
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # 广播 B,nk,T,hd ---> B,nq,T,hd
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    # 广播 B,nv,T,hd ---> B,nq,T,hd
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # TODO 新增代码，shift 过程, 分别对 query、key 和 value 进行 shift 操作
    # query_states: [B,nq,T,hd] ---> [B*(T//group_size),nq,group_size,hd]
    query_states = shift_1(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    # key_states: [B,nq,T,hd] ---> [B*(T//group_size),nq,group_size,hd]
    key_states = shift_1(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    # value_states: [B,nq,T,hd] ---> [B*(T//group_size),nq,group_size,hd]
    value_states = shift_1(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)

    # query_states: B,nq,T,hd
    # key_states.transpose(2, 3): B,nq,hd,T
    # attn_weights: (B,nq,T,hd) * (B,nq,hd,T) ---> B,nq,T,T
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # TODO 新增, 检测维度
    if attn_weights.size() != (bsz * num_group, self.num_heads, group_size, group_size):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_group, self.num_heads, group_size, group_size)}, but is"
            f" {attn_weights.size()}"
        )
    # # 源代码存在两个问题, 已注释
    #     # 1. 源代码没有使用到对角线的attention mask, 这在left pad 的情况可能会有问题
    #     # 2. repeat 函数重复没有考虑"序列"相连
    #     # 关于 repeat 与 expand 的区别，详见 https://zhuanlan.zhihu.com/p/555322123
    # attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)

    # TODO 新增代码: 更改 attention mask 的维度以适应新的 attn_weights
        # 原始的 attention mask 维度为 [B,1,T,T], 1 表示 num head 这个维度, 后续会广播
        # attention mask: [B,1,T,T] --> [B * num_group,1,group_size,group_size]
    atten_group = []
    for i in range(num_group):
        # 取对角线 attention mask 分块
        atten_group.append(attention_mask[:, :, i*group_size: i*group_size + group_size, i*group_size: i*group_size + group_size])
        # attention mask: [B, num_group, 1, group_size, group_size]
    attention_mask = torch.stack(atten_group, dim=1)
        # attention mask: [B, num_group, 1, group_size, group_size] --> [B*num_group, 1, group_size, group_size]
    attention_mask = attention_mask.reshape(bsz * num_group,1,group_size,group_size)
    # 验证维度
    if attention_mask is not None:
        if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
            raise ValueError(
                f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    # attn_weights: B,nq,T,T
    # value_states: B,nq,T,hd
    # attn_output: (B,nq,T,T) * (B,nq,T,hd) --> B,nq,T,hd
    attn_output = torch.matmul(attn_weights, value_states)

    # TODO, 改,检查注意力输出的形状是否正确
    if attn_output.size() != (bsz * num_group, self.num_heads, group_size, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    
    # [B*num_group, nh, group_size, hd] --> [B*num_group, group_size, nh, hd]
    attn_output = attn_output.transpose(1, 2).contiguous()
    # [B*num_group, group_size, nh, hd] --> [B,q_len, nh, hd]
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
    # TODO 反向 shift 复原,这里使用 chunk cat处理不使用 in-place的方式, 保持梯度流畅, 公式详情见: https://arxiv.org/pdf/2309.12307
    # attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)
    # chunk cat 沿着 2 维度分割或合并, roll 沿着 1 维度(序列维度) 向右滚动 group_size//2 复原
    attn_output = torch.cat(
        (attn_output.chunk(2, dim=2)[0], attn_output.chunk(2, dim=2)[1].roll(group_size//2, dims=1))
        , dim=2)

    # [bsz, q_len, nh, hd] --> [bsz, q_len, hidden_size]
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        # B,T,D --> B,T,D
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value



def replace_model_s2attn_only_train(use_flash_attn=True, model_type='llama',enable_s2attention=False):
    """
        警告: 仅在训练阶段/验证集验证loss时进行替换, 推理阶段(generate)不使用该函数
    """
    logger.success(f"'use_flash_attn' is set to '{use_flash_attn}'")
    logger.success(f"'enable_s2attention' is set to '{enable_s2attention}'")
    logger.success(f"now model type is '{model_type}'")
    # 使用flash attention时评估算力
    if use_flash_attn:  
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        if cuda_major < 8:
            warnings.warn(
                "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
                "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            )

    if enable_s2attention: # 是否使用s2attention
        if model_type == 'llama':
            if use_flash_attn:
                # 替换 LlamaFlashAttention2 中的 forward 函数
                transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_forward_s2flashattn
            else:
                # 替换 LlamaAttention 中的 forward 函数
                transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_forward_s2noflashattn
        else:
            logger.warning(f"{model_type} is not supported 'shift short attention' now, use normal attention instead.")
    

    else:
        pass
