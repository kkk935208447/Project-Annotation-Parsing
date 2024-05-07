# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator
import torch.nn as nn


def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)


def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


class ExponentialMovingAverage:

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.ema = None

    def update(self, num):
        prev_ema = num if self.ema is None else self.ema
        self.ema = self.alpha * prev_ema + (1.0 - self.alpha) * num
        return self.ema

    def get(self):
        return self.ema if self.ema is not None else 0.

"""
这段代码实现了一个名为 get_tokenizer 的函数,用于获取预训练模型的 tokenizer。
tokenizer 是自然语言处理任务中一个非常重要的组件,用于将文本转换为模型可以理解的数字表示(token)。
该函数根据传入的预训练模型名称或路径,使用 Hugging Face Transformers 库来加载对应的 tokenizer。
下面我将对每一行代码进行详细注释并解析使用的技术。
"""
# 定义一个函数 get_tokenizer,接受两个参数:
# model_name_or_path: 预训练模型的名称或本地路径
# fast_tokenizer: 是否使用快速 tokenizer,默认为 True
def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    # 1. 如果 model_name_or_path 中包含 "llama",表示使用 LLaMA 模型
    if "llama" in model_name_or_path:
        # 从 transformers 库中导入 LlamaTokenizer, AutoTokenizer
        from transformers.models.llama import AutoTokenizer
        # 使用 from_pretrained 方法加载 LlamaTokenizer, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        # 如果 tokenizer 的 pad_token 为 None
        if tokenizer.pad_token is None:
            # 直接添加一个特殊标记 '[PAD]' 作为 pad_token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # 设置 padding 的方向为右侧
            tokenizer.padding_side = 'right'
    else:
        # 2. 如果 model_name_or_path 不包含 "llama",表示使用其他预训练模型
        # 使用 AutoTokenizer 根据模型路径自动加载对应的 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        # 将 pad_token 设置为 eos_token (结束标记)
        tokenizer.pad_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        # 设置 padding 的方向为右侧
        tokenizer.padding_side = 'right'
    return tokenizer  # 返回加载好的 tokenizer


# 这段代码使用了以下技术:

# 文件路径处理: 使用os.path.exists和os.path.join函数检查文件路径是否存在并拼接路径。
# JSON文件解析: 使用json.load函数读取JSON文件,获取模型的实际名称。
# Hugging Face tokenizer加载: 使用get_tokenizer函数从Hugging Face模型中加载tokenizer。
# 特殊标记添加: 使用tokenizer的add_special_tokens方法向tokenizer中添加特殊标记。
# 使用这些技术的原因是为了方便地加载Hugging Face预训练模型的tokenizer,并根据需要添加自定义的特殊标记。这对于处理和预处理自然语言数据非常重要,可以提高模型的性能和灵活性。

# 该函数解决了以下问题:

# 统一了从本地路径或在线资源加载tokenizer的方式。
# 处理了添加特殊标记的需求,使tokenizer能够适应特定的任务或数据。
# 简化了tokenizer加载的过程,提高了代码的可读性和可维护性。
def load_hf_tokenizer(model_name_or_path,
                      fast_tokenizer=True,
                      add_special_tokens=None):
    # 这个函数用于加载Hugging Face的tokenizer
    # 它接受以下参数:
    # 1. model_name_or_path: 预训练模型的名称或本地路径
    # 2. fast_tokenizer: 是否使用快速tokenizer,默认为True
    # 3. add_special_tokens: 需要添加的特殊标记,如果是字符串则自动转换为列表
    if os.path.exists(model_name_or_path):
        # 如果model_name_or_path是一个本地路径并且存在
        # Locally tokenizer loading has some issue, so we need to force download
        # 本地加载tokenizer存在一些问题,因此需要强制下载
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            # 如果config.json文件存在
            model_json_file = json.load(open(model_json))

            # TODO:修改官方代码, 为了解决Hugging Face给定本地路径,强制线上加载模型的问题, ***** *****之间的官方源码
            # ******
            # # 从config.json文件中获取实际的模型名称
            # model_name = model_json_file.get("_name_or_path",
            #                                  model_name_or_path)
            
            # # 使用get_tokenizer函数加载tokenizer
            # tokenizer = get_tokenizer(model_name,
            #                           fast_tokenizer=fast_tokenizer)
            # ******

        # TODO:添加代码
        # model_name_or_path是不是本地路径都直接使用get_tokenizer函数加载tokenizer
        tokenizer = get_tokenizer(model_name_or_path,
                                fast_tokenizer=fast_tokenizer)

    else:
        # 如果model_name_or_path不是本地路径
        # 直接使用get_tokenizer函数加载tokenizer
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    if add_special_tokens is not None:
        # 如果需要添加特殊标记
        # 如果add_special_tokens是字符串,则转换为列表
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        # 使用tokenizer的add_special_tokens方法添加特殊标记
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})

    # 返回加载后的tokenizer对象
    return tokenizer


# 该函数使用了以下技术:
# Hugging Face格式保存:该函数用于将PyTorch模型和tokenizer保存为Hugging Face格式,从而可以使用hf.from_pretrained加载。这种格式方便模型的共享和部署。
# 数据并行处理:代码考虑了模型可能是torch.nn.parallel.DistributedDataParallel或torch.nn.DataParallel实例的情况,并获取其module属性作为要保存的模型。这是为了处理数据并行训练的情况。
# LoRA权重过滤:在保存模型权重时,代码会过滤掉与LoRA(Low-Rank Adaptation)相关的权重。LoRA是一种模型压缩技术,可以减小模型大小,但在保存最终模型时通常不需要保留LoRA权重。
# 该函数解决了以下问题:
# 将训练好的PyTorch模型和tokenizer保存为标准的Hugging Face格式,方便后续加载和部署。
# 处理数据并行训练的情况,确保保存正确的模型权重。
# 过滤掉不必要的LoRA权重,减小保存的模型文件大小。
def save_hf_format(model, tokenizer, args, sub_folder=""):
    # 用于保存Hugging Face格式的模型和tokenizer,以便可以使用hf.from_pretrained加载
    
    # 如果模型是一个torch.nn.parallel.DistributedDataParallel或torch.nn.DataParallel实例,则获取其module属性作为要保存的模型, hasattr 判断里面有没有 "module" 这个字符串
    model_to_save = model.module if hasattr(model, 'module') else model

    # 定义要保存的文件名
    CONFIG_NAME = "config.json"   # 模型配置文件
    WEIGHTS_NAME = "pytorch_model.bin"  # 模型权重文件

    # 构建输出目录路径
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)      # 创建输出目录,如果不存在则创建

    # 构建输出文件路径
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    # 准备要保存的模型权重字典
    save_dict = model_to_save.state_dict()

    # 从要保存的权重中删除所有与LoRA相关的权重
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]

    # 保存模型权重
    torch.save(save_dict, output_model_file)

    # 保存模型配置
    model_to_save.config.to_json_file(output_config_file)

    # 保存tokenizer词汇表
    # TODO 修改官方代码, 源代码只能保存词汇表,没有保存其他的信息,例如tokenizer的特殊符号,这里修改为
    # tokenizer.save_vocabulary(output_dir)
    # TODO 修改代码
    tokenizer.save_pretrained(output_dir)

# set random seed
def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    # 1. 该函数的目的是计算并行进程间张量的平均值。
    # 2. 在分布式训练中,每个进程持有部分数据和相应的梯度。为了获取全局的平均梯度,需要在所有进程间进行通信和计算。

    # 3. torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    # 使用PyTorch的分布式通信原语torch.distributed.all_reduce()对所有进程中的tensor进行求和操作。
    # op参数指定了进行的操作,此处使用ReduceOp.SUM表示对所有进程的tensor进行元素级别的求和。
    # 这一步骤将所有进程的tensor累加到每个进程上,获得全局的tensor之和。
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    # 4. tensor = tensor / torch.distributed.get_world_size()
    # 获取当前分布式环境中的进程数量,即并行训练的进程数。
    # 由于前一步骤获得的是所有进程tensor的累加和,因此需要将其除以进程数量以获得平均值。
    # 这一步骤计算了全局平均张量,作为最终的输出结果。
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


# This function is a modified version of code available in the from_pretrained API of HuggingFace Transformers
# The code is copied and modified from: https://github.com/huggingface/transformers/blob/5ee9693a1c77c617ebc43ef20194b6d3b674318e/src/transformers/modeling_utils.py#L498
# This function helps load a HF format checkpoint into a DeepSpeed wrapped model that has been sharded using ZeRO Stage 3
# 这个函数的主要作用是将预训练模型的状态字典加载到一个经过 DeepSpeed 包装并使用 ZeRO 优化的模型中。它考虑了 ZeRO 优化第3阶段的特殊处理,以确保与分片模型的兼容性。
# 关键点:
# 函数接受要加载状态字典的目标模型、预训练模型的状态字典、状态字典键的前缀以及 DeepSpeed 的 ZeRO 优化阶段作为参数。
# 采用递归的方式逐层加载模型参数,以确保能够处理模块的子层。
# 对于使用 ZeRO 优化第3阶段的模型,需要特殊处理,先聚合参数,然后再从状态字典加载,最后再重新分片。
# 在多 GPU 训练环境中,只有排名为0的进程负责实际的参数加载操作。
# 函数最后删除状态字典副本,以便更早被垃圾回收。
def load_state_dict_into_model(model_to_load=None,  # 1. 要加载状态字典的目标模型
                               state_dict=None,     # 2. 要加载的预训练模型的状态字典
                               start_prefix="",     # 3. 状态字典中键的前缀,用于指定要加载的部分权重
                               zero_stage=0):       # 4. DeepSpeed的ZeRO优化阶段,用于控制加载的兼容性

    # copy state_dict so _load_from_state_dict can modify it
    # 5. 创建 state_dict 的副本,以便 _load_from_state_dict 函数可以对其进行修改
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # 6. 用于记录加载过程中的错误消息
    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    # 7. 需要递归地应用 _load_from_state_dict 函数,因为 PyTorch 的实现不会复制模块子层的参数。
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        # 9. 准备 _load_from_state_dict 函数所需的参数
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        # 10. 检查状态字典中是否有以当前前缀开头的参数,如果没有则可以提早退出
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if zero_stage == 3:
                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                # 11. 如果使用了 ZeRO 优化的第3阶段,则需要特殊处理
                # 由于模型被分片,每个分片只有部分状态字典,因此只需要加载当前分片中存在的参数
                named_parameters = dict(
                    module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [
                    named_parameters[k] for k in state_dict.keys()
                    if k in named_parameters
                ]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    # 12. 使用 deepspeed.zero.GatheredParameters 上下文管理器,先将参数聚合,然后再从状态字典加载,最后再将参数分片
                    with deepspeed.zero.GatheredParameters(params_to_gather,
                                                           modifier_rank=0):
                        # 13. 仅在排名为0的进程中执行参数加载
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                # 14. 如果不使用 ZeRO 优化的第3阶段,则直接调用 _load_from_state_dict 加载参数
                module._load_from_state_dict(*args)

        # 15. 递归地处理模块的子层
        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    # 16. 调用内部的 load 函数,开始加载模型参数
    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    # 17. 删除 state_dict 副本,以便更早被垃圾回收
    del state_dict

    return error_msgs    # 18. 返回加载过程中记录的错误消息


# 该函数用于根据模型参数的名称和特性,将它们划分为三个组,以便在优化器中应用不同的策略。
# 第一组参数是需要应用权重衰减,且不属于 LoRA 层的参数。
# 第二组参数是需要应用权重衰减,且属于 LoRA 层的参数,它们会使用一个特殊的学习率。
# 第三组参数是不应该应用权重衰减的参数,如偏置和层归一化参数。
# 函数使用了 LoRA 技术,这是一种低秩适应的方法,可以在不修改模型结构的情况下对模型进行微调。
# 函数最终会移除空组,并返回非空的参数组列表,以供创建优化器使用。
def get_optimizer_grouped_parameters(
    model,           # 1. model 参数是需要优化的模型。
    weight_decay,    # 2. weight_decay 参数是权重衰减系数,用于防止模型过拟合。
    lora_lr=5e-4,   # 3. LoRA (Low-Rank Adaptation)层的学习率,默认为5e-4

    #一个列表,包含了不应该应用权重衰减的参数名称,通常包括偏置(bias)和层归一化(layer norm)参数。
    no_decay_name_list=[   
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",   
        "ln_f.weight"
    ],
    # 5. LoRA层参数名称列表
    lora_name_list=["lora_right_weight", "lora_left_weight"],    
):
    # 6. 根据参数名称和要求,将模型参数划分为三个组:
    # 第一组:需要应用权重衰减,且不属于 LoRA 层的参数
    """
        # 列表表达式等价于:
        first_group_params = []
        for name, param in model.named_parameters():
            # 检查参数名称是否不包含 no_decay_name_list 中的任何字符串
            if not any(nd in name.lower() for nd in no_decay_name_list):
                # 检查参数是否可训练
                if param.requires_grad:
                    # 检查参数名称是否不包含 lora_name_list 中的任何字符串
                    if not any(nd in name.lower() for nd in lora_name_list):
                        first_group_params.append(param)
    """
    # 第二组:需要应用权重衰减,且属于 LoRA 层的参数
    # 第三组:不应用权重衰减的参数,如偏置和层归一化参数
    optimizer_grouped_parameters = [
        {
            "params": [ # 7. 第一组:需要应用权重衰减,且不属于LoRA层的参数
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower()
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,      # 8. 应用指定的权重衰减系数
        },
        {
            "params": [   # 9. 第二组:需要应用权重衰减,且属于LoRA层的参数
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower()
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,    # 10. 应用指定的权重衰减系
            "lr":
            lora_lr     # 11. 应用LoRA层的特殊学习率
        },
        {
            "params": [  # 12. 第三组:不应用权重衰减的参数,如偏置和层归一化参数
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":   # 13. 权重衰减系数设置为0
            0.0,
        },
    ]

    # 14. 移除空组,避免优化器出错
    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    # 15. 返回非空的参数组列表,用于创建优化器
    return non_empty_groups


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]

# 在这个注释版本中,我主要解释了以下几个方面:

# 函数参数的作用,包括主模型、EMA 模型、衰减因子、设备和 ZeRO 优化阶段。
# 使用 torch.no_grad() 上下文管理器禁止梯度计算,以提高计算效率。
# 在 ZeRO 阶段 3 优化下,需要使用 deepspeed.zero.GatheredParameters 上下文管理器聚合参数。
# 如何根据设备将参数数据移动到对应的设备上。
# 指数移动平均公式的原理及其在模型训练中的作用,即平滑模型参数变化,提高模型稳定性和泛化性能。
# 1. model 参数是需要更新的主要模型实例。
# 2. model_ema 参数是需要更新的指数移动平均(EMA)模型实例。
# 3. beta 参数是用于计算指数移动平均的衰减因子,默认为 0.992。
# 4. device 参数指定了模型参数所在的设备,如 CPU 或 GPU。
# 5. zero_stage 参数指定了 DeepSpeed 的 ZeRO 优化阶段,用于控制模型参数的分片和优化策略。
def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    # 6. 检查当前的 ZeRO 优化阶段是否为 3。
    zero_stage_3 = (zero_stage == 3)
    # 7. 使用 torch.no_grad() 上下文管理器,禁止梯度计算。
    with torch.no_grad():
        # 8. 遍历主模型和 EMA 模型的所有参数,更新 EMA 模型的参数。
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            # 9. 如果使用了 ZeRO 阶段 3 优化,则获取需要聚合的参数。
            # _z3_params_to_fetch 是一个自定义函数,用于确定在 ZeRO 阶段 3 下需要聚合的参数。
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            # 10. 判断是否需要进行参数聚合。
            should_gather_param = len(params_to_fetch) > 0
            # 11. 使用 deepspeed.zero.GatheredParameters 上下文管理器,处理需要聚合的参数。
            # deepspeed.zero.GatheredParameters 是 DeepSpeed 提供的一个工具,用于在 ZeRO 优化中聚合参数。
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                # 12. 获取主模型参数的数据。
                data = param.data
                # 13. 如果指定了设备,则将参数数据移动到对应的设备上。
                if device is not None:
                    data = data.to(device)
                # 14. 使用指数移动平均公式更新 EMA 模型参数。
                # 指数移动平均可以平滑模型参数的变化,提高模型的稳定性和泛化性能。
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


# 这个函数可以把分散在各个设备上的model参数拉倒同一个进程中, 然后保存
def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    # 1. 该函数的目的是保存模型权重,主要用于DeepSpeed的ZeRO Stage 3优化场景。ZeRO是DeepSpeed中的一种优化策略,旨在减少模型参数和梯度所需的内存,支持更大的模型。
    # Stage 3是ZeRO优化的最高阶段,可以将模型参数和梯度分散到多个GPU上。我们做的是需要把Stage 3分散在各个设备上的model参数拉倒同一个进程中, 然后保存
    zero_stage_3 = (zero_stage == 3)  # 2. 检查是否为ZeRO Stage 3优化场景
    # 3. 创建保存模型权重的目录
    os.makedirs(save_dir, exist_ok=True) 
    # 4. 保存模型权重的文件名
    WEIGHTS_NAME = "pytorch_model.bin"
    # 5. 构建模型权重文件的完整路径
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    # 6. 获取需要保存的模型对象
    # 如果是DataParallel模式,则获取model_ema.module
    # 否则直接获取model_ema
    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    
    if not zero_stage_3: # 7. 如果不是ZeRO Stage 3优化场景
        if global_rank == 0: # 8. 只有全局rank为0的进程保存模型权重
            torch.save(model_to_save.state_dict(), output_model_file)

    else: # 9. 如果是ZeRO Stage 3优化场景
        output_state_dict = {}  # 10. 创建一个新的state_dict字典,用于存储需要保存的权重
        for k, v in model_to_save.named_parameters(): # 11. 遍历模型的所有参数
            # 12. 如果该参数参与了ZeRO Stage 3优化(即被分散到多个GPU上)
            if hasattr(v, 'ds_id'):
                # 13. 使用DeepSpeed的GatheredParameters上下文管理器,正确地收集和处理分散的参数
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]),enabled=zero_stage_3):
                    # 14. 将收集后的参数数据移动到CPU上, 转到cpu即global_rank为0的进程
                    v_p = v.data.cpu()  
            else:  # 15. 如果该参数没有参与ZeRO Stage 3优化
                v_p = v.cpu()  # 16. 直接将参数数据移动到CPU上

            # 17. 如果是全局rank为0的进程,且参数名称不包含"lora"(排除LoRA层参数)
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p # 18. 将该参数添加到输出state_dict中, 包含cpu的参数
        # 19. 只有全局rank为0的进程保存模型权重
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file) # 20. 保存输出state_dict
        del output_state_dict  # 21. 删除临时的output_state_dict字典,释放内存
