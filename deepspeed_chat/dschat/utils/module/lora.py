# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import math
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
import deepspeed

# 该代码实现了一个名为 LinearLayer_LoRA 的模块,用于在大型预训练语言模型中有效地进行微调。它使用了以下技术:

# LoRA (Low-Rank Adaptation):LoRA 是一种用于微调大型预训练模型的技术,它通过在每个线性层中添加一个低秩投影矩阵来修改模型权重,从而显著降低了所需的参数数量。这解决了直接微调大型模型需要大量额外内存的问题。
# DeepSpeed ZeRO 优化:代码中考虑了 DeepSpeed 库的 ZeRO 优化,在阶段 3 时,它尝试获取权重张量的分片形状,以适应 ZeRO 优化的内存优化策略。
# Dropout:代码支持在 LoRA 层的输出上应用 Dropout 正则化,以防止过拟合。
# 参数初始化:代码使用 Kaiming 均匀分布初始化 LoRA 层的右投影矩阵,并使用全零初始化左投影矩阵。
# 权重融合:代码提供了 fuse_lora_weight 和 unfuse_lora_weight 方法,用于在推理阶段将 LoRA 层融合到原始权重中,以加速计算;或在训练阶段将其分离,以便继续优化 LoRA 层参数。
class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    # 1. 这是一个实现 LoRA (Low-Rank Adaptation) 的简单线性层
    # 目前仅支持线性层,LoRA 是一种用于有效微调大型预训练语言模型的技术
    # 它通过在每个线性层中添加一个低秩投影矩阵来修改模型权重,从而显著降低了所需的参数数量
    # 这解决了直接微调大型模型需要大量额外内存的问题
    def __init__(self,
                 weight,    # 2. weight 参数是原始线性层的权重张量,通常是预训练模型中的一部分
                 lora_dim=0,  # 3. lora_dim 参数指定 LoRA 层的低秩投影维度,它决定了 LoRA 层引入的新参数数量
                              # 较小的 lora_dim 可以减少参数数量,但可能会降低模型性能
                 lora_scaling=1,  # 4. lora_scaling 参数是 LoRA 层输出的缩放因子,用于控制 LoRA 层对原始权重的影响程度
                                  # 一般设置为 1,但可以根据实际情况进行调整
                 lora_droppout=0,  # 5. lora_droppout 参数指定应用于 LoRA 层输出的 Dropout 比例,用于正则化和防止过拟合
                                   # 设置为 0 表示不使用 Dropout
                 bias=None):    # 6. bias 参数是原始线性层的可选偏置张量
        super(LinearLayer_LoRA, self).__init__()
        self.weight = weight
        self.bias = bias

        # 7. 检查 lora_dim 是否大于 0,因为 LoRA 需要一个正的低秩投影维度
        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )
        
        # 8. 尝试获取权重张量的 DeepSpeed 分片形状,适用于使用 DeepSpeed 库的 ZeRO 优化阶段 3
            # DeepSpeed 是一个深度学习优化库,用于提高大型模型训练的效率和性能
            # ZeRO 优化是 DeepSpeed 中的一种技术,通过分片和优化内存利用来减少模型训练所需的内存占用
        try:
            # for zero stage 3, 优化ZeRO3
            rows, columns = weight.ds_shape
        except:
            # 9. 如果无法获取 DeepSpeed 分片形状,则使用原始权重张量的形状
            rows, columns = weight.shape
        # 10. 初始化 LoRA 层的右投影矩阵,作为可训练参数
            # 右投影矩阵的形状为 (输出特征维度, lora_dim),用于将原始权重投影到低秩空间
            # 左投影矩阵的形状为 (lora_dim, 输入特征维度),用于将低秩投影结果映射回原始权重空间
        self.lora_right_weight = nn.Parameter(torch.zeros(columns,lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        # 12. 计算 LoRA 层输出的缩放因子,确保其与原始权重具有相似的量级
        self.lora_scaling = lora_scaling / lora_dim

        # 13. 根据 lora_droppout 参数初始化 Dropout 层或标识映射
        # Dropout 是一种正则化技术,通过在训练期间随机丢弃一些神经元来防止过拟合
        if lora_droppout > 0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        # 14. 禁用原始权重的梯度计算,因为只需要优化 LoRA 层的参数
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        # 15. fuse_lora 标志用于控制是否将 LoRA 层融合到原始权重中
        # 在推理阶段,可以将 LoRA 层融合到原始权重中以加速计算
        self.fuse_lora = False

    def eval(self):
        # 16. 在评估模式下,将 LoRA 层的 Dropout 设置为评估模式
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        # 18. 在训练模式下,将 LoRA 层的 Dropout 设置为训练模式
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        # 20. 使用 Kaiming 均匀分布初始化 LoRA 右投影矩阵
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        # 21. 使用全零初始化 LoRA 左投影矩阵
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        # 22. 将 LoRA 层融合到原始权重中,以加速推理
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        # 23. 将 LoRA 层从原始权重中分离,以便继续训练
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, input):
        # 24. 如果 LoRA 层已融合到原始权重中,直接进行线性变换
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            # 25. 否则,将 LoRA 层的输出与原始线性层的输出相加
            return F.linear(
                input, self.weight,
                self.bias) + (self.lora_dropout(input) @ self.lora_right_weight
                              @ self.lora_left_weight) * self.lora_scaling



# 该函数的主要目的是将模型中指定的线性层替换为LoRA (Low-Rank Adaptation)层。LoRA是一种用于微调大型预训练模型的技术,它通过引入低秩矩阵来适应新任务,从而大大减少需要微调的参数数量,提高了内存和计算效率。

# 该函数使用了以下技术:

# LoRA (Low-Rank Adaptation): 一种用于微调大型预训练模型的技术,通过引入低秩矩阵来适应新任务,减少需要微调的参数数量。
# PyTorch模块遍历: 使用model.named_modules()遍历模型中的所有模块,并根据名称和类型筛选出需要替换的线性层。
# 递归访问模型属性: 使用recursive_getattr和recursive_setattr函数递归访问和设置模型中的属性,以便替换指定的线性层。
# 该函数的关键步骤如下:

# 遍历模型中的所有模块,并筛选出名称包含part_module_name字符串且类型为nn.Linear的线性层。
# 对于每个需要替换的线性层,使用recursive_getattr函数获取其在模型中的实例。
# 创建一个LinearLayer_LoRA对象,该对象是一个包装类,用于将原始的线性层替换为LoRA层。它需要传入原始线性层的权重和偏置,以及LoRA的秩、缩放因子和dropout比例。
# 使用recursive_setattr函数将原始线性层替换为新创建的LoRA层。
# 返回修改后的模型。
# 通过使用LoRA技术,该函数能够在保持大部分预训练参数不变的情况下,有效地微调大型预训练模型以适应新任务,从而显著减少所需的内存和计算资源。这对于在资源受限的环境中微调大型模型尤为有用。

# convert the linear layer to LoRA
def convert_linear_layer_to_lora(model,  # 1. model是要修改的模型
                                 part_module_name,  # 2. part_module_name是一个字符串,用于指定要修改的线性层的名称部分
                                 lora_dim=0,  # 3. lora_dim是LoRA的秩,即低秩矩阵的维度,默认为0
                                 lora_scaling=1,    # 4. lora_scaling是LoRA的缩放因子,默认为1
                                 lora_droppout=0):  # 5. lora_droppout是LoRA中应用于低秩矩阵的dropout比例,默认为0
    # 6. replace_name是一个列表,用于存储需要替换为LoRA层的线性层的完整名称
    replace_name = []

    # 7. 遍历模型中的所有模块
    for name, module in model.named_modules():
        # 8. 检查当前模块是否为nn.Linear类型,并且名称包含part_module_name字符串
        if isinstance(module, nn.Linear) and part_module_name in name:
            # 9. 如果满足条件,将模块的完整名称添加到replace_name列表中
            replace_name.append(name)

     # 10. 遍历需要替换的线性层
    for name in replace_name:
        # 11. 使用recursive_getattr函数获取模型中对应名称的模块
        module = recursive_getattr(model, name)

        # 12. 创建一个LinearLayer_LoRA对象,它是一个包装类,用于将原始的线性层替换为LoRA层
        # LinearLayer_LoRA的参数包括:原始线性层的权重和偏置,以及LoRA的秩、缩放因子和dropout比例
        tmp = LinearLayer_LoRA(
            module.weight, lora_dim, lora_scaling, lora_droppout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        
        # 13. 使用recursive_setattr函数将原始线性层替换为新创建的LoRA层
        recursive_setattr(model, name, tmp)
    # 14. 返回修改后的模型
    return model


def _z3_params_to_fetch(param_list):
    # 过滤 ZeRO3 参数
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.
        partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    ]


# 该函数的主要作用是将模型中的LoRA(Low-Rank Adaptation)层转换为标准的线性层。LoRA是一种参数有效微调技术,通过引入少量可训练参数来适应新任务,而无需微调整个大型预训练模型。在训练结束后,需要将LoRA层转换为标准线性层,以便模型可以直接用于推理,无需额外的LoRA计算。它涉及以下几个关键技术:
# 模型遍历: 通过遍历模型中的所有模块,找出LoRA层的名称并存储在replace_name列表中。这是将LoRA层转换为标准线性层的第一步。
# LoRA层识别: 该函数依赖于LinearLayer_LoRA类来识别LoRA层。如果一个模块是LinearLayer_LoRA的实例,就将其名称添加到replace_name列表中。
# ZeRO Stage 3优化: DeepSpeed的ZeRO优化策略旨在减少模型参数和梯度所需的内存,支持更大的模型。Stage 3是ZeRO优化的最高阶段,可以将模型参数和梯度分散到多个GPU上。该函数检查是否需要进行ZeRO Stage 3优化,并在必要时使用deepspeed.zero.GatheredParameters上下文管理器正确地收集和处理参数。
# LoRA权重融合: 对于每个需要替换的LoRA层,该函数调用module.fuse_lora_weight()方法,将LoRA权重融合到标准线性层中。这一步骤实际上将LoRA层转换为标准线性层。
# 通过使用这些技术,该函数可以将训练后的模型中的LoRA层转换为标准线性层,从而使模型可以直接用于推理,无需额外的LoRA计算。这种转换对于部署和inference非常重要,因为它可以提高推理速度并减少内存占用。
# 需要注意的是,该函数依赖于DeepSpeed的ZeRO优化策略,因此在使用前需要正确初始化DeepSpeed环境。此外,它还依赖于LinearLayer_LoRA类和fuse_lora_weight()方法的实现,这些实现需要与模型架构和LoRA技术相匹配。

# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model):
    # 1. 该函数的目的是将模型中的LoRA(Low-Rank Adaptation)层转换为标准的线性层。
    # LoRA是一种参数有效微调技术,通过引入少量可训练参数来适应新任务,而无需微调整个大型预训练模型。
    # 在训练结束后,需要将LoRA层转换为标准线性层,以便模型可以直接用于推理,无需额外的LoRA计算。
    replace_name = []  # 2. 存储需要替换的LoRA层名称的列表
    # 3. 遍历模型中的所有模块,找出LoRA层的名称并存储在replace_name列表中。
    for name, module in model.named_modules(): 
        if isinstance(module, LinearLayer_LoRA):
            replace_name.append(name)
    # 4. 对于每个需要替换的LoRA层:
    for name in replace_name:
        module = recursive_getattr(model, name)  # 5. 使用recursive_getattr获取LoRA层模块对象
        # 6. 检查是否需要进行ZeRO Stage 3优化
        # ZeRO是DeepSpeed中的一种优化策略,旨在减少模型参数和梯度所需的内存。
        # Stage 3是ZeRO优化的最高阶段,可以将模型参数和梯度分散到多个GPU上,从而支持更大的模型。
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        # 7. 使用DeepSpeed的GatheredParameters上下文管理器,确保在ZeRO Stage 3优化下正确地收集和处理参数。
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.lora_left_weight,
                module.lora_right_weight
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.fuse_lora_weight()  # 8. 调用LoRA层的fuse_lora_weight()方法,将LoRA权重融合到标准线性层中
    # 9. 返回转换后的模型,其中所有LoRA层已被替换为标准线性层。
    return model


def only_optimize_lora_parameters(model, force_optimize_params=[]):
    # turn off the gradient of all the parameters except the LoRA parameters
    # 1. 该函数的目的是在模型中仅优化LoRA (Low-Rank Adaptation)相关的参数,同时冻结其他参数,不让它们进行梯度更新。
    # 2. force_optimize_params是一个列表,用于指定除了LoRA参数之外还需要优化的其他参数名称。

    # 3. 遍历模型的所有参数
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name or name in force_optimize_params:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def make_model_gradient_checkpointing_compatible(model):
    # Higgingface added this enable input require grads function to make gradient checkpointing work for lora-only optimization
    # 1. 该函数的目的是使模型与梯度检查点(Gradient Checkpointing)技术兼容,以减少内存占用。

    # 2. 检查模型是否具有enable_input_require_grads方法,如果有,则调用该方法。
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # 3. 如果模型没有enable_input_require_grads方法,但具有get_input_embeddings方法,则注册一个前向钩子函数。
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            # 4. 该前向钩子函数将输出的requires_grad属性设置为True,以确保梯度可以正确传播。
            output.requires_grad_(True)

        # 5. 在模型的输入嵌入层上注册前向钩子函数。
        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    return model
