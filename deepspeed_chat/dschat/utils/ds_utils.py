# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

# deepspeed 配置文件
"""
在这段代码中,主要使用了以下技术:
DeepSpeed配置
该函数的主要目的是生成DeepSpeed训练配置字典,用于初始化DeepSpeed分布式训练环境。

ZeRO优化器
通过zero_opt_dict配置ZeRO优化器的参数,如优化阶段、offload设备、参数持久化阈值等。
支持ZeRO优化的不同阶段(0、1、2、3),可以有效减少显存占用。

混合精度训练
通过设置dtype_config配置,支持FP16和BF16混合精度训练,可以加速训练并节省显存。

LoRA (Low-Rank Adaptation)
如果启用了混合精度LoRA,会设置相关的ZeRO优化器配置,如零量化非可训练权重和ZeRO高性能优化分区大小。

混合引擎
通过hybrid_engine配置,支持混合引擎功能,如最大输出tokens数、推理张量并行大小、参数固定等。

TensorBoard
通过tensorboard配置,支持TensorBoard可视化,设置输出路径和作业名称。
这段代码的主要作用是根据不同的参数和配置,生成DeepSpeed训练配置字典。通过配置ZeRO优化器、混合精度训练、LoRA和混合引擎等技术,可以提高训练效率、节省显存,并支持可视化等功能。该配置字典将在后续的DeepSpeed初始化过程中使用
"""
def get_train_ds_config(offload,
                        dtype,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,  # 推理张量并行大小
                        release_inference_cache=False, # 是否释放推理缓存
                        pin_parameters=True,  # 是否固定参数
                        tp_gather_partition_size=8,  # 张量并行聚合分区大小
                        max_out_tokens=512, # 最大输出tokens数
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False, # 是否开启混合精度的lora
                        tb_path="",
                        tb_name=""):

    device = "cpu" if offload else "none"
    # 设置offload设备
    # 如果offload为True,则将offload设备设置为"cpu",否则设置为"none"

    # 设置数据类型和相关配置
    if dtype == "fp16":
        # 如果数据类型为fp16(半精度浮点数)
        data_type = "fp16"
        # 设置dtype_config,启用混合精度训练,并设置损失缩放窗口大小为100
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif dtype == "bf16":
        # 如果数据类型为bf16(Brain浮点数)
        data_type = "bfloat16"
        # 设置dtype_config,启用混合精度训练
        dtype_config = {"enabled": True}

    # 设置ZeRO优化器相关配置    
    zero_opt_dict = {
        # stage表示ZeRO优化的阶段,可选值为0、1、2、3
        "stage": stage,
        "offload_param": {
            # offload_param设置参数offload的设备, 上面已经设置为cpu了
            "device": device
        },
        "offload_optimizer": {
            # offload_optimizer设置优化器offload的设备
            "device": device
        },
        # stage3参数持久化阈值
        "stage3_param_persistence_threshold": 1e4,
        # stage3最大活跃参数数量
        "stage3_max_live_parameters": 3e7,
        # stage3预取桶大小
        "stage3_prefetch_bucket_size": 3e7,
        # 是否启用内存高效线性层
        "memory_efficient_linear": False
    }

    # 如果启用了混合精度LoRA
    if enable_mixed_precision_lora:
        # 设置零量化非可训练权重
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        # 如果世界大小(进程数)不等于设备数量
        if dist.get_world_size() != get_accelerator().device_count():
            # 设置ZeRO高性能优化分区大小
            zero_opt_dict["zero_hpz_partition_size"] = get_accelerator(
            ).device_count()

    # 返回DeepSpeed训练配置字典        
    return {
        # 训练批次大小(全体)
        "train_batch_size": GLOBAL_BATCH_SIZE,
        # 每个GPU的训练微批次大小
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        # 每隔多少步打印一次日志
        "steps_per_print": 10,
        # ZeRO优化器配置
        "zero_optimization": zero_opt_dict,
        # 数据类型和相关配置
        data_type: dtype_config,
        # 梯度裁剪阈值
        "gradient_clipping": 1.0,
        # 是否预缩放梯度
        "prescale_gradients": False,
        # 是否启用Wall Clock Breakdown
        "wall_clock_breakdown": False,
        # 混合引擎配置,RLHF才会开启,train与eval模式高性能
        "hybrid_engine": {
            # 是否启用混合引擎
            "enabled": enable_hybrid_engine,
            # 最大输出tokens数
            "max_out_tokens": max_out_tokens,
            # 推理张量并行大小
            "inference_tp_size": inference_tp_size,
            # 是否释放推理缓存
            "release_inference_cache": release_inference_cache,
            # 是否固定参数
            "pin_parameters": pin_parameters,
            # 张量并行聚合分区大小
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        # TensorBoard配置
        "tensorboard": {
            # 是否启用TensorBoard
            "enabled": enable_tensorboard,
            # TensorBoard输出路径
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            # TensorBoard日志名称
            "job_name": f"{tb_name}_tensorboard"
        }
    }



"""
在这段代码中,主要使用了以下技术:
1. **DeepSpeed配置设置**
   - `get_eval_ds_config`函数用于获取DeepSpeed评估配置
   - 根据输入参数`offload`和`dtype`设置相应的配置选项

2. **CPU Offload**
   - 当`offload`为`True`时,将一些张量从GPU卸载到CPU,以节省GPU内存
   - 在这种情况下,设置`device`为`"cpu"`

3. **混合精度训练**
   - 根据`dtype`参数,设置使用`FP16`或`BF16`数据类型进行混合精度训练
   - 在配置字典中启用相应的数据类型选项

4. **ZeRO优化器配置**
   - 设置`ZeRO`优化器的阶段、参数持久性阈值、卸载设备类型等配置

5. **其他选项**
   - 设置全局批次大小、微批次大小、梯度裁剪阈值等训练参数
   - 控制是否启用梯度预缩放、wall clock breakdown等选项

   这段代码的主要目的是根据给定的参数,生成DeepSpeed评估配置字典。通过配置不同的选项,可以控制混合精度训练、CPU Offload、ZeRO优化器行为等方面,以优化模型评估过程。同时,它还设置了一些训练相关的参数,如批次大小、梯度裁剪阈值等,为模型评估做好准备。
"""
def get_eval_ds_config(offload, dtype, stage=0):
    # 根据offload参数确定设备类型
    # 如果offload为True,表示将一些张量从GPU卸载到CPU,以节省GPU内存
    # 此时设备类型为"cpu"
    # 否则,设备类型为"none"
    device = "cpu" if offload else "none"

    # 根据dtype参数设置数据类型和相关配置
    if dtype == "fp16":
        # 如果dtype为"fp16",表示使用16位浮点数
        data_type = "fp16"
        # 设置相关配置
        dtype_config = {
            "enabled": True, # 启用16位浮点数
        }
    elif dtype == "bf16":
        # 如果dtype为"bf16",表示使用16位大小的Bfloat16数据类型
        data_type = "bfloat16"
        # 设置相关配置
        dtype_config = {"enabled": True}  # 启用Bfloat16数据类型

    # 设置ZeRO优化器的配置
    zero_opt_dict = {
        "stage": stage, # ZeRO优化器的阶段,0/1/2/3表示不同的优化策略
        "stage3_param_persistence_threshold": 1e4, # 阶段3参数持久性阈值
        "offload_param": {
            "device": device  # 指定卸载设备类型
        },
        "memory_efficient_linear": False  # 不启用内存高效线性层
    }

    # 返回DeepSpeed评估配置字典
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,  # 全局批次大小
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,  # 每个GPU的微批次大小
        "steps_per_print": 10,  # 每10步打印一次日志
        "zero_optimization": zero_opt_dict,  # ZeRO优化器配置
        data_type: dtype_config,  # 数据类型配置
        "gradient_clipping": 1.0,  # 梯度裁剪阈值
        "prescale_gradients": False,  # 不启用梯度预缩放
        "wall_clock_breakdown": False  # 不启用wall clock breakdown
    }
