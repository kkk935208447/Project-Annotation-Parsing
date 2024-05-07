# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


# This function can be used to print throughput for Step 1 and 2 only
"""
技术解释：
此函数用于计算和打印训练过程中的吞吐量指标,包括模型参数数量、延迟、TFLOPS、每秒样本数、每个序列的时间、批大小和序列长度。
它利用了分布式训练的概念,只在特定的进程上打印指标,避免了多个进程重复打印。
函数使用了HuggingFace模型配置,获取模型的层数、隐藏层大小和词表大小等参数。
它考虑了是否使用梯度检查点和LoRA等优化技术,并相应地调整了计算系数。
函数通过计算FLOPS和TFLOPS来估计模型的计算能力和吞吐量。
最终,它格式化并打印出各种指标,供用户监控训练进度和性能。

参数/函数解释：
hf_model是HuggingFace预训练模型的实例。
args是命令行参数的集合,包含了一些重要的训练配置,如max_seq_len(序列长度)、per_device_train_batch_size(批大小)、gradient_checkpointing(是否使用梯度检查点)和lora_dim(LoRA维度)等。
e2e_time是端到端时间,用于计算每秒样本数。
rank是当前进程的排名,用于确定是否打印指标。
get_hf_configs是一个获取HuggingFace模型配置的函数。
torch.distributed.get_world_size()是PyTorch分布式包中的函数,用于获取分布式训练中的GPU数量。
calculate_flops是一个计算FLOPS的函数,它考虑了梯度检查点、批大小、序列长度和模型配置等因素。

技术应用原因：
计算和打印吞吐量指标可以帮助用户监控模型训练的进度和性能,了解模型的计算能力和训练效率。
利用分布式训练可以加速训练过程,但也需要在特定进程上打印指标,避免重复打印。
获取HuggingFace模型配置是为了计算FLOPS和其他指标,这些配置对于模型的计算和内存需求至关重要。
考虑梯度检查点和LoRA等优化技术是为了更准确地估计模型的计算量和内存需求。
计算FLOPS和TFLOPS可以反映模型的计算复杂度和性能,有助于评估和优化模型。
打印各种指标可以为用户提供全面的训练信息,帮助调试和优化模型。
"""
# This function can be used to print throughput for Step 1 and 2 only
# 1. 此函数用于打印Step 1和Step 2的吞吐量
def print_throughput(hf_model, args, e2e_time, rank=0):
    
    if rank <= 0: # 2. 仅在rank <= 0的进程上打印吞吐量
        # 3. 从HuggingFace模型配置中获取层数、隐藏层大小和词表大小
        hf_config = hf_model.config
        num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)

        # 4. 获取分布式训练的GPU数量
        gpus_per_model = torch.distributed.get_world_size()
        # 5. 从命令行参数中获取序列长度和批大小
        seq_length = args.max_seq_len
        batch_size = args.per_device_train_batch_size
        # 6. 计算每秒样本数
        samples_per_second = batch_size / e2e_time
        # 7. 根据是否使用梯度检查点确定一个系数
        checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3
        # 8. 如果使用LoRA,根据LoRA维度调整系数
        if args.lora_dim > 0:
            k = args.lora_dim * 2 / hidden_size
            checkpoint_activations_factor -= (1 - k)
        # 9. 计算模型参数总数
        hf_model._num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in hf_model.parameters()
        ])
        # 10. 将模型参数总数量纲转换为B(十亿)
        params_in_billions = hf_model._num_params / (1e9)

        # Megatron paper's formula to calculate training flops
        # 11. 计算训练FLOPS
        train_flops_per_iteration = calculate_flops(
            checkpoint_activations_factor, batch_size, seq_length, hf_config)

        # 12. 计算训练TFLOPS
        train_tflops = train_flops_per_iteration / (e2e_time * gpus_per_model *
                                                    (10**12))

        # 13. 格式化参数字符串
        param_string = f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"
        # 14. 打印吞吐量相关指标
        print(
            f"Model Parameters: {param_string}, Latency: {e2e_time:.2f}s, TFLOPs: {train_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Sequence Length: {seq_length}"
        )

"""
该函数的主要目的是计算和打印RLHF训练中Actor模型和Critic模型的各种性能指标,包括:
    端到端延迟、TFLOPS和每秒处理的样本数
    生成(推理)阶段的延迟、每个token的延迟、TFLOPS和带宽
    训练阶段的延迟和TFLOPS
    Actor模型和Critic模型的参数数量
    这些指标可以用于评估模型的性能和效率,并帮助优化训练过程。该函数还使用了DeepSpeed库和一些公式来计算FLOPS和其他指标,这些技术通常用于加速大型模型的训练。
"""
# Enhanced version of the function above that provides calculations and printing for Step 3
def print_throughput_step3(actor_model,  # 1. actor_model是一个Hugging Face模型,代表了RLHF训练中的Actor模型,用于生成文本
                           critic_model, # 2. critic_model是一个DeepSpeed Engine对象,包含了RLHF训练中的Critic模型(Reward模型),用于评估生成文本的质量
                           args,  # 3. args是一个命名空间对象,包含了各种训练参数和配置
                           e2e_time, # 4. e2e_time是端到端的总运行时间(包括生成和训练阶段)
                           gen_exp_time, # 5. gen_exp_time是生成阶段(推理)的运行时间
                           train_time,  # 6. train_time是训练阶段的运行时间
                           rank=0):  # 7. rank是当前进程的等级,默认为0(主进程)
    # 8. 只在主进程上打印输出
    if rank <= 0:  
        # Actor model passed here is a HF model.
        # 9. actor_model是一个Hugging Face模型,因此可以从actor_model.config获取模型配置
        actor_hf_config = actor_model.config
        # Critic model passed here is  a DeepSpeed Engine. The module inside is the Reward model (that wraps a HF model).
        # 10. critic_model是一个DeepSpeed Engine对象,需要从critic_model.module获取包装的Hugging Face模型配置
        critic_hf_config = critic_model.module.config

        # 11. 获取Actor模型和Critic模型的配置信息,包括层数、隐藏层大小和词表大小
        actor_num_layers, actor_hidden_size, actor_vocab_size = get_hf_configs(
            actor_hf_config)
        critic_num_layers, critic_hidden_size, critic_vocab_size = get_hf_configs(
            critic_hf_config)

        # 12. 获取每个模型使用的GPU数量
        gpus_per_model = torch.distributed.get_world_size()
        # 13. 计算序列长度,即prompt长度加上生成文本的最大长度
        seq_length = args.max_answer_seq_len + args.max_prompt_seq_len
        # 14. 计算批量大小,取决于是否使用无监督数据集
        batch_size = args.per_device_generation_batch_size * args.generation_batches * args.ppo_epochs * gpus_per_model * 1 if args.unsupervised_dataset_name is None else 2
        # 15. 计算每秒处理的样本数
        samples_per_second = batch_size / e2e_time

        # 16. 计算Actor模型和Critic模型的activation checkpoint因子
        # 该因子用于估计模型的计算量,取决于是否使用梯度检查点和LoRA技术
        actor_checkpoint_activations_factor = 4 if args.actor_gradient_checkpointing else 3
        critic_checkpoint_activations_factor = 4 if args.critic_gradient_checkpointing else 3
        if args.actor_lora_dim > 0:
            k = args.actor_lora_dim * 2 / actor_hidden_size
            actor_checkpoint_activations_factor -= (1 - k)
        if args.critic_lora_dim > 0:
            k = args.critic_lora_dim * 2 / critic_hidden_size
            critic_checkpoint_activations_factor -= (1 - k)

        # 17. 计算Actor模型和Critic模型的参数数量(以十亿为单位)
        actor_model._num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in actor_model.parameters()
        ])
        actor_params_in_billions = actor_model._num_params / (1e9)

        critic_model._num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in critic_model.parameters()
        ])
        critic_params_in_billions = critic_model._num_params / (1e9)

        # Megatron paper's formula to calculate training flops
        # 18. 使用Megatron论文中的公式计算Actor模型和Critic模型的训练FLOPS
        actor_train_flops_per_iteration = calculate_flops(
            actor_checkpoint_activations_factor, batch_size, seq_length,
            actor_hf_config)
        critic_train_flops_per_iteration = calculate_flops(
            critic_checkpoint_activations_factor, batch_size, seq_length,
            critic_hf_config)

        # 19. 计算训练的TFLOPS(每秒浮点运算数量,以万亿次为单位)
        total_train_flops = actor_train_flops_per_iteration + critic_train_flops_per_iteration
        train_tflops = total_train_flops / (train_time * gpus_per_model *
                                            (10**12))

        gen_bs = args.per_device_generation_batch_size * gpus_per_model

        # Modified formula for calculating flops in the forward pass only
        # 20. 使用修改后的公式计算生成(推理)阶段的FLOPS
        gen_flops_per_iteration = (
            24 * gen_bs * seq_length * actor_num_layers *
            (actor_hidden_size**2)) * (
                1.0 + (seq_length / (6.0 * actor_hidden_size)) +
                (actor_vocab_size /
                 (16.0 * actor_num_layers * actor_hidden_size)))

        # 21. 计算生成的TFLOPS(每秒浮点运算数量,以万亿次为单位)
        gen_tflops = gen_flops_per_iteration / (gen_exp_time * gpus_per_model *
                                                (10**12))

        # 22. 根据Actor模型的数据类型确定每个参数占用的字节数
        if actor_hf_config.torch_dtype == torch.float16:
            num_bytes = 2
        elif actor_hf_config.torch_dtype == torch.float32:
            num_bytes = 4
        else:
            num_bytes = -1

        # 23. 计算每个token的延迟(以毫秒为单位)
        pertok_lat = gen_exp_time / args.max_answer_seq_len
        # 24. 计算生成的带宽(以GB/秒为单位)
        gen_bw = 1 / pertok_lat * actor_model._num_params * num_bytes / 1e9
        # 25. 计算总的FLOPS,包括训练和生成阶段
        total_flops_per_iteration = total_train_flops + gen_flops_per_iteration * args.generation_batches
        # 26. 计算总的TFLOPS
        total_tflops = total_flops_per_iteration / (e2e_time * gpus_per_model *
                                                    (10**12))
        
        # 27. 打印端到端性能指标
        print(
            f"End-to-End => Latency: {e2e_time:.2f}s, TFLOPs: {total_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Total Seq. Length: {seq_length}"
        )
        # 28. 打印生成(推理)性能指标
        print(
            f"Generation => Latency: {gen_exp_time:.2f}s, Per-token Latency {pertok_lat*1000:.2f} ms, TFLOPs: {gen_tflops:.2f}, BW: {gen_bw if num_bytes > 0 else num_bytes:.2f} GB/sec, Answer Seq. Length: {args.max_answer_seq_len}"
        )
        # 29. 打印训练性能指标
        print(
            f"Training   => Latency: {train_time:.2f}s, TFLOPs: {train_tflops:.2f}"
        )
        # 30. 打印actor模型与critic模型的参数量
        actor_param_string = f"{actor_params_in_billions:.3f} B" if actor_params_in_billions != 0 else "NA"
        critic_param_string = f"{critic_params_in_billions:.3f} B" if critic_params_in_billions != 0 else "NA"
        print(
            f"Actor Model Parameters => {actor_param_string}, Critic Model Parameters => {critic_param_string}"
        )


"""
该函数使用了Megatron-LM论文中的公式来计算每次迭代的FLOPS(浮点运算次数)。
公式考虑了多个因素,包括批量大小、序列长度、层数、隐藏层大小、词表大小和激活值重计算开销。
该公式被广泛用于估计transformer模型的计算开销,有助于评估模型的性能和优化策略。
"""
# Helper function to calculate FLOPs using the Megatron-LM paper's formula
# calculate_flops函数可用于估计模型的计算开销,从而评估不同优化策略(如ZeRO优化)的效果。
def calculate_flops(checkpoint_activations_factor, # 1. checkpoint_activations_factor是一个浮点数,用于计算激活值重计算的开销
                    batch_size,  # 2. batch_size是整数,表示批量大小
                    seq_length,  # 3. seq_length是整数,表示序列长度
                    hf_config  # 4. hf_config是Hugging Face模型的配置对象,包含了模型的参数信息
                    ):
    # 5. 从Hugging Face模型配置中获取层数、隐藏层大小和词表大小
    num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)
    # 6. 使用Megatron-LM论文中的公式计算每次迭代的FLOPS
    # 该公式考虑了层数、隐藏层大小、词表大小、序列长度和激活值重计算开销等因素
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size *
                           seq_length * num_layers * (hidden_size**2)) * (
                               1.0 + (seq_length / (6.0 * hidden_size)) +
                               (vocab_size /
                                (16.0 * num_layers * hidden_size)))
    return flops_per_iteration

# get_hf_configs函数则用于从Hugging Face模型中提取关键参数,以便与DeepSpeed集成。
"""
该函数用于从Hugging Face模型的配置对象中获取层数、隐藏层大小和词表大小。
由于不同的Hugging Face模型可能使用不同的属性名称,因此该函数尝试多种属性名来获取这些参数。
最后会断言这些参数都被成功获取,否则会引发异常。
"""
# 7. 该函数用于从Hugging Face模型配置中获取层数、隐藏层大小和词表大小
def get_hf_configs(hf_config):
    # 8. 尝试从配置对象中获取层数,不同的Hugging Face模型可能使用不同的属性名称
    num_layers = getattr(hf_config, "num_hidden_layers",
                         getattr(hf_config, "n_layer", None))
    # 9. 尝试从配置对象中获取隐藏层大小,不同的Hugging Face模型可能使用不同的属性名称
    hidden_size = getattr(hf_config, "hidden_size",
                          getattr(hf_config, "n_embd", None))
    # 10. 尝试从配置对象中获取词表大小
    vocab_size = getattr(hf_config, "vocab_size", None)
    # 11. 确保层数、隐藏层大小和词表大小都被成功获取
    assert all(
        (num_layers, hidden_size, vocab_size)
    ), "Could not determine number of layers, hidden size, and vocab size of the model"

    return num_layers, hidden_size, vocab_size
