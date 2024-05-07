#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""

engine = DeepSpeedRLHFEngine(actor_model_name_or_path=actor_model_name_or_path,
                             critic_model_name_or_path=critic_model_name_or_path,
                             tokenizer=tokenizer,
                             args=args)
trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
    out = trainer.generate_experience(prompt_batch)
    actor_loss, critic_loss = trainer.train_rlhf(out)

"""
import argparse
import os
import random
import time
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from transformers import (
    SchedulerType,
    default_data_collator,
)

import deepspeed

from dschat.rlhf.ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from dschat.rlhf.rlhf_engine import DeepSpeedRLHFEngine
from dschat.utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer, \
    ExponentialMovingAverage
from dschat.utils.module.lora import convert_lora_to_linear_layer
from dschat.utils.perf import print_throughput_step3
from deepspeed.accelerator import get_accelerator

writer = None


def parse_args():
    global writer
    parser = argparse.ArgumentParser(
        description="(Step 3) RLHF training arguments")

    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='2,4,4',
        help=
        'Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` '
        'will use 60%% of data for phase 1, 20%% for phase 2 and 20%% for phase 3.'
    )
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument("--unsup_coef",
                        type=float,
                        default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,  # OPT模型前面会添加一个符号, 当然也有其他的语言模型没有, 所以提供了一个选项, 这个值通常是0或1
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )
    parser.add_argument(
        "--per_device_generation_batch_size",
        type=int,
        default=16,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
    parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=16,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument("--generation_batches",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")  # ppo重要性采样后历史数据使用的次数
    parser.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--critic_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",  # checkpoint的保存点
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # DeepSpeed
    # 目前--enable_hybrid_engine会报一个错误, 移除 --enable-hybric-engine, 加上 --offload_reference_model 减少内存占用
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation. This makes generation slower but requires less memory."
    )
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size."
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature."
    )
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Critic model (and reward).')
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        '--critic_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Critic model.')
    parser.add_argument(
        "--actor_dropout",
        type=float,
        default=None,
        help="If actor dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the actor model."
    )
    parser.add_argument(
        "--critic_dropout",
        type=float,
        default=None,
        help="If critic dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the critic model."
    )
    ## LoRA for efficient training setting
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument("--critic_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--critic_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--actor_lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial actor LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial critic LoRA learning rate (after the potential warmup period) to use."
    )
    ## Make EMA as an optional feature
    parser.add_argument('--enable_ema',
                        action='store_true',
                        help='Enable EMA checkpoint for the model.')
    ## Mixed Precision ZeRO++
    parser.add_argument(
        '--enable_mixed_precision_lora',
        action='store_true',
        help='Enable Mixed Precision ZeRO++ for training and generation.')
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.'
        'This applies for both actor and critic models.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step3_tensorboard")
    #TODO 上面这个参数在 1/2/3阶段,均需要打开,为了保证 tokenizer 可以准确的分词 <|endoftext|> ,也就是说bash后需要加上 --add_eot_token
    # OPT 模型是无法分词 <|endoftext|>的
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    ## Actor/critic model overflow alignment
    parser.add_argument(
        '--align_overflow',
        action='store_true',
        help='Align loss scale overflow between actor and critic')
    ## Print actor model answers during training
    parser.add_argument('--print_answers',
                        action='store_true',
                        help='Print prompt and answers during training')
    parser.add_argument(
        "--print_answers_interval",
        type=int,
        default=1,
        help="If --print_answers enabled, controls the printing interval.")
    ## Testing
    parser.add_argument(
        '--enable_test_mode',
        action='store_true',
        help=
        'Enable a testing mode that terminates training based on args.test_stop_step'
    )
    parser.add_argument(
        "--test_stop_step",
        type=int,
        default=0,
        help=
        "Training non-overflow step at which to terminate training during testing."
    )

    # 1. deepspeed.add_config_arguments(parser) 函数用于向命令行参数解析器 (parser) 添加 DeepSpeed 相关的配置参数。
    # 2. 这允许用户在命令行中指定 DeepSpeed 的各种配置选项,如优化策略、混合精度、检查点等。
    parser = deepspeed.add_config_arguments(parser)
    # 3. parser.parse_args() 函数用于从命令行参数中解析出配置值,并将它们存储在 args 对象中。
    args = parser.parse_args()
    # 4. 如果启用了 TensorBoard,则打印 TensorBoard 日志的保存路径,并创建一个 SummaryWriter 实例。
    # 5. TensorBoard 是一种可视化工具,用于监控和可视化模型训练过程,包括损失函数、学习率、梯度等指标。
    # 6. SummaryWriter 是 TensorBoard 的一个核心组件,用于将训练数据写入日志文件,供 TensorBoard 可视化。
    if args.enable_tensorboard:
        print(
            f"Tensorboard logs going to: {args.tensorboard_path}/step3_tensorboard_logs"
        )
        writer = SummaryWriter(
            f"{args.tensorboard_path}/step3_tensorboard_logs")

    # Validate settings
    # 7. 这部分代码用于验证配置设置的有效性。
    # 8. 如果 inference_tp_size (推理时张量并行的大小) 大于 1,则要求 actor_zero_stage (生成模型的 ZeRO 优化阶段) 必须为 3。
    # 9. ZeRO 阶段 3 是 DeepSpeed 中最高级别的优化策略,支持张量分片 (Tensor Sharding),可以训练更大的模型。
    # 10. 张量分片是一种将模型权重分割到多个 GPU 上的技术,从而减少单个 GPU 上的内存需求。
    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    if args.actor_zero_stage == 2 and args.critic_zero_stage == 2 and args.enable_hybrid_engine and args.offload and args.actor_lora_dim == 0:
        # 11. 这部分代码检查了一组特定的配置组合是否有效。
        # 12. 如果生成模型和奖励模型都使用 ZeRO 阶段 2,同时启用了混合引擎、CPU Offload,但没有使用 LoRA 技术,则会引发一个 ValueError 异常。
        # 13. 这是因为该配置组合在训练时可能会导致不稳定的行为,目前不受支持。
        # 14. LoRA (Low-Rank Adaptation) 是一种用于微调大型预训练语言模型的技术,它通过引入低秩投影来修改模型权重,从而显著降低所需的参数数量。
        # 15. 在这种情况下,建议使用 LoRA 技术或调整其他配置选项以确保训练稳定性。
        raise ValueError(
            "The combination of [actor_zero_stage==2, critic_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!"
        )

    return args

# 1. args 参数是一个包含各种配置选项的命名空间对象。
# 2. tokenizer 参数是一个用于文本标记化和解码的 Tokenizer 对象。
# 3. train_phase 参数指定了当前的训练阶段,默认为 3
def create_datasets(args, tokenizer, train_phase=3):
    # 4. 检查是否启用了无监督训练,根据 args.unsupervised_dataset_name 和 args.unsupervised_dataset_config_name 参数。
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    # 5. 调用 create_prompt_dataset 函数创建提示训练数据集。
    # create_prompt_dataset 是一个自定义函数,用于从指定的数据路径和配置中创建提示训练数据集。
    prompt_train_dataset, _ = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_prompt_seq_len)
    # 6. 如果启用了无监督训练,则调用 get_unsupervised_data 函数创建无监督训练数据集。
    # get_unsupervised_data 是一个自定义函数,用于获取无监督训练数据。
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    # 7. 创建数据加载器和采样器。
    # DataCollatorRLHF 是一个自定义的数据收集器,用于格式化 RLHF 训练所需的数据。
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size)
    # 8. 根据是否使用分布式训练,创建不同类型的采样器。
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    # 9. 创建提示训练数据加载器。
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_generation_batch_size)
    # 10. 如果启用了无监督训练,则创建无监督训练数据加载器。
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_generation_batch_size)
    else:
        # 11. 如果未启用无监督训练,则创建一个虚拟的无监督训练数据加载器。
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader
    # 12. 计算每个 epoch 的更新步数和总迭代次数。
    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_generation_batch_size / args.per_device_training_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    # 总迭代次数。
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)
    # 13. 返回提示训练数据加载器、无监督训练数据加载器和总迭代次数。
    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters


def main():
    # 1. 解析命令行参数,获取训练所需的配置信息。
    args = parse_args()
    # 2. 根据是否使用分布式训练,设置设备并初始化分布式后端。
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    # 3. 获取当前进程的全局排名。
    args.global_rank = torch.distributed.get_rank()

    # print args, 仅使用序号为0的进程来打印
    print_rank_0("***** Parmeters *****".rjust(62), args.global_rank)
    arguments = vars(args)
    for key, value in arguments.items():
        print_rank_0(f"{key}".rjust(50)+ f": {value}", args.global_rank)
        
    # 4. 根据是否启用无监督训练,调整梯度累积步数。
    # 如果启用无监督训练,则将梯度累积步数加倍,以保持总梯度的稳定性。
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    # 5. 设置随机种子,确保实验的可重复性。
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    # 6. 加载合适的 tokenizer,是否设置额外的特殊 token。
    # load_hf_tokenizer 是一个自定义函数,用于加载匹配模型的 tokenizer,并根据模型家族设置填充 token。
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.actor_model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    # 7. 创建提示训练数据集、无监督训练数据集和总迭代次数。
    # create_datasets 是一个自定义函数,用于创建训练所需的数据集和数据加载器。
    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3)

    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    # 8. 创建 RLHF 引擎,负责模型的创建、检查点加载、初始化 DeepSpeed 相关组件。
    # DeepSpeedRLHFEngine 是一个自定义的类,用于管理 RLHF 训练的各个组件。
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)

    # Mixed Precision ZeRO++
    # 9. 如果启用了混合精度 LoRA,则量化非可训练参数,以减少内存占用。
    # 混合精度训练和 LoRA 是提高模型训练效率的两种技术,可以结合使用。
    if args.enable_mixed_precision_lora:
        assert args.actor_lora_dim > 0, "Mixed Precision LoRA requires LoRA to be enabled"
        assert args.actor_zero_stage == 3, "Mixed Precision LoRA requires Zero stage 3"
        rlhf_engine.actor.optimizer.quantize_nontrainable_params()
        print_rank_0("Mixed Precision ZeRO++ enabled")

    # 10. 根据是否启用无监督训练,创建不同类型的 PPO 训练器。
    # DeepSpeedPPOTrainerUnsupervised 和 DeepSpeedPPOTrainer 是自定义的训练器类,用于 RLHF 训练。
    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    # 11. 创建用于生成经验和无监督训练的小型数据集。
    # MiniDataset 是一个自定义的数据集类,用于存储和管理训练所需的小型数据批次。
    exp_mini_dataset = MiniDataset(args.generation_batches,
                                   args.per_device_training_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batches,
                                     args.per_device_training_batch_size)

    # Train!
    # 12. 打印训练信息,包括总迭代次数等。
    print_rank_0(
        f"***** Running training (total_iters={num_total_iters}) *****",
        args.global_rank)

    # 13. 初始化一些训练相关的变量,如非溢出步数、平均奖励等。
    non_overflow_step_count = 0
    step_average_reward = 0.
    ema_reward_score = ExponentialMovingAverage()

    # 14. 开始训练过程。
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            args.global_rank)
        for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)):
            # 15. 将提示训练批次移动到设备上。
            batch_prompt = to_device(batch_prompt, device)

            # prompts = batch_prompt['prompt']
            # length = prompts.size(-1)
            # if length > args.max_prompt_seq_len:
            #     prompts = prompts[:, length - args.max_prompt_seq_len:]
            #     raise ValueError("Prompt length is too long")
            # 16. 生成训练经验。
            out = trainer.generate_experience(batch_prompt['prompt'],
                                              batch_prompt['prompt_att_mask'],
                                              step)
            # 17. 记录训练开始时间。
            training_start = time.time()
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                # 19. 如果没有无监督训练数据,则创建一个虚拟的无监督训练数据集。
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_generation_batch_size])
            # 20. 将生成的训练经验添加到小型数据集中。
            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                # 21. 执行多轮 PPO 训练,更新角色和批评模型。
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                average_reward = 0
                # 22. 如果启用了梯度检查点,则启用梯度检查点功能。
                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()
                # 23. 使用训练经验和无监督数据训练角色和批评模型。
                for ppo_ep in range(args.ppo_epochs):
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                        actor_loss_sum += actor_loss.item()
                        critic_loss_sum += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()
                        # 24. 如果启用了无监督训练,则训练无监督模型。
                        if unsupervised_training_enabled:
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)
                            unsup_loss_sum += unsup_loss.item()

                        inner_iter += 1
                        if args.enable_ema:
                            # 25. 如果启用了指数移动平均,则更新角色模型的 EMA。
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)
                    # 26. 随机打乱训练经验和无监督数据集,以增加数据多样性。
                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)
                 # 27. 记录训练结束时间,并计算端到端训练时间。
                end = time.time()
                training_time = end - training_start
                e2e_time = training_time + trainer.generate_time * args.generation_batches  # it is an approximation, we did not include, e.g., rw forward time etc
                # 28. 打印训练指标,包括损失、奖励等。
                print_rank_0(
                    f'Epoch: {epoch} | Step: {step} | PPO Epoch: {ppo_ep+1} | Actor Loss: {actor_loss_sum/inner_iter} | Critic Loss: {critic_loss_sum/inner_iter} | Unsupervised Loss: {unsup_loss_sum/inner_iter}',
                    args.global_rank)
                print_throughput_step3(rlhf_engine.actor.module,
                                       rlhf_engine.critic, args, e2e_time,
                                       trainer.generate_time, training_time,
                                       args.global_rank)
                # 29. 计算平均奖励,并更新指数移动平均奖励分数。
                average_reward = get_all_reduce_mean(average_reward).item()
                step_average_reward += average_reward / args.gradient_accumulation_steps_actor
                if (step + 1) % args.gradient_accumulation_steps_actor == 0:
                    ema_reward_score.update(step_average_reward)
                    step_average_reward = 0.
                # 30. 打印当前的平均奖励分数和指数移动平均奖励分数。
                print_rank_0(
                    f"Average reward score: {average_reward/inner_iter} | EMA reward score: {ema_reward_score.get()}",
                    args.global_rank)
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)
                # 31. 如果启用了 TensorBoard,则记录训练过程中的指标。
                if args.enable_tensorboard and torch.distributed.get_rank(
                ) == 0:
                    writer.add_scalar('reward',
                                      average_reward / inner_iter,
                                      global_step=step)
                    writer.add_scalar('actor_loss',
                                      actor_loss.item(),
                                      global_step=step)
                    writer.add_scalar('actor_loss_sum',
                                      actor_loss_sum,
                                      global_step=step)
                    writer.add_scalar('critic_loss',
                                      critic_loss.item(),
                                      global_step=step)
                    writer.add_scalar('critic_loss_sum',
                                      critic_loss_sum,
                                      global_step=step)
                    writer.flush()
            # 32. 如果启用了梯度检查点,则禁用梯度检查点功能。
            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()
            # 33. 获取角色和批评模型的梯度溢出状态。
            actor_overflow, critic_overflow = trainer.get_overflow()
            # 34. 如果没有发生梯度溢出,则更新非溢出步数计数器。
            if not actor_overflow and not critic_overflow:
                non_overflow_step_count += 1
            # 35. 如果启用了测试模式,且达到了测试停止步数,则退出训练。
            if args.enable_test_mode and non_overflow_step_count == args.test_stop_step:
                break
        # 36. 如果启用了测试模式,则退出训练。
        if args.enable_test_mode:
            break
    # 37. 保存训练好的模型。
    if args.output_dir is not None:
        print_rank_0('saving model ...')
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)
        if args.enable_ema:
            rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                rlhf_engine.actor_ema)

        if torch.distributed.get_rank() == 0:
            save_hf_format(rlhf_engine.actor,
                           tokenizer,
                           args,
                           sub_folder='actor')
            save_hf_format(rlhf_engine.critic,
                           tokenizer,
                           args,
                           sub_folder='critic')
            if args.enable_ema:
                save_hf_format(rlhf_engine.actor_ema,
                               tokenizer,
                               args,
                               sub_folder='actor_ema')

        # 38. 如果批评模型使用了 ZeRO 阶段 3 优化,则使用 save_zero_three_model 函数保存批评模型。
            # ZeRO 是一种分布式训练优化策略,可以帮助减少内存消耗和提高训练效率。
            # 阶段 3 是最激进的优化策略,会将模型参数进行更细粒度的分片。
        if args.actor_zero_stage == 3:
            save_zero_three_model(rlhf_engine.actor,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'actor'),
                                  zero_stage=args.actor_zero_stage)
            if args.enable_ema:
                save_zero_three_model(rlhf_engine.actor_ema,
                                      global_rank=args.global_rank,
                                      save_dir=os.path.join(
                                          args.output_dir, 'actor_ema'),
                                      zero_stage=args.actor_zero_stage)
        if args.critic_zero_stage == 3:
            save_zero_three_model(rlhf_engine.critic,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'critic'),
                                  zero_stage=args.critic_zero_stage)


if __name__ == "__main__":
    main()
