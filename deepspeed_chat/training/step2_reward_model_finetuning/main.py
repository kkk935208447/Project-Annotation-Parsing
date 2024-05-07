#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator

from dschat.utils.model.model_utils import create_critic_model  # 创建critic模型
from dschat.utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from dschat.utils.ds_utils import get_train_ds_config # 构建 deepspeed init 参数
from dschat.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',   # 只要1-3个阶段随机种子一致, 仅仅加载一阶段缓存的数据
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 20%% of data for phase 1, 40%% for phase 2'
                        'and 40%% for phase 3.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    # OPT模型在输入的开头有固定数量（1个）的填充标记。
    # "我们在其他模型中没有看到这一点，但目前将其保留为一个选项。" 该值一般是 0 或 1
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", # 训练epoch数
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
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
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",  # checkpoint保存
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )

    # Evaluation
    parser.add_argument("--eval_interval", # 间隔评估
                        type=int,
                        default=0,
                        help="If > 0, perform evaluation at this interval")
    parser.add_argument("--eval_iters",
                        type=int,
                        default=100,
                        help="Maximum evaluation iterations")
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',   # 计算loss是否用fp32
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.')

    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step2_tensorboard")
    #TODO 上面这个参数在 1/2/3阶段,均需要打开,为了保证 tokenizer 可以准确的分词 <|endoftext|> ,也就是说bash后需要加上 --add_eot_token
    # OPT 模型是无法分词 <|endoftext|>的
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

# 代码主要涉及以下关键技术要点:
# DeepSpeed 微批次训练: DeepSpeed 使用了微批次训练策略,将大批次划分为多个小批次,以减少内存占用。每个微批次只进行一次反向传播和优化,需要累积多次梯度更新才能完成一次完整的训练步骤。
# 梯度检查点: 梯度检查点是一种节省内存的技术,可以在一定程度上减少内存占用。但同时也会增加计算开销,需要在内存和计算之间权衡。
# DeepSpeed ZeRO 优化: DeepSpeed 提供了多种 ZeRO 优化策略,用于减少模型参数和激活值的内存占用。不同的优化阶段 (0/1/2/3) 代表了不同的优化程度,阶段越高,优化效果越好,但也需要更多的内存。在使用 ZeRO 优化阶段 3 时,每个 GPU 只拥有模型的一部分,因此需要在每个 GPU 上保存对应的那部分模型。
# LoRA (Low-Rank Adaptation): LoRA 是一种用于微调大型预训练模型的技术,它通过引入低秩投影来修改模型权重,从而显著降低所需的参数数量。如果只优化 LoRA 层的参数,可以进一步减少内存占用。
# CPU Offload: DeepSpeed 支持将一些张量从 GPU 卸载到 CPU,以节省 GPU 内存。
# 混合精度训练: DeepSpeed 支持使用混合精度 (如 FP16) 进行训练,以减少内存占用和加速计算。
def main():
    args = parse_args()

    if args.local_rank == -1:
        # 如果 local_rank 为 -1,表示运行在单机单 GPU 模式下
        # 使用 DeepSpeed 的 get_accelerator().device_name() 函数获取设备名称,并创建 PyTorch 设备对象
        device = torch.device(get_accelerator().device_name())
    else:
        # 如果 local_rank 不为 -1,表示运行在分布式多机多 GPU 模式下
        # 使用 DeepSpeed 的 set_device 函数设置当前进程使用的 GPU 设备
        get_accelerator().set_device(args.local_rank)
         # 创建 PyTorch 设备对象,指定使用的 GPU 编号和进程编号
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # torch.distributed.init_process_group(backend='nccl')  #初始化PyTorch分布式后端,用于同步多GPU之间的通信,注释掉这一行,因为DeepSpeed会自动初始化分布式后端
        # 使用DeepSpeed初始化分布式训练环境
        deepspeed.init_distributed()

    # 3. 获取当前进程的全局排名
    # 在分布式训练中,每个进程都有一个唯一的全局排名,用于确定进程的角色和任务
    args.global_rank = torch.distributed.get_rank()

    # 4. 获取 DeepSpeed 训练配置
    # DeepSpeed提供了多种优化策略,通过配置可以启用不同的优化功能
    # 参数包括:
    # offload: 是否启用CPU offload,将一些张量从GPU卸载到CPU以节省GPU内存
    # dtype: 张量的数据类型,例如torch.float16用于混合精度训练
    # stage: ZeRO优化的阶段,0/1/2/3表示不同的优化策略,阶段越高,优化程度越大,但也需要更多内存,0 表示不优化
    ds_config = get_train_ds_config(
                                    offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    # enable_tensorboard: 是否启用TensorBoard可视化
                                    # tb_path: TensorBoard日志保存路径
                                    # tb_name: TensorBoard运行的名称
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step2_model")
    # 5. 设置训练微批次大小和总批次大小
    # DeepSpeed使用微批次训练策略,将大批次划分为多个小批次,以减少内存占用
    # train_micro_batch_size_per_gpu 指定每个GPU上的微批次大小
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    # train_batch_size 指定总的批次大小,等于每GPU批次大小乘以GPU数量,再乘以梯度累积步数
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    # 6. 如果提供了种子,则设置随机种子
    # 设置随机种子可以确保实验结果的可重复性
    set_random_seed(args.seed)
    torch.distributed.barrier()  # 多进程阻塞同步

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    # add_special_tokens决定是否将该token添加到tokenizer的词表中
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    # Hugging Face提供了一种简便的方式加载tokenizer
    # end_of_conversation_token指定对话结束的特殊token
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    # 8. 创建评分模型
    # 评分模型用于对生成的文本进行打分和排序
    # create_critic_model函数创建了一个基于预训练语言模型的评分模型
    # 参数包括:
    # model_name_or_path: 预训练模型的路径或名称
    # tokenizer: 上面加载的tokenizer
    # ds_config: DeepSpeed配置
    # num_padding_at_beginning: 输入序列前面填充的token数量
    # dropout: Dropout比例,用于正则化
    # zero_stage: DeepSpeed ZeRO优化阶段
    # compute_fp32_loss: 是否使用FP32计算损失函数
    rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   args.num_padding_at_beginning,
                                   dropout=args.dropout,
                                   zero_stage=args.zero_stage,
                                   compute_fp32_loss=args.compute_fp32_loss)

    # Model bigscience/bloom-560m has large variance at ln_f.weight parameter
    # This makes bf16 finetuning hard.
    # In general, since we are replacing the model head, it makes sense to reset
    # the LN that precedes it.
    # 9. 对于特定模型,重置 LN 层的权重和偏置
    # 某些预训练模型可能存在权重初始化不佳的问题,导致训练不稳定
    # 这里针对 bigscience/bloom- 模型,重置了最后一层LayerNorm的权重和偏置
    # 只有在DeepSpeed ZeRO优化阶段3时才需要这一步骤
    # 通过这种方式,可以解决 bigscience/bloom- 预训练模型的权重初始化问题,从而获得更稳定的训练过程。
    # 同时,由于只在 rank 0 的进程中执行了这一操作,因此不会影响分布式训练的效率。
    force_optimize_params = []
    if "bigscience/bloom-" in args.model_name_or_path:
        # 判断是否使用 DeepSpeed ZeRO 优化阶段 3
        zero_init_enabled = (args.zero_stage == 3)
        # 获取模型中最后一层 LayerNorm 的权重和偏置参数
        params = [
            rm_model.rwtranrsformer.ln_f.weight,
            rm_model.rwtranrsformer.ln_f.bias
        ]
        # 使用 DeepSpeed 的 GatheredParameters 上下文管理器
        # 这个上下文管理器用于在 ZeRO 优化阶段 3 下操作模型参数
        # modifier_rank=0 表示只有 rank 0 的进程可以修改参数
        # enabled=zero_init_enabled 表示只在使用 ZeRO 优化阶段 3 时启用这一功能
        with deepspeed.zero.GatheredParameters(params,
                                               modifier_rank=0,
                                               enabled=zero_init_enabled):
            # 只有 rank 0 的进程或者不使用 ZeRO 优化阶段 3 时,才执行以下操作
            if deepspeed.comm.get_rank() == 0 or not zero_init_enabled:
                # 重新初始化最后一层 LayerNorm 的权重为全 1,偏置为全 0
                torch.nn.init.ones_(rm_model.rwtransformer.ln_f.weight)
                torch.nn.init.zeros_(rm_model.rwtransformer.ln_f.bias)
        # 将重新初始化的参数名添加到 force_optimize_params 列表中
        force_optimize_params.extend(
            ['rwtransformer.ln_f.weight', 'rwtransformer.ln_f.bias'])

    # 10. 如果启用了 LoRA,则将线性层转换为 LoRA 层
    # LoRA (Low-Rank Adaptation) 是一种用于微调大型预训练模型的技术
    # 它通过引入低秩投影来修改模型权重,从而显著降低所需的参数数量
    # 参数包括:
    # lora_module_name: 需要应用LoRA的模块名称
    # lora_dim: LoRA层的低秩投影维度,决定了引入的新参数数量
    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        # 如果只优化LoRA层的参数,可以进一步减少内存占用
        if args.only_optimize_lora:
            # 将LoRA层中的权重参数添加到force_optimize_params列表中
            force_optimize_params.append('v_head.weight')
            # 使用only_optimize_lora_parameters函数,将模型设置为只优化LoRA层的参数
            # 传入的参数包括:
            # rm_model: 待优化的模型
            # force_optimize_params: 需要优化的参数列表
            rm_model = only_optimize_lora_parameters(rm_model,
                                                     force_optimize_params)
            # 使用make_model_gradient_checkpointing_compatible函数,使模型兼容梯度检查点功能
            # 梯度检查点可以在一定程度上减少内存占用,但会增加计算开销
            rm_model = make_model_gradient_checkpointing_compatible(rm_model)

    # 11. 创建训练和评估数据集
    # 参数包括:
    # local_rank: 当前进程的本地rank
    # data_split: RLHF三个阶段数据的比例
    # data_output_path: 处理后的数据保存路径
    # train_phase: 指定训练阶段,不同阶段的数据处理方式可能不同,本次是二阶段
    # seed: 随机种子,用于数据集的随机划分
    # tokenizer: 上面加载的tokenizer
    # max_seq_len: 输入序列的最大长度
    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len)

    # DataLoaders creation:
    # 12. 创建数据加载器
    # 使用PyTorch的DataLoader加载训练和评估数据集
    # 根据是否为分布式训练,使用不同的采样器(sampler)
    # 数据CollatorReward用于对批次数据进行处理和打包
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    # 13. 定义评估函数
    def evaluation_reward(model, dataloader, eval_iters):
        model.eval()   # 1. 将模型设置为评估模式,关闭dropout和批量归一化层
        # correct_predictions: 正确预测的数量
        # total_predictions: 总预测数量
        # chosen_scores: 选择的序列的平均分数
        # rejected_scores: 拒绝的序列的平均分数
        correct_predictions = 0
        total_predictions = 0
        chosen_scores = 0.
        rejected_scores = 0.
        for _step, _batch in enumerate(dataloader):
             # 将批次数据移动到设备(GPU 或 CPU)上
            _batch = to_device(_batch, device)
            # 使用 no_grad ,不计算梯度。
            with torch.no_grad():
                _outputs = model(**_batch)  # 在评估模式下,对批次数据进行前向传播,获取模型输出。
            # 从模型输出中提取选择和拒绝的序列的分数
            chosen = _outputs["chosen_mean_scores"]
            rejected = _outputs["rejected_mean_scores"]
            # 计算正确预测的数量,即选择序列的分数大于拒绝序列的分数
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            # 累加选择和拒绝序列的平均分数
            chosen_scores += _outputs["chosen_mean_scores"].mean().float()
            rejected_scores += _outputs["rejected_mean_scores"].mean().float()
            if (_step + 1) == eval_iters:   # 如果已经评估了指定的迭代次数,则提前退出循环,以加快评估速度。
                break
        # 计算准确率
        _acc = correct_predictions / total_predictions
        chosen_scores = chosen_scores / (_step + 1)    # 计算选择序列的平均分数
        rejected_scores = rejected_scores / (_step + 1)  # 计算拒绝序列的平均分数
        try:
            # 如果在分布式训练环境中,使用 get_all_reduce_mean 函数对指标进行跨进程求平均,确保指标的正确性。
            _acc = get_all_reduce_mean(_acc).item()
            chosen_scores = get_all_reduce_mean(chosen_scores).item()
            rejected_scores = get_all_reduce_mean(rejected_scores).item()
        except:
            pass
        return chosen_scores, rejected_scores, _acc   # 返回评估指标

    # Split weights in two groups, one with weight decay and the other not.
    # 14. 划分参数组
    # 将模型参数划分为两组,一组应用权重衰减,另一组不应用权重衰减,这是一种常见的正则化技术,可以帮助防止过拟合
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay, args.lora_learning_rate)
    # 15. 创建优化器
    # DeepSpeed 提供了多种优化器,根据是否启用 CPU offload 选择不同的优化器,AdamOptimizer 是 DeepSpeed 自定义的优化器,可以与其他优化功能无缝集成,如 CPU offload 和混合精度训练
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))    # betas 参数指定了 Adam 优化器的指数衰减率
    
    # 16. 计算每个 epoch 的更新步数
    # 更新步数等于训练数据集的批次数量除以梯度累积步数
    # math.ceil 函数用于向上取整,以确保所有批次都被处理
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    # 17. 获取学习率调度器
    # 学习率调度器用于在训练过程中动态调整学习率,以提高模型性能
    # get_scheduler 函数根据指定的调度器类型(如线性warmup、余弦等)创建相应的调度器实例
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    # 18. 使用 DeepSpeed 初始化模型、优化器和学习率调度器
    # DeepSpeed 提供了一种高效的方式来初始化这些组件,并启用各种优化功能
    # 例如:ZeRO 优化、CPU offload、混合精度训练等
    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,     
        config=ds_config,   # DeepSpeed 配置
        lr_scheduler=lr_scheduler,
        dist_init_required=True)    # 指示是否需要初始化分布式环境

    # 19. 启用梯度检查点(如果配置了)
    # 梯度检查点是一种内存优化技术,可以在一定程度上减少内存占用,但会增加计算开销
    # 它通过在计算图的某些点重新计算激活值,而不是存储所有中间激活值,从而节省内存
    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    # 21. 进行初始评估
    # 在训练开始前,在评估数据集上评估模型的初始性能
    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    reward_score, reject_score, acc = evaluation_reward(
        rm_model, eval_dataloader, args.eval_iters)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, "
        f"rejected_last_scores (lower is better) : {reject_score}, "
        f"acc (higher is better) : {acc}", args.global_rank)

    total_micro_steps = 0  # 初始化微批次步数计数器为 0
    for epoch in range(args.num_train_epochs):
        # 打印当前 epoch 的开始信息,包括 epoch 编号和总微批次数
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()   # 将模型设置为训练模式,启用 dropout 和批量归一化层
        mean_loss = 0   # 初始化平均损失为 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)  # 对批次数据进行前向传播,获取模型输出
            loss = outputs["loss"]  
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()   # 累加损失值
            total_micro_steps += 1     # 累加微批次步数
            # 判断是否达到梯度累积步数
            gas_boundary = (total_micro_steps %
                            args.gradient_accumulation_steps == 0)
            # 计算总的训练步数,等于微批次步数除以梯度累积步数
            total_steps = total_micro_steps // args.gradient_accumulation_steps
            # 如果达到了评估间隔,在评估数据集上评估模型性能
            if args.eval_interval and gas_boundary and (
                    total_steps % args.eval_interval == 0):
                print_rank_0(f"Iter {total_steps}: Evaluating reward",
                             args.global_rank)
                # 调用评估函数,计算选择序列分数、拒绝序列分数和准确率
                reward_score, reject_score, acc = evaluation_reward(
                    rm_model, eval_dataloader, args.eval_iters)
                print_rank_0(
                    f"Iter {total_steps}: c_scores: {reward_score}, r_scores: {reject_score}, "
                    f"diff: {reward_score - reject_score}, acc: {acc}",
                    args.global_rank)
                # 将模型重新设置为训练模式, 因为evaluation_reward会把模型转化为eval模式
                rm_model.train()

        # 打印当前 epoch 的平均损失
        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)
        # Evaluate reward_loss on the validation set.
        # 在每个 epoch 结束时,在评估数据集上评估模型的性能,用于监控训练进度
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        # 调用评估函数,计算选择序列分数、拒绝序列分数和准确率
        reward_score, reject_score, acc = evaluation_reward(
            rm_model, eval_dataloader, args.eval_iters)
        print_rank_0(
            f"chosen_last_scores (higher is better) : {reward_score}, "
            f"rejected_last_scores (lower is better) : {reject_score}, "
            f"acc (higher is better) : {acc}", args.global_rank)
        # 更新吞吐量计时器的epoch计数
        rm_model.tput_timer.update_epoch_count()

    # 保存模型
    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)
        # 如果使用了 LoRA (Low-Rank Adaptation) 技术,将 LoRA 层转换为线性层,以便保存模型
        rm_model = convert_lora_to_linear_layer(rm_model)

        if args.global_rank == 0:
            # 在 rank 0 的进程中,将模型保存为 Hugging Face 格式
            save_hf_format(rm_model, tokenizer, args)
        if args.zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            # 如果使用了 DeepSpeed ZeRO 优化阶段 3,则需要在每个 GPU 上保存模型的一部分
            # 因为在 ZeRO 优化阶段 3 下,模型参数是被分片并分散在不同的 GPU 上
            save_zero_three_model(rm_model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)

if __name__ == "__main__":
    main()