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
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

from dschat.utils.data.data_utils import create_prompt_dataset
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from dschat.utils.ds_utils import get_train_ds_config
from dschat.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from dschat.utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
from dschat.utils.perf import print_throughput


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
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 20%% of data for phase 1, 40%% for phase 2'
                        'and 40%% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        # 数据相关文件（如洗牌索引）应该存储在哪里？这需要放在节点的本地存储上（而不是共享存储）,默认即可
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
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
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
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
    parser.add_argument("--output_dir",
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
                        help="local_rank for distributed training on gpus, 该参数默认即可,不用去调节, 如果有需要可以使用bash中的环境变量来控制使用那些卡")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
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
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',  # 计算loss是否用fp32
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Tokenizer
    #TODO 上面这个参数在 1/2/3阶段,均需要打开,为了保证 tokenizer 可以准确的分词 <|endoftext|> ,也就是说bash后需要加上 --add_eot_token
    # OPT 模型是无法分词 <|endoftext|>的
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # 设置设备(GPU/CPU)并初始化分布式训练环境
    if args.local_rank == -1:
        # 如果local_rank为-1,表示单机单GPU训练,使用get_accelerator().device_name()获取设备名称,并创建torch.device对象
        device = torch.device(get_accelerator().device_name())
    else:  # 否则,表示分布式训练
        # 使用get_accelerator().set_device(args.local_rank)设置当前进程使用的GPU
        get_accelerator().set_device(args.local_rank)
        # 创建torch.device对象,指定当前进程的GPU编号
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # 初始化分布式后端,用于同步不同节点/GPU之间的数据
        # 使用DeepSpeed的init_distributed()函数初始化分布式训练环境, 内部用的是 torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    # 获取当前进程的全局排名(rank)
    args.global_rank = torch.distributed.get_rank()

    # print args, 仅使用序号为0的进程来打印
    print_rank_0("***** Parmeters *****".rjust(62), args.global_rank)
    arguments = vars(args)
    for key, value in arguments.items():
        print_rank_0(f"{key}".rjust(50)+ f": {value}", args.global_rank)

    # 获取DeepSpeed训练配置
    # get_train_ds_config()是一个函数,用于获取DeepSpeed训练配置
    # offload:是否启用CPU offload,将一些张量从GPU卸载到CPU以节省GPU内存
    # dtype:张量的数据类型,例如torch.float16用于混合精度训练
    # stage:ZeRO优化的阶段,0/1/2/3表示不同的切片优化策略,0表示不使用ZeRO
    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    # enable_tensorboard:是否启用TensorBoard可视化
                                    # tb_path:TensorBoard日志保存路径
                                    # tb_name:TensorBoard运行的名称
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    
    # 重新设置每个GPU的训练微批次大小,args.per_device_train_batch_size是每个GPU的训练批次大小
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    
    # 重新设置总批次大小 # torch.distributed.get_world_size()返回分布式训练中的总GPU数量 # args.gradient_accumulation_steps是梯度累积的步数
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    # 如果提供了随机种子,一键设置训练的随机种子
    set_random_seed(args.seed)
    # 同步所有进程,等待所有进程到达这一点
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    # load_hf_tokenizer将根据模型系列获取正确的tokenizer并设置填充标记
    args.end_of_conversation_token = "<|endoftext|>"  # args.end_of_conversation_token是对话结束的特殊标记
    # 如果args.add_eot_token为True,则将结束标记作为额外的特殊标记
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    # load_hf_tokenizer()是一个函数,用于加载Hugging Face的tokenizer
    # args.model_name_or_path是预训练模型的名称或路径
    # fast_tokenizer=True表示使用快速tokenizer
    # add_special_tokens添加额外的特殊标记
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    # 创建Hugging Face模型
    # create_hf_model()是一个函数,用于创建Hugging Face模型
    # AutoModelForCausalLM是自回归语言模型的类
    # args.model_name_or_path是预训练模型的名称或路径
    # tokenizer是前面加载的tokenizer对象
    # ds_config是DeepSpeed的训练配置
    # args.dropout是dropout率
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            dropout=args.dropout)

    # 设置是否使用FP32计算损失函数
    if args.compute_fp32_loss:
        # print_rank_0()是一个函数,只在rank为0的进程中打印
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32",
            args.global_rank)
        # causal_lm_model_to_fp32_loss()是一个函数,将自回归语言模型的损失计算改为FP32
        causal_lm_model_to_fp32_loss(model)

    # 应用LoRA(Low-Rank Adaptation)技术
    if args.lora_dim > 0:
        # convert_linear_layer_to_lora()是一个函数,将线性层转换为LoRA形式
        # args.lora_module_name是要应用LoRA的模块名称
        # args.lora_dim是LoRA的秩
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            # only_optimize_lora_parameters()是一个函数,只优化LoRA参数,冻结其他参数
            model = only_optimize_lora_parameters(model)
            # 使模型兼容梯度检查点
            model = make_model_gradient_checkpointing_compatible(model)

    # Prepare the data
    # 准备数据
    # create_prompt_dataset()是一个函数,用于创建提示数据集
    # args.local_rank是当前进程的本地rank
    # args.data_path是数据路径
    # args.data_split是数据分割方式, 指的是RLHF三个阶段每个阶段数据的占比
    # args.data_output_path是数据输出路径, 缓存路径, 用于缓存预处理的数据
    # train_phase是训练阶段
    # args.seed是随机种子
    # tokenizer是前面加载的tokenizer对象
    # args.max_seq_len是最大序列长度
    # sft_only_data_path是只包含SFT数据的路径
    train_phase = 1 # 训练阶段一, SFT
    train_dataset, eval_dataset = create_prompt_dataset(args.local_rank,args.data_path,args.data_split,
                                                        args.data_output_path,train_phase,args.seed,
                                                        tokenizer,args.max_seq_len,
                                                        sft_only_data_path=args.sft_only_data_path)
    # DataLoaders creation:
    # 创建数据加载器
    if args.local_rank == -1:
        # 1. 如果local_rank等于-1,表示当前是单机单GPU训练模式, 在这种情况下,使用RandomSampler和SequentialSampler作为训练集和评估集的采样器
        # 3. RandomSampler用于从训练集中随机采样数据,有助于提高模型的泛化能力
        train_sampler = RandomSampler(train_dataset)
        # 4. SequentialSampler用于按顺序遍历评估集,确保评估过程的一致性和可重复性
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        # 5. 如果local_rank不等于-1,表示当前是分布式多GPU训练模式, 在这种情况下,使用DistributedSampler作为训练集和评估集的采样器
        # 7. DistributedSampler可以将数据集划分为多个子集,每个子集由不同的GPU处理
        # 8. 这种分布式采样方式可以加速训练过程,提高计算效率
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    # 9. DataLoader是PyTorch的数据加载器,用于加载数据集并进行批处理, 它可以与不同的采样器(Sampler)结合使用,以实现不同的采样策略
    # 11. default_data_collator是一个默认的数据处理collator函数,其实就是 collate_fn
    # 13. collate_fn参数用于指定数据collator函数,这里使用了default_data_collator  
    # 14. 创建训练数据加载器,使用train_sampler作为采样器,batch_size由args.per_device_train_batch_size指定
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    # 15. 创建评估数据加载器,使用eval_sampler作为采样器,batch_size由args.per_device_eval_batch_size指定
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    # 定义评估函数
    def evaluation(model, eval_dataloader):
        # 设置模型为评估模式
        model.eval()
        losses = 0
        # 遍历评估数据加载器
        for step, batch in enumerate(eval_dataloader):
            # 将批次数据移动到设备上
            # to_device()是一个函数,用于将数据移动到指定设备上
            batch = to_device(batch, device)
            with torch.no_grad():
                # 模型前向计算,获取输出
                outputs = model(**batch)

            # 获取损失
            loss = outputs.loss
            # 累加损失
            losses += loss.float()
        # 计算平均损失
        losses = losses / (step + 1)
        try:
            # 使用get_all_reduce_mean()函数在分布式训练中计算平均损失
            losses = get_all_reduce_mean(losses)
        except:
            pass
        try:
            # 计算perplexity
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        # 返回perplexity和平均损失
        return perplexity, losses.item()

    # Split weights in two groups, one with weight decay and the other not.
    # 将权重分为两组,一组使用权重衰减,另一组不使用, get_optimizer_grouped_parameters()是一个函数,用于将模型参数分组
    # model是前面创建的模型对象, args.weight_decay是权重衰减系数, args.lora_learning_rate是LoRA参数的学习率
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)
    # 创建优化器, 如果args.offload为True,使用DeepSpeedCPUAdam优化器,否则使用FusedAdam优化器
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam

    # 创建优化器对象
    # optimizer_grouped_parameters是前面分组的参数
    # args.learning_rate是初始学习率
    # betas是Adam优化器的超参数,betas=(0.9, 0.95): 这是 Adam 优化器的超参数之一，
    # 用于设定一阶矩估计的指数衰减率（通常是 β1）和二阶矩估计的指数衰减率（通常是 β2）。在 Adam 优化算法中，这两个参数通常设定为分别为 0.9 和 0.999。β1 控制一阶矩估计的衰减程度，β2 控制二阶矩估计的衰减程度。
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    # 计算每个epoch的更新步数, 向上取整, len(train_dataloader)是训练数据加载器的长度,即批次数, args.gradient_accumulation_steps是梯度累积的步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # 创建学习率调度器
    # get_scheduler()是一个函数,用于创建学习率调度器
    # args.num_train_epochs是训练的epoch数
    # num_update_steps_per_epoch是每个epoch的更新步数
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,  # args.lr_scheduler_type是调度器类型
        optimizer=optimizer,   # optimizer是前面创建的优化器对象
        num_warmup_steps=args.num_warmup_steps,  # args.num_warmup_steps是warmup步数
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,  # 即总的迭代次数
    )
    # 初始化DeepSpeed引擎, model初始化以后就是deepspeed的对象的,与huggingface的对象有些区别
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,    # model是前面创建的模型对象
        optimizer=optimizer,   # optimizer是前面创建的优化器对象
        args=args,       # args是命令行参数对象
        config=ds_config,     # ds_config是DeepSpeed的训练配置
        lr_scheduler=lr_scheduler,     # lr_scheduler是前面创建的学习率调度器
        dist_init_required=True)    # 是否初始化分布式环境

    # 启用梯度检查点(如果需要)
    if args.gradient_checkpointing:
        # model.gradient_checkpointing_enable()启用梯度检查点,以节省内存
        model.gradient_checkpointing_enable()

    # Train!
    # 开始训练
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    # 评估初始perplexity
    perplexity, eval_loss = evaluation(model, eval_dataloader)
    print_rank_0(f"ppl: {perplexity}, loss: {eval_loss}", args.global_rank)

    # 训练循环
    for epoch in range(args.num_train_epochs):
        # 打印当前训练进度信息
        # print_rank_0()是一个函数,只在rank为0的进程中打印, args.global_rank是当前进程的全局排名, len(train_dataloader)是训练数据加载器的长度,即批次数
        print_rank_0(f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        # 设置模型为训练模式
        model.train()
        import time
        # 遍历训练数据加载器
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            # 将批次数据移动到设备上
            # to_device()是一个函数,用于将数据移动到指定设备上
            batch = to_device(batch, device)
            # 模型前向计算,获取输出
            # **batch是Python的解包操作,将字典batch中的键值对作为参数传递给模型
            # use_cache=False表示不使用缓存
            outputs = model(**batch, use_cache=False)
            # 获取损失
            loss = outputs.loss
            # 打印损失(如果需要)
            if args.print_loss:
                # torch.distributed.get_rank()返回当前进程在分布式训练中的排名
                print(f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}")
            # 反向传播
            model.backward(loss)
            # 更新模型参数
            model.step()
            end = time.time()
            # 打印吞吐量(仅在rank为0的进程中执行)
            if torch.distributed.get_rank() == 0:
                # print_throughput()是一个函数,用于打印模型的吞吐量/性能等, args是命令行参数对象, end - start是一个micro batch所用的时间
                print_throughput(model.model, args, end - start,args.global_rank)
                
        # Evaluate perplexity on the validation set.
        # 打印评估perplexity的信息, 一个epoch评估一次
        print_rank_0(f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",args.global_rank)
        # 调用evaluation函数评估perplexity
        # evaluation()是一个自定义函数,用于评估模型在验证集上的perplexity和损失
        perplexity, eval_loss = evaluation(model, eval_dataloader)
        # 打印评估结果
        print_rank_0(f"ppl: {perplexity}, loss: {eval_loss}", args.global_rank)
        # 更新吞吐量计时器的epoch计数
        model.tput_timer.update_epoch_count()

    # 保存最终模型
    if args.output_dir is not None:
        # 打印正在保存模型的信息
        print_rank_0('saving the final model ...', args.global_rank)

        # 将LoRA转换为线性层
        # convert_lora_to_linear_layer()是一个函数,用于将带有LoRA的层转换为标准的线性层
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            # 在rank为0的进程中,使用save_hf_format()函数保存模型
            # save_hf_format()是一个函数,用于将模型保存为Hugging Face格式, 主要保存, model, model的config, tokenizer 三部分
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            # 如果使用ZeRO优化阶段3,每个GPU只有模型的一部分, 这个函数可以把分散在各个设备上的model参数拉倒同一个进程中, 然后保存
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
