# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import time
import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM, get_scheduler

from dschat.utils.ds_utils import get_train_ds_config, get_eval_ds_config
from dschat.utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from dschat.utils.model.model_utils import create_hf_model, create_critic_model
from dschat.utils.utils import get_optimizer_grouped_parameters
"""
TODOs:
  * support HF models for critic (for debugging), must be a previously saved ckpt from step-2
  * determine ds_config/zero_stage based on model size, gpu style, world size, etc
    - get model size by creating simple meta model
    - 1.3b: zero-2 for actor/ref models, zero-0 for others
    - 13b+: zero-3 for all models
"""

# 1. model_name 参数是模型的名称,用于在日志中显示。
# 2. stime 参数是可选的,如果提供则表示记录了模型初始化的开始时间,用于计算初始化所需的时间。
def log_init(model_name, stime=None):
    # 3. 检查当前进程是否为主进程(rank 0),只有主进程负责打印日志。
    if torch.distributed.get_rank() == 0:
        # 4. 根据是否提供 stime 参数,决定日志中使用的前缀("start" 或 "end")。
        tag = "start" if stime is None else "end"
        # 5. 根据是否提供 stime 参数,决定动词的时态("initializing" 或 "initialized")。
        suffix = "ing" if stime is None else "ed"
        # 6. 初始化用于存储初始化时间的变量。
        duration = ""
        # 7. 如果提供了 stime 参数,则计算初始化所需的时间并格式化。
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        # 8. 构建日志消息,包含模型名称、初始化状态和耗时信息。
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        # 9. 计算星号的数量,使得日志消息居中显示。
        stars = (90 - len(msg)) // 2
        # 10. 如果星号数量为奇数,添加一个额外的星号。
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        # 11. 打印格式化后的日志消息。
        print("*" * stars + msg + "*" * stars + extra_star)
        # 12. 如果是记录初始化开始时间,则返回当前时间。
        return time.time()


class DeepSpeedRLHFEngine():
    # 1. 这是一个用于 RLHF (Reinforcement Learning from Human Feedback) 训练的 DeepSpeed 引擎类。
    # 2. RLHF 是一种通过人类反馈来微调和优化语言模型的技术,它结合了强化学习和人类评分,可以使语言模型生成更加符合人类期望的输出。
    # 3. DeepSpeed 是一个深度学习优化库,用于提高大型模型训练的效率和性能。
    def __init__(self, actor_model_name_or_path, critic_model_name_or_path,
                 tokenizer, args, num_total_iters):
        # 4. __init__ 方法用于初始化 RLHF 引擎,接受以下参数:
        #    actor_model_name_or_path: 生成模型(Actor Model)的名称或路径。
        #    critic_model_name_or_path: 奖励模型(Critic Model)的名称或路径。
        #    tokenizer: 用于文本标记化和解码的 Tokenizer 对象。
        #    args: 包含各种配置参数的命名元组或对象。
        #    num_total_iters: 总的训练迭代次数。
        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer
        # 5. 初始化生成模型(Actor Model)。
        self.actor = self._init_actor(
            actor_model_name_or_path=actor_model_name_or_path)
        # 6. 初始化参考模型(Reference Model),用于生成基线输出。
        self.ref = self._init_ref(
            actor_model_name_or_path=actor_model_name_or_path)
        # 7. 如果启用了指数移动平均(Exponential Moving Average, EMA),则初始化 EMA 模型。
        #    EMA 是一种用于平滑模型权重更新的技术,可以提高模型性能和稳定性。
        self.actor_ema = None
        if self.args.enable_ema:
            self.actor_ema = self._init_ema(
                actor_model_name_or_path=actor_model_name_or_path)
        # 8. 初始化奖励模型(Critic Model)。
        self.critic = self._init_critic(
            critic_model_name_or_path=critic_model_name_or_path)
        # 9. 初始化奖励模型(Reward Model),用于评估生成的文本序列的质量。
        self.reward = self._init_reward(
            critic_model_name_or_path=critic_model_name_or_path)
        # 10. 如果配置了启用奖励模型的梯度检查点功能,则启用它。
        #     梯度检查点是一种训练技术,它通过重新计算部分激活值来减少内存使用,从而允许训练更大的模型。
        if self.args.critic_gradient_checkpointing:
            self.critic.gradient_checkpointing_enable()

    def _init_actor(self, actor_model_name_or_path):
        # 11. 记录初始化生成模型的开始时间,用于计时。
        stime = log_init("Actor")

        # DS Config
        ds_config = get_train_ds_config(
            # 12. offload 参数指示是否启用 CPU Offload,将一些张量从 GPU 卸载到 CPU 以节省 GPU 内存。
            offload=self.args.offload,
            # 13. dtype 参数指定张量的数据类型,例如 torch.float16 用于混合精度训练。
            dtype=self.args.dtype,
            # 14. actor_zero_stage 参数指定了 DeepSpeed 的 ZeRO 优化阶段,用于控制模型参数的分片和优化策略。
            #     0/1/2/3 表示不同的优化策略,3 是最高级别的优化,可以支持训练更大的模型。
            stage=self.args.actor_zero_stage,
            # 15. enable_hybrid_engine 参数指示是否启用 DeepSpeed 的混合引擎,将部分计算卸载到 CPU 以节省 GPU 内存。
            enable_hybrid_engine=self.args.enable_hybrid_engine,
            # 16. inference_tp_size 参数指定推理时张量并行的大小,用于控制计算和内存使用。
            inference_tp_size=self.args.inference_tp_size,
            # 17. release_inference_cache 参数指示是否在推理时释放一些缓存,以节省内存。
            release_inference_cache=self.args.release_inference_cache,
            # 18. pin_parameters 参数指示是否将模型参数固定在内存中,以提高性能。
            pin_parameters=(not self.args.unpin_actor_parameters),
            # 19. tp_gather_partition_size 参数用于控制张量并行时的分区大小,影响计算和内存使用。
            tp_gather_partition_size=self.args.tp_gather_partition_size,
            # 20. max_out_tokens 参数指定生成序列的最大长度,用于限制内存使用。
            max_out_tokens=self.args.max_prompt_seq_len + self.args.max_answer_seq_len,
            # 21. enable_tensorboard 参数指示是否启用 TensorBoard 可视化,用于监控训练过程。
            enable_tensorboard=self.args.enable_tensorboard,
            # 22. enable_mixed_precision_lora 参数指示是否启用混合精度训练和 LoRA (Low-Rank Adaptation) 技术。
            #     LoRA 是一种用于微调大型预训练语言模型的技术,它通过引入低秩投影来修改模型权重,从而显著降低所需的参数数量。
            enable_mixed_precision_lora=self.args.enable_mixed_precision_lora,
            # 23. tensorboard_path 参数指定 TensorBoard 日志保存路径。
            tb_path=self.args.tensorboard_path,
            # 24. tb_name 参数指定 TensorBoard 运行的名称。
            tb_name="step3_actor")
        # 25. train_micro_batch_size_per_gpu 参数指定每个 GPU 的微批次大小,用于控制内存使用和计算效率。
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        # 26. train_batch_size 参数指定总的批次大小,通过乘以 GPU 数量和梯度累积步数计算得到。
        ds_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        # Model
        # 28. create_hf_model 函数用于从预训练模型创建一个 Hugging Face 模型实例。
        actor_model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=actor_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            dropout=self.args.actor_dropout)

        # LoRA
        # 29. 如果指定了 LoRA 维度,则将线性层转换为 LoRA 层,以减少参数数量。
        if self.args.actor_lora_dim > 0:
            actor_model = convert_linear_layer_to_lora(
                actor_model, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)
            if self.args.only_optimize_lora:
                # 30. 如果只优化 LoRA 参数,则进行相应的优化和梯度检查点兼容性处理。
                actor_model = only_optimize_lora_parameters(actor_model)
                actor_model = make_model_gradient_checkpointing_compatible(
                    actor_model)

        # Optimizer
        # 31. 根据是否启用 CPU Offload,选择 DeepSpeed 的 CPU Adam 优化器或 FusedAdam 优化器。
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        # 32. 获取优化器的分组参数,包括权重衰减和 LoRA 学习率。
        optim_params = get_optimizer_grouped_parameters(
            actor_model, self.args.actor_weight_decay,
            self.args.actor_lora_learning_rate)
        # 33. 创建 Adam 优化器实例,使用指定的学习率和 Beta 值。
        optim = AdamOptimizer(optim_params,
                              lr=self.args.actor_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        # 34. 创建学习率调度器,用于在训练过程中动态调整学习率。
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        #TODO: move enable_hybrid_engine and pin_parameters to ds_config
        # 35. 使用 DeepSpeed 初始化生成模型引擎,将模型、优化器、学习率调度器和配置传入。
        actor_engine, *_ = deepspeed.initialize(model=actor_model,
                                                optimizer=optim,
                                                lr_scheduler=lr_scheduler,
                                                config=ds_config)
        # 36. 记录初始化生成模型的结束时间,用于计算所用时间。
        log_init("Actor", stime=stime)
        # 37. 返回初始化后的生成模型引擎。
        return actor_engine
    # 38. 记录初始化参考模型的开始时间,用于计时。
    def _init_ref(self, actor_model_name_or_path):
        stime = log_init("Ref")
        # DS Config
        zero_stage = self.args.actor_zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory for ref model
            # 39. 如果actor模型不使用 ZeRO-3 优化,则假设有足够的内存来创建参考模型,并将 zero_stage 设置为 0。
            zero_stage = 0
        # 40. 获取参考模型的 DeepSpeed 评估配置,包括是否卸载模型、数据类型和 ZeRO 优化阶段。
        ds_config = get_eval_ds_config(self.args.offload_reference_model,
                                       self.args.dtype, zero_stage)
        # 41. 设置参考模型的微批次大小和总批次大小,与生成模型相同。
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor
        # 42. 创建参考模型实例,使用与生成模型相同的预训练模型和 DeepSpeed 配置。
        ref_model = create_hf_model(AutoModelForCausalLM,
                                    actor_model_name_or_path, self.tokenizer,
                                    ds_config)
        # 43. 使用 DeepSpeed 初始化参考模型引擎。
        ref_engine, *_ = deepspeed.initialize(model=ref_model,
                                              config=ds_config)
        # 44. 记录初始化参考模型的结束时间,用于计算所用时间。
        log_init("Ref", stime=stime)
        # 45. 返回初始化后的参考模型引擎。
        return ref_engine

    def _init_ema(self, actor_model_name_or_path):
        # 46. 记录初始化 EMA 模型的开始时间,用于计时。
        stime = log_init("EMA")
        # DS Config
        zero_stage = self.args.actor_zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            # 47. 如果生成模型不使用 ZeRO-3 优化,则假设有足够的内存来创建 EMA 模型,并将 zero_stage 设置为 0。
            zero_stage = 0
        # 48. 获取 EMA 模型的 DeepSpeed 评估配置,包括是否卸载模型、数据类型和 ZeRO 优化阶段。
        ds_config = get_eval_ds_config(self.args.offload_reference_model,
                                       self.args.dtype, zero_stage)
        # 49. 设置 EMA 模型的微批次大小和总批次大小,与生成模型相同。
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor
        # 50. 创建 EMA 模型实例,使用与生成模型相同的预训练模型和 DeepSpeed 配置。
        actor_model_ema = create_hf_model(AutoModelForCausalLM,
                                          actor_model_name_or_path,
                                          self.tokenizer, ds_config)
        # 51. 如果指定了 LoRA 维度,则将 EMA 模型的线性层转换为 LoRA 层,以减少参数数量。
        if self.args.actor_lora_dim > 0:
            actor_model_ema = convert_linear_layer_to_lora(
                actor_model_ema, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)
        # 52. 使用 DeepSpeed 初始化 EMA 模型引擎。
        ema_engine, *_ = deepspeed.initialize(model=actor_model_ema,
                                              config=ds_config)
        # 53. 记录初始化 EMA 模型的结束时间,用于计算所用时间。
        log_init("EMA", stime=stime)
        # 54. 返回初始化后的 EMA 模型引擎。
        return ema_engine

    def _init_critic(self, critic_model_name_or_path):
        # 55. 记录初始化奖励模型的开始时间,用于计时。
        stime = log_init("Critic")
        ds_config = get_train_ds_config(
            offload=self.args.offload,
            dtype=self.args.dtype,
            # 56. critic_zero_stage 参数指定了奖励模型的 DeepSpeed ZeRO 优化阶段。
            stage=self.args.critic_zero_stage,
            enable_tensorboard=self.args.enable_tensorboard,   
            tb_path=self.args.tensorboard_path,            # 57. 如果启用了 TensorBoard,则设置相关的路径和名称。
            tb_name="step3_critic")
        # 58. 设置奖励模型的微批次大小和总批次大小。
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps
        # 59. 获取奖励模型的 DeepSpeed 评估配置,用于评估和推理。
        ds_eval_config = get_eval_ds_config(offload=False,
                                            dtype=self.args.dtype,
                                            stage=self.args.critic_zero_stage)
        # We need to set train batch size and micro batch size here to pass the sanity check of DeepSpeed engine.
        # 60. 设置评估配置中的微批次大小和总批次大小,用于通过 DeepSpeed 引擎的检查。
        ds_eval_config['train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        ds_eval_config['train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size() * self.args.gradient_accumulation_steps

        # Model
        # 61. 创建奖励模型实例,使用指定的预训练模型、Tokenizer、DeepSpeed 配置和其他相关参数。
        critic_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_eval_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            dropout=self.args.critic_dropout,
            zero_stage=self.args.critic_zero_stage)

        # LoRA
        # 62. 如果指定了 LoRA 维度,则将奖励模型的线性层转换为 LoRA 层,以减少参数数量。
        if self.args.critic_lora_dim > 0:
            critic_model = convert_linear_layer_to_lora(
                critic_model, self.args.critic_lora_module_name,
                self.args.critic_lora_dim)
            if self.args.only_optimize_lora:
                # 63. 如果只优化 LoRA 参数,则进行相应的优化和梯度检查点兼容性处理。
                critic_model = only_optimize_lora_parameters(critic_model)
                critic_model = make_model_gradient_checkpointing_compatible(
                    critic_model)

        # Optimizer
        # 64. 根据是否启用 CPU Offload,选择 DeepSpeed 的 CPU Adam 优化器或 FusedAdam 优化器。
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        # 65. 获取优化器的分组参数,包括权重衰减和 LoRA 学习率。
        optim_params = get_optimizer_grouped_parameters(
            critic_model, self.args.critic_weight_decay,
            self.args.critic_lora_learning_rate)
        # 66. 创建 Adam 优化器实例,使用指定的学习率和 Beta 值。
        optim = AdamOptimizer(optim_params,
                              lr=self.args.critic_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        # 67. 创建学习率调度器,用于在训练过程中动态调整学习率。
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        # 68. 使用 DeepSpeed 初始化奖励模型引擎,将模型、优化器、学习率调度器和配置传入。
        critic_engine, *_ = deepspeed.initialize(model=critic_model,
                                                 optimizer=optim,
                                                 lr_scheduler=lr_scheduler,
                                                 config=ds_config)
        # 69. 记录初始化奖励模型的结束时间,用于计算所用时间。
        log_init("Critic", stime=stime)
        # 70. 返回初始化后的奖励模型引擎。
        return critic_engine

    def _init_reward(self, critic_model_name_or_path):
        # 71. 记录初始化奖励模型的开始时间,用于计时。
        stime = log_init("Reward")
        # DS Config
        zero_stage = self.args.critic_zero_stage
        if zero_stage != 3:
            # If critic is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            # 72. 如果奖励模型不使用 ZeRO-3 优化,则假设有足够的内存来创建奖励模型,并将 zero_stage 设置为 0。
            zero_stage = 0
        # 73. 获取奖励模型的 DeepSpeed 评估配置,包括是否卸载模型、数据类型和 ZeRO 优化阶段。
        ds_config = get_eval_ds_config(offload=self.args.offload,
                                       dtype=self.args.dtype,
                                       stage=zero_stage)
        # 74. 设置奖励模型的微批次大小和总批次大小。
        ds_config['train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        ds_config['train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size() * self.args.gradient_accumulation_steps
        # 75. 获取奖励模型的 DeepSpeed 评估配置,用于评估和推理。
        ds_eval_config = get_eval_ds_config(offload=False,
                                            dtype=self.args.dtype,
                                            stage=zero_stage)

        # We need to set train batch size and micro batch size here to pass the sanity check of DeepSpeed engine.
        # 76. 设置评估配置中的微批次大小和总批次大小,用于通过 DeepSpeed 引擎的检查。
        ds_eval_config['train_micro_batch_size_per_gpu'] = self.args.per_device_training_batch_size
        ds_eval_config['train_batch_size'] = self.args.per_device_training_batch_size * torch.distributed.get_world_size() * self.args.gradient_accumulation_steps

        # Model
        # 77. 创建奖励模型实例,使用与奖励模型相同的参数。
        reward_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_eval_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            dropout=self.args.critic_dropout,
            zero_stage=zero_stage)
        # 78. 使用 DeepSpeed 初始化奖励模型引擎。
        reward_engine, *_ = deepspeed.initialize(model=reward_model,
                                                 config=ds_config)
        # 79. 记录初始化奖励模型的结束时间,用于计算所用时间。
        log_init("Reward", stime=stime)
         # 80. 返回初始化后的奖励模型引擎
        return reward_engine
