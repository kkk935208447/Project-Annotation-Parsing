# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import time
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from dschat.utils.utils import print_rank_0

# 这个函数print_all_ranks是一个用于分布式训练环境中的辅助函数,它的主要目的是收集并打印所有 rank 上的某个值。
# 在分布式训练中,每个计算节点(如 GPU)被分配一个 rank,并独立执行模型训练的一部分。
# 这个函数利用 PyTorch 的分布式通信功能,在所有 rank 上执行并行操作,将每个 rank 上的值收集到一个张量中,然后在 rank 0 上打印出这个张量。
def print_all_ranks(tag, value, rank):
    # 1. 这个函数用于在分布式训练环境中打印所有 rank 上的某个值。
    # 2. 分布式训练是一种将模型训练任务分配到多个计算节点(如 GPU)上的技术,可以加速训练过程并支持训练更大的模型。

    # 3. torch.distributed.get_world_size() 函数返回分布式训练环境中的总进程数,即 rank 的数量。
    world_size = torch.distributed.get_world_size()

    # 4. 创建一个全零张量,其长度等于 world_size,用于存储所有 rank 上的值。
    # 5. to(get_accelerator().current_device_name()) 将张量移动到当前的设备上(如 GPU)。
    # 6. get_accelerator() 是一个函数,用于获取当前的硬件加速器(如 GPU 或 CPU)。
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name())
    # 7. 将当前 rank 上的值赋值给 all_tensor 对应的位置。
    all_tensor[rank] = value
    # 8. torch.distributed.all_reduce() 函数用于在所有 rank 上执行一个并行操作(如求和),并将结果广播回每个 rank。
    # 9. op=torch.distributed.ReduceOp.SUM 指定了使用求和操作。
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    # 10. print_rank_0() 是一个自定义函数,用于在 rank 0 上打印输出。
    # 11. 在分布式环境中,通常只需要在一个 rank 上打印输出即可,以避免重复打印。
    print_rank_0(f'{tag} {all_tensor}', rank)


# 这个函数的作用是计算给定模型的参数 L2 范数。它利用了 PyTorch 的 no_grad() 上下文管理器,避免了梯度计算,从而节省内存和计算资源。
# 对于使用 DeepSpeed 的 ZeRO 优化策略的模型,该函数还会临时 gather 分散在不同设备上的参数,以确保能够正确计算参数的 L2 范数。
# 这种技术在训练大型模型时非常有用,可以帮助开发人员监控模型参数的复杂度和稳定性,从而进行更好的正则化和调优。
def get_model_norm(model):
    # 3. 使用 torch.no_grad() 上下文管理器,可以在不计算梯度的情况下执行前向传播,从而节省内存和计算资源。
        # 4. 在评估模型参数范数时,通常不需要梯度信息,因此使用 torch.no_grad() 是一种常见的优化技巧。
    with torch.no_grad():
        # 5. 初始化一个变量 total,用于累积所有参数的 L2 范数。
        total = 0.0
         # 6. 遍历模型的所有参数(如权重和偏置)。
        for param in model.parameters():
            # 7. 检查当前参数是否需要进行 gather 操作。
            # 8. 在使用 DeepSpeed 的 ZeRO 优化策略时,某些参数可能会被分发到不同的设备上,需要先 gather 回来才能计算范数。
            # 9. 'ds_id' 和 'ds_status' 是 DeepSpeed 内部使用的属性,用于跟踪参数的分布式状态。
            should_gather = hasattr(param,'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            # 10. 使用 deepspeed.zero.GatheredParameters 上下文管理器,可以临时将分散的参数 gather 回到当前设备上。
            # 11. 这样就可以在不影响模型训练的情况下,计算参数的 L2 范数。
            with deepspeed.zero.GatheredParameters(param,enabled=should_gather):
                # 12. 计算当前参数的 L2 范数,并累加到 total 变量中。
                # 13. 将参数转换为 float 类型是为了确保计算精度,因为有些参数可能是 half 或 bfloat16 类型。
                total += float(param.float().norm())
     # 14. 返回所有参数 L2 范数的总和,作为模型参数的整体范数。
    return total

# 这个函数gather_log_probs的主要作用是从模型的输出 logits 中,提取与目标标签 labels 对应的对数概率值。这是一种常见的技术,用于计算分类模型的损失函数。
# 它首先使用 log_softmax 将 logits 转换为概率分布,然后通过 gather 操作选择目标标签对应的对数概率值。
# 这种技术在训练分类模型时非常常见,可以作为损失函数的一部分,用于优化模型预测概率与真实标签之间的差距。
def gather_log_probs(logits, labels):
    # 3. 使用 F.log_softmax 函数计算 logits 的对数 softmax 值,得到每个类别的对数概率。
    # 4. 这是一种将模型输出转换为概率分布的常用技术。
    # 5. dim=-1 表示沿着最后一个维度(即类别维度)计算 softmax。
    log_probs = F.log_softmax(logits, dim=-1)
    # 6. 使用 .gather() 函数从 log_probs 中收集与 labels 对应的对数概率值。
    # 7. dim=-1 表示沿着最后一个维度进行 gather 操作。
    # 8. labels.unsqueeze(-1) 将最后一个维度挤压掉,以便于 gather。
    # 9. 这种 gather 操作可以高效地从模型输出中提取目标标签对应的对数概率值。
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    # 10. 将最后一个维度挤压掉,即每个样本对应的对数概率值。
    return log_probs_labels.squeeze(-1)


# 这段代码实现了一个基于 DeepSpeed 和 PPO 算法的强化学习训练器,用于训练对话系统等自然语言处理任务。主要特点包括:
# 使用 actor-critic 架构,分别训练 actor 模型 (生成模型) 和 critic 模型 (价值函数模型)。
# 结合了 KL 散度奖励和外部奖励模型的分数,用于指导强化学习训练。
# 实现了 PPO 算法的核心部分,如策略梯度损失、价值函数损失、优势函数和折返函数的计算。
# 支持 DeepSpeed 优化,如模型并行和 ZeRO 优化,用于加速大型模型的训练。
# 包含了一些辅助方法,如生成序列、计算模型范数等,用于监控和调试训练过程。
class DeepSpeedPPOTrainer():
    # 1. 这是一个名为 DeepSpeedPPOTrainer 的类,用于实现基于 DeepSpeed 的 PPO (Proximal Policy Optimization) 算法训练。
    # 2. PPO 是一种强化学习算法,常用于训练对话系统等自然语言处理任务。DeepSpeed 是一个用于加速深度学习训练的库,它提供了多种优化技术,如模型并行、ZeRO 优化等。
    def __init__(self, rlhf_engine, args):
        # 4. rlhf_engine 可能是一个包含预训练模型的对象,用于初始化 actor、critic、ref 和 reward 模型。
        # 5. args 是一个包含训练配置参数的对象,如 batch_size、learning_rate 等。
        self.rlhf_engine = rlhf_engine
        # 6. actor 模型是一个生成模型,用于生成文本序列作为强化学习的动作。
        self.actor_model = self.rlhf_engine.actor
        # 7. critic 模型是一个价值函数模型,用于估计生成序列的预期奖励。
        self.critic_model = self.rlhf_engine.critic
        # 8. ref 模型是一个参考模型,通常是一个预训练的语言模型,用于计算 KL 散度奖励, 即一阶段的SFT
        self.ref_model = self.rlhf_engine.ref
        # 9. reward 模型是一个奖励模型,用于评估生成序列的质量,并给出奖励分数。
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        # 10. 从配置参数中获取最大答案长度和对话结束标记的 ID。
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(args.end_of_conversation_token)['input_ids'][-1]
        # 11. 检查是否启用了 DeepSpeed 的 ZeRO 优化阶段 3,这是最高级别的优化,可以支持训练更大的模型。
        self.z3_enabled = args.actor_zero_stage == 3
        # 12. 检查是否需要计算 FP32 精度的损失,以提高数值稳定性。
        self.compute_fp32_loss = self.args.compute_fp32_loss

        # In case the generated experience is not valid (too short), we use the last valid
        # generated experience. Alternatively, we can skip the step (on all workers).
        # For now, use the last valid experience which is a simpler solution
        # 13. 初始化一个变量,用于存储上一次有效的生成经验,以防止当前生成的经验无效。
        self.last_generated_experience = None

        # Those value can be changed
        # 14. KL 控制系数,用于平衡 KL 散度奖励和其他奖励之间的权重。
        self.kl_ctl = 0.1
        # 15. 奖励裁剪值,用于将奖励分数限制在一个合理的范围内。
        self.clip_reward_value = 5
        # 16. PPO 算法中使用的策略裁剪范围,用于限制新旧策略之间的差异。
        self.cliprange = 0.2
        # 17. PPO 算法中使用的价值裁剪范围,用于限制新旧价值估计之间的差异。
        self.cliprange_value = 0.2
         # 18. 折扣因子,用于计算未来奖励的贴现值。
        self.gamma = 1.0
        # 19. GAE (Generalized Advantage Estimation) 中使用的权重,用于平衡偏差和方差。
        self.lam = 0.95
        # 20. 用于记录生成序列所需的时间,可用于监控和调试。
        self.generate_time = 0.0

    # 对生成的序列进行过滤,丢弃答案长度太短的序列。
    # 如果所有序列的答案长度都太短,则返回 None。
    # 打印输出提示和答案,用于监控和调试。
    # 将有效的答案序列连接成一个新的张量,作为最终输出。
    # 目的是确保生成的序列质量足够高,避免在后续的强化学习训练中引入噪声。
    def _generate_sequence(self, prompts, mask, step):
        #TODO 疑问:官方代码生成经验时,句子最短最长都是最大长度是否合理?
        # 24. 计算生成序列的最大长度,等于最大答案长度加上提示长度。
        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        # 25. 根据模型类型,设置是否启用采样模式。这是为了避免一个已知的概率/NaN 错误。
        if self.actor_model.module.config.model_type == "llama":
            kwargs = dict(do_sample=False)
        else:
            kwargs = dict()

        with torch.no_grad():
            # 26. 使用 actor 模型生成序列,传入提示、掩码、最大长度等参数。
            # 27. 使用 no_grad() 上下文管理器,避免计算梯度,从而节省内存和计算资源。
            # 28. synced_gpus 参数根据是否启用了 ZeRO 优化来设置,用于控制多 GPU 同步。
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled,
                **kwargs)

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        # 29. 从生成的序列中提取答案部分,并计算每个答案的有效长度。
        # 30. 这是为了过滤掉答案长度太短的序列,因为这可能是由于直接使用预训练检查点而导致的。
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        # 31. 根据配置,打印输出提示和答案,用于监控和调试。
        if self.args.print_answers and (step % self.args.print_answers_interval== 0):
            print(f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}")
            print(f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}")

        # 32. 遍历每个生成的序列,如果答案长度小于等于 1,则丢弃该序列。
        # 33. 打印被丢弃的序列,用于监控和调试。
        # 34. 将有效的序列添加到 out_seq 列表中。
        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                    f'answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
                )
                continue
            else:
                out_seq.append(seq[i:i + 1])
        # 35. 如果所有生成的序列都被丢弃,则打印错误信息并返回 None。
        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                f'-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
            )
            return None
        # 36. 将有效的序列连接成一个新的张量,作为最终输出。
        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim
         # 37. 返回生成的序列。
        return out_seq

    # 38. 这个方法用于根据提示生成经验,包括模型输出、奖励分数和价值估计。
    # 39. prompts 是输入的提示,mask 是相应的掩码张量,step 是训练步数。
    # 52. 使用 detach() 方法将价值估计从计算图中分离出来,避免梯度计算。
    def generate_experience(self, prompts, mask, step):
        self.eval()  # 40. 将模型切换到评估模式,以生成序列。
        generate_start = time.time()
        # 41. 调用 _generate_sequence 方法生成序列,并记录生成时间。
        seq = self._generate_sequence(prompts, mask, step)
        generate_end = time.time()

        # 42. 如果生成的序列无效,则使用上一次有效的序列。
        # 43. 否则,将当前生成的经验存储为最后一次有效经验。
        if seq is None:
            assert self.last_generated_experience is not None, f'Invalid generated experience at {step=}'
            prompts = self.last_generated_experience['prompts']
            seq = self.last_generated_experience['seq']
        else:
            self.last_generated_experience = {'prompts': prompts, 'seq': seq}

        self.train() # 44. 生成完成后,将模型切换回训练模式。
        # 45. 获取填充标记 ID,并根据生成的序列计算注意力掩码。
        # TODO 官方的代码会强制使用非pad_token_id外所有的token都计算, 是否可以根据experience中的anwser中第一个命中了eos_toen_id的索引, 然后将后面的单词全部mask掉????
        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        # 46. 使用 actor 模型计算输出,传入生成的序列和注意力掩码。
        # 47. 使用 no_grad() 上下文管理器,避免在推理阶段计算梯度。
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            # 48. 使用参考模型计算输出,传入生成的序列和注意力掩码。即SFT
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            # 49. 使用奖励模型计算奖励分数,传入生成的序列、注意力掩码和提示长度。
            # 50. 使用 detach() 方法将奖励分数从计算图中分离出来,避免梯度计算。
            reward_score = self.reward_model.forward_value(seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach()
            # 51. 使用 critic 模型计算价值估计,传入生成的序列和注意力掩码。
            # 52. 使用 detach() 方法将价值估计从计算图中分离出来,避免梯度计算。
            # 53. 只返回序列的价值估计,而不包括最后一个标记的估计,因为它是填充标记。
            values = self.critic_model.forward_value(seq, attention_mask, return_value_only=True).detach()[:, :-1]

        # 54. 获取 actor 和 ref 模型的 logits 输出。
        # 55. 如果配置了计算 FP32 损失,则将 logits 转换为 float 类型,以提高数值稳定性。
        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        self.generate_time = generate_end - generate_start   # 56. 计算生成序列所需的时间,可用于监控和调试

        # 57. 将提示、log 概率、参考 log 概率、价值估计、奖励分数、输入序列和注意力掩码打包成一个字典,作为生成经验的输出。
        # 58. gather_log_probs 是一个辅助函数,用于从 logits 中提取指定标记的 log 概率,这里是提取生成序列(除去起始标记)对应的 log 概率。
        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),     # 切片:-1, 这是因为我们要错位取到 <|endoftext|> 这个位置
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]), # 对于CausalLM来说，logits第t个时间步的置信值是为了预测第t+1步的seq token，因此logitsl， ：-1，：1与seq［：，1］才是“预测与标签”的关系：
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask
        }

    # 这个方法实现了强化学习中的奖励计算。
    # 它结合了 KL 散度奖励和外部奖励模型的分数,并对奖励值进行了裁剪以保持数值稳定性。
    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        # 61. 首先计算 KL 散度估计,作为基础奖励。KL 散度表示生成序列与参考序列的差异程度,用于鼓励生成不同于参考序列的输出。
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        # 62. 初始化奖励值为 KL 散度估计。
        rewards = kl_divergence_estimate
        # 63. 计算每个序列的结束位置,以便在该位置添加奖励分数。
        start = prompts.shape[1] - 1   # 对于CausalLM来说，logits第t个时间步的置信值是为了预测第t+1步的seq token，因此logitsl， ：-1，：1与seq［：，1］才是“预测与标签”的关系：
        ends = start + action_mask[:, start:].sum(1) + 1
        # 64. 将奖励分数裁剪到一个合理的范围内,避免过大或过小的值。
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        # 65. 对于每个批次,在序列结束位置添加裁剪后的奖励分数
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]
        # 66. 返回计算得到的奖励值。
        return rewards

    # 这个方法实现了 RLHF 训练的一个步骤。
    # 它根据当前生成的序列计算 actor 和 critic 的损失函数,并更新模型参数。
    # 同时还处理了溢出情况,并计算了优势函数和折返函数,这是强化学习算法的核心部分。
    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        # 3. 从输入字典中获取提示张量。
        prompts = inputs['prompts']                 # (B,T/2)
        # 4. 从输入字典中获取旧策略的 log 概率张量。
        log_probs = inputs['logprobs']              # (B,T-1)
        # 5. 从输入字典中获取参考模型的 log 概率张量。
        ref_log_probs = inputs['ref_logprobs']      # (B,T-1)
        # 6. 从输入字典中获取奖励分数张量。
        reward_score = inputs['rewards']            # (B,)
        # 7. 从输入字典中获取旧价值估计张量。
        values = inputs['value']                    # (B,T-1)
        # 8. 从输入字典中获取注意力掩码张量。
        attention_mask = inputs['attention_mask']   # (B,T)
        # 9. 从输入字典中获取生成的序列张量。
        seq = inputs['input_ids']                   # (B,T)
        # 10. 计算提示的最后一个位置索引,用于后续处理。
        start = prompts.size()[-1] - 1
        # 11. 从注意力掩码中获取动作掩码, 错一位是因为auto regressive需要错一位
        action_mask = attention_mask[:, 1:]         # (B,T-1)

        old_values = values
        with torch.no_grad():
            # 12. 调用 compute_rewards 方法计算旧奖励,传入提示、旧策略 log 概率、参考 log 概率、奖励分数和动作掩码。
            # 13. old_rewards 是KL散度修正后的奖励, (B,T-1)
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            # 14. 计算每个序列的对话结束位置索引。
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            # 15. 对于每个序列,将对话结束后的奖励和价值估计置零。
            # 16. 这是因为在对话结束后,不应该再有奖励或价值估计,否则会导致优势值和回报值计算错误。
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            # 17. 调用 get_advantages_and_returns 方法计算优势值和回报值,传入旧价值估计、旧奖励和提示的开始位置。
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        # 18. 将生成的序列和注意力掩码打包成一个字典,作为模型输入。
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        # 使用 actor 模型计算 logits,传入输入字典,并禁用缓存。从 actor 模型的 logits 中提取 log 概率,用于计算 actor 损失。
        # 19. 计算新策略
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        # 21. 调用 actor_loss_fn 方法计算 actor 损失,传入新的 log 概率、旧的 log 概率、优势值和动作掩码。
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        # 22. 反向传播 actor 损失,以更新 actor 模型参数。
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            # 23. 如果不需要对齐溢出检查,则直接更新 actor 模型参数。
            self.actor_model.step()

        # 24. 使用 critic 模型计算新的价值估计,传入输入字典,只返回价值估计,并禁用缓存。调用 critic_loss_fn 方法计算 critic 损失,传入新的价值估计、旧的价值估计、回报值和动作掩码。
        # 计算新价值
        value = self.critic_model.forward_value(**batch,return_value_only=True,use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
        # 26. 反向传播 critic 损失,以更新 critic 模型参数。
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            # 27. 如果配置了对齐溢出检查,则检查 actor 模型的梯度是否发生了溢出。
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            # 28. 检查 critic 模型的梯度是否发生了溢出。
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)
            # 29. 获取当前进程的进程号,用于打印日志信息。
            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                # 30. 如果 actor 模型发生了溢出,而 critic 模型没有发生溢出,则跳过更新两个模型的步骤,并打印相关信息。
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                # 31. 如果 critic 模型发生了溢出,而 actor 模型没有发生溢出,则跳过更新两个模型的步骤,并打印相关信息。
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                # 32. 如果 actor 模型和 critic 模型都发生了溢出,则跳过更新两个模型的步骤,并打印相关信息。
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()   # 33. 如果没有发生溢出,则更新 actor 模型参数。

        self.critic_model.step()  # 34. 更新 critic 模型参数。
        # 35. 返回 actor 损失和 critic 损失,用于监控和评估训练过程。
        return actor_loss, critic_loss

    # 79. 这个方法用于获取 actor 和 critic 模型的梯度溢出状态。
    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        # 80. 如果使用 bf16 精度,则不期望发生梯度溢出,因此直接返回 False。
        if self.args.dtype == "bf16":
            return False, False
        # 81. 获取 actor 和 critic 模型的梯度溢出状态。
        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow
        return actor_overflow, critic_overflow

    # 这个方法实现了 PPO 算法中的策略梯度损失函数。
    # 它采用了裁剪策略,以避免过大的更新步长,从而提高训练稳定性。
    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        # 94. 计算对数概率比,并将其与动作掩码相乘。
        log_ratio = (logprobs - old_logprobs) * mask
        # 95. 计算概率比。
        ratio = torch.exp(log_ratio)
        # 96. 计算策略梯度损失的两个部分。
        # 97. 第一部分直接使用概率比,第二部分使用裁剪后的概率比,以避免过大的更新步长。
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        # 98. 计算最终的策略梯度损失,取两个部分的最大值,并对应用动作掩码后求和。
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        # 99. 返回策略梯度损失。
        return pg_loss

    # 这个方法实现了价值函数损失的计算。
    # 它也采用了裁剪策略,以限制更新步长,从而提高训练稳定性。
    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        # 102. 对当前序列的价值估计进行裁剪,以限制更新步长。
        values_clipped = torch.clamp(values,old_values - self.cliprange_value,old_values + self.cliprange_value,)
        # 103. 如果配置了计算 FP32 精度的损失,则将价值估计转换为 FP32 格式。
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()
        # 104. 计算价值函数损失的两个部分,分别使用当前价值估计和裁剪后的价值估计。
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        # 105. 计算最终的价值函数损失,取两个部分的最大值,并对应用动作掩码后求和。
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        # 106. 返回价值函数损失。
        return vf_loss
    
    # 这个方法实现了 GAE 算法,用于计算强化学习训练中的关键量:优势函数 (advantage)和折返函数(return)。。
    # 优势函数表示当前动作相对于基线的优势,而折返函数则是预期的累积奖励。
    # 这两个量在策略梯度和价值函数损失的计算中扮演着重要角色。
    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        # 97. 使用 GAE (Generalized Advantage Estimation) 算法计算优势值。
        # 98. 从序列末尾开始反向计算每个时间步的优势值。
        # 99. 计算过程中使用折扣因子 gamma 和 lambda 参数。
        # 100. 最终得到优势值和回报,其中回报等于优势值加上价值估计。
        # 101. 使用 detach() 方法将优势值从计算图中分离出来,避免梯度计算。
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        # 102. 这个方法用于验证模型是否处于训练模式。
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        # 103. 这个方法用于验证模型是否处于评估模式。
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        # 104. 这个方法用于将模型切换到训练模式。
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        # 105. 这个方法用于将模型切换到评估模式。
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    # 106. 这个方法用于打印模型参数的范数,可用于监控和调试。
    # 107. tag 是一个字符串标签,用于标识打印的内容。
    def dump_model_norms(self, tag):
        # 108. 使用 get_model_norm 函数获取每个模型的参数范数。
        # 109. 使用 print_all_ranks 函数打印每个模型的参数范数,并附加标签和 rank 信息。
        # 110. 这些信息可用于监控模型参数的变化,并发现潜在的问题,如梯度爆炸等。
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


# 这个方法实现了无监督的 PPO 算法训练,包括前向计算、损失计算、反向传播和参数更新。
# 无监督训练通常用于预训练语言模型,以捕获语言的统计规律和语义信息。
# 该方法接受输入数据和无监督损失系数作为参数,并返回无监督损失值。
class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):
    # 1. 这是一个名为 DeepSpeedPPOTrainerUnsupervised 的类,继承自 DeepSpeedPPOTrainer 类。
    # 2. 该类用于实现无监督的 PPO 算法训练,通常用于预训练语言模型。
    def __init__(self, *args, **kwargs):   # 3. 初始化方法,接受任意数量的位置参数和关键字参数。
        # 4. 调用父类 DeepSpeedPPOTrainer 的初始化方法,以初始化共享的属性和方法。
        super().__init__(*args, **kwargs)

    # 5. 这个方法用于无监督训练,接受 inputs 和 unsup_coef 作为输入参数。
    # 6. inputs 是一个包含输入数据的字典,例如输入序列和注意力掩码等。
    # 7. unsup_coef 是无监督损失的系数,用于控制无监督损失对总损失的贡献。
    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        # 8. 调用 _validate_training_mode 方法,验证模型是否处于训练模式。
        self._validate_training_mode()
        # 9. 使用 actor 模型对输入数据进行前向计算,获取输出。
        # 10. use_cache=False 表示不使用缓存,以节省内存。
        outputs = self.actor_model(**inputs, use_cache=False)
        # 11. 从模型输出中获取损失值。
        loss = outputs.loss
        # 12. 计算加权损失,即无监督损失乘以系数。
        # 13. 反向传播加权损失,计算模型参数的梯度。
        self.actor_model.backward(unsup_coef * loss)
        # 14. 更新 actor 模型的参数,基于计算出的梯度。
        self.actor_model.step()
        # 15. 返回无监督损失值,用于监控和评估训练过程。
        return loss
