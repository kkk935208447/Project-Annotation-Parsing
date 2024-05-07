# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn


## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
# 代码的来源,它是从 CarperAI/trlx 仓库的 reward_model.py 文件修改而来。
class RewardModel(nn.Module):
    # 实现奖励模型(Reward Model)。
    # 奖励模型是 RLHF技术中的一个关键组件,用于评估生成的文本序列的质量,并为优化语言模型提供奖励信号。
    # RLHF 是一种通过人类反馈来微调和优化语言模型的技术,它结合了强化学习和人类评分,可以使语言模型生成更加符合人类期望的输出。
    def __init__(self,
                 base_model,  # 3. base_model 参数是一个预训练的语言模型实例,例如 GPT-2、OPT 或 GPT-Neo,用于生成文本序列。
                 tokenizer,  # 4. tokenizer 参数是一个用于文本标记化和解码的 tokenizer 对象,通常与 base_model 相对应。
                 # 8. num_padding_at_beginning 参数指定了输入序列开头的填充数量,一般是0 或 1,默认为 0。有些模型在序列开头会添加特殊的填充 token,这个参数用于处理这种情况。
                 num_padding_at_beginning=0, 
                 # 9. compute_fp32_loss 参数是一个布尔值,指示是否使用 32 位浮点数计算损失,默认为 False。使用 32 位浮点数可以提高计算精度,但会占用更多内存。
                 compute_fp32_loss=False):  
        super().__init__()
        # 7. 获取 base_model 的配置对象。
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        # 11. 根据模型配置创建奖励头(Reward Head),它是一个线性层,用于从模型的隐藏状态中预测奖励值。
        # 12. 每一个token的奖励值是一个标量,表示生成的文本序列的质量分数,用于在 RLHF 中计算奖励。
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use "word_embed_proj_dim" as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            # word_embed_proj_dim 是 OPT 模型中一个特殊的维度,直接用于将词嵌入投影到一个低维空间,以减少计算复杂度
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,1,bias=False)

        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)

        # 11. 将 base_model 作为奖励模型的底层语言模型。
        self.rwtransformer = base_model
        # 18. 获取填充 token 的 ID,用于处理序列中的填充。
        self.PAD_ID = tokenizer.pad_token_id
        # 计算loss是否转化为fp32
        self.compute_fp32_loss = compute_fp32_loss

    def gradient_checkpointing_enable(self):
        # 19. 启用基础语言模型的梯度检查点功能,用于节省内存。
        # 20. 梯度检查点是一种训练技术,它通过重新计算部分激活值来减少内存使用,从而允许训练更大的模型。
        # 21. 在 RLHF 中,由于需要处理大量的输入数据,启用梯度检查点可以有效地节省内存。
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        # 14. 禁用基础语言模型的梯度检查点功能。
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,   # 15. input_ids 参数是一个批次的输入 token ID 序列。
                past_key_values=None,  # 24. past_key_values 参数用于传递上一个时间步的键值对,以支持自回归生成。在生成长序列时,可以利用这个参数来提高效率。
                attention_mask=None,  # 25. attention_mask 参数用于指定哪些 token 应该被关注,哪些应该被遮蔽
                position_ids=None,  # 18. position_ids 参数用于指定每个 token 的位置编码。
                head_mask=None,     # 27. head_mask 参数用于控制哪些多头注意力头被使用,可以用于加速计算或进行模型剪枝。
                inputs_embeds=None, # 28. inputs_embeds 参数用于直接传递输入的嵌入向量,替代 input_ids。这可以用于在不重新编码输入的情况下,对已有的嵌入进行进一步处理。
                use_cache=False):  # 29. use_cache 参数指示是否使用缓存键值对以加速自回归生成。在生成长序列时,使用缓存可以显著提高效率。
        loss = None

        # 30. 根据模型类型设置一些额外的参数,例如是否需要 head_mask。
        # llama模型时,没有head_mask参数选项
        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        # 31. 将输入传递给底层语言模型,获得 transformer_outputs。
        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        # 24. 从 transformer_outputs 中获取隐藏状态。
        hidden_states = transformer_outputs[0]
        # 34. 将隐藏状态传递给奖励头,获得奖励值。奖励值是一个标量序列,表示每个 token 的质量分数。
        # logits, 没有进行log
        rewards = self.v_head(hidden_states).squeeze(-1)
        # 26. 初始化一个列表,用于存储好回答的平均分数。
        chosen_mean_scores = []
        # 27. 初始化一个列表,用于存储坏回答的平均分数。
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        # 37. 将输入和奖励分为两部分,选择和拒绝。这是 RLHF 中的一种技术,通过比较两个候选序列的质量来计算loss。
        assert len(input_ids.shape) == 2  # 38. 确保 input_ids 的形状是二维的,即 (batch_size, sequence_length)。
        bs = input_ids.shape[0] // 2        # 39. 批次大小为 input_ids 的行数除以 2,因为输入中包含了选择和拒绝两部分。
        seq_len = input_ids.shape[1]        

        chosen_ids = input_ids[:bs]  # bs x seq x 1  # 32. 选择部分的 token ID 序列。
        rejected_ids = input_ids[bs:]       # 33. 拒绝部分的 token ID 序列。
        chosen_rewards = rewards[:bs]       # 34. 选择部分的奖励值。
        rejected_rewards = rewards[bs:]     # 35. 拒绝部分的奖励值。

        # Compute pairwise loss. Only backprop on the different tokens before padding
        # 45. 计算成对损失,只在不同的 token 之前反向传播。这是 RLHF 中的另一种技术,通过比较两个候选序列的差异来计算损失。
        loss = 0.
        for i in range(bs): # 每条样本长短不一,需要一条一条比较
            chosen_id = chosen_ids[i]  # 37. 当前批次的选择 token ID 序列。
            rejected_id = rejected_ids[i]   # 38. 当前批次的拒绝 token ID 序列。
            chosen_reward = chosen_rewards[i]  # 39. 当前批次的选择奖励值序列。
            rejected_reward = rejected_rewards[i] # 40. 当前批次的拒绝奖励值序列。

            # 50. 获得所有==self.PAD_ID的索引
            c_inds = (chosen_id == self.PAD_ID).nonzero()
             # 51. 对于 OPT 模型,需要使用第二个填充 token 作为序列结束位置,因为它在序列开头添加了特殊的填充 token。
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            # 43. 找到选择和拒绝序列开始不同的位置。
            check_divergence = (chosen_id != rejected_id).nonzero()

            # 53. 如果没有不同的位置,则使用选择序列的最后一个 token 作为参考。这种情况可能发生在生成的序列完全相同的时候。
            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                # 54. 否则,找到拒绝序列中第一个填充 token 的位置,用于确定有效序列的结束位置。
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)  # 46. 选择较长的序列作为结束位置。
                # 47. 获取两个序列开始不同的位置。
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0  # 57. 确保不同位置大于 0,避免出现无效的序列。
            # 58. 从不同位置到结束位置截取选择和拒绝的奖励值序列,用于计算损失。
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            # 59. 将选择和拒绝序列的最后一个 token 的奖励值作为平均分数存储,用于计算最终的奖励。
            # 注意: 因为 deepspeed chat 把 eos id == pad id, 实际上奖励值取最后一个有效token时, eos 或 pad id 都会被去除, 
            # 因此, 实际上<|endoftext|> 所对应的logits就是整个句子的奖励值reward.
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

             # 60. 如果设置了 compute_fp32_loss,则将奖励值转换为 32 位浮点数,以提高计算精度。
            if self.compute_fp32_loss:
                c_truncated_reward = c_truncated_reward.float()
                r_truncated_reward = r_truncated_reward.float()
            # 61. 计算截取的选择和拒绝奖励值序列之间的对数 sigmoid 损失,并累加到总损失中。
            # 62. 对数 sigmoid 损失是一种常用的二分类损失函数,它可以有效地衡量两个序列之间的质量差异。
            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()
        # 63. 计算平均损失,用于优化奖励模型的参数。
        loss = loss / bs
        # 64. 将平均分数转换为 PyTorch 张量,以便进一步处理。
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        # 65. 返回损失、选择平均分数和拒绝平均分数,用于后续的 RLHF 训练过程。
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self,
                      input_ids=None,  # 66. input_ids 参数是一个批次的输入 token ID 序列。
                      attention_mask=None,   # 67. attention_mask 参数用于指定哪些 token 应该被关注,哪些应该被遮蔽。
                      past_key_values=None,  # 58. past_key_values 参数用于传递上一个时间步的键值对,以支持自回归生成。
                      position_ids=None,  # 69. position_ids 参数用于指定每个 token 的位置编码。
                      head_mask=None,    # 70. head_mask 参数用于控制哪些多头注意力头被使用。
                      inputs_embeds=None,   # 61. inputs_embeds 参数用于直接传递输入的嵌入向量,替代 input_ids。
                      return_value_only=False,  # 72. return_value_only 参数指示是否只返回奖励值,而不计算损失。
                      prompt_length=0,   # 73. prompt_length 参数指定输入序列中提示的长度。但在某些场景下,输入序列可能包含一个提示部分和生成部分。
                      use_cache=False):   # 74. use_cache 参数指示是否使用缓存键值对以加速自回归生成。

        # 30. 根据模型类型设置一些额外的参数,例如是否需要 head_mask。
        # llama模型时,没有head_mask参数选项
        if self.config.model_type == "llama":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        # 66. 将输入传递给底层语言模型,获得 transformer_outputs。
        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        # 67. 从 transformer_outputs 中获取隐藏状态。
        hidden_states = transformer_outputs[0]
        # 68. 将隐藏状态传递给奖励头,获得奖励值。
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values  # 69. 如果只需要返回奖励值,则直接返回
        
        else:
            #  80. 否则,计算选择的结束分数。结束分数是指序列最后一个 token 的奖励值,用于评估序列的质量。
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            # 71. 获取批次大小。
            bs = values.size(0)
            # 72. 获取序列长度
            seq_len = input_ids.shape[1]
            # 73. 初始化一个列表,用于存储选择的结束分数。
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]  # 74. 当前批次的输入 token ID 序列。
                value = values[i]  # 75. 当前批次的奖励值序列。

                # 76. 找到输入序列中第一个填充 token 的位置,从 prompt_length 开始搜索。
                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning

                # 77. 将选择和拒绝序列的最后一个 token 的奖励值作为平均分数存储,用于计算最终的奖励。
                # 注意: 因为 deepspeed chat 把 eos id == pad id, 实际上奖励值取最后一个有效token时, eos 或 pad id 都会被去除, 
                # 因此, 实际上<|endoftext|> 所对应的logits就是整个句子的奖励值reward.
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            # 78. 返回奖励值和选择的结束分数。
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
