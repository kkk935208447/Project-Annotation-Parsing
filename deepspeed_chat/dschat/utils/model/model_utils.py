# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig

from dschat.utils.model.reward_model import RewardModel
from dschat.utils.utils import load_state_dict_into_model, print_rank_0


# 修改dropout
def configure_dropout(model_config, dropout):
    if dropout is not None:
        # 6. 遍历模型配置中与 Dropout 相关的几个属性,包括 dropout、attention_dropout、hidden_dropout 和 activation_dropout。
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            # 7. 检查模型配置是否包含当前遍历的属性。
            if hasattr(model_config, key):
                # 8. 打印一条信息,说明正在设置对应属性的值。
                print(f"Setting model_config.{key} to {dropout}")
                # 9. 使用 setattr 函数将模型配置中对应的属性设置为提供的 dropout 值。
                setattr(model_config, key, dropout)


# 该函数的主要目的是修改CausalLM模型的前向传播函数,使其在计算损失函数时使用fp32精度。具体来说,它执行以下操作:
# 定义一个新的causal_lm_forward函数,该函数封装了模型的原始前向传播逻辑。
# 在causal_lm_forward函数中,先调用模型的原始前向传播函数获得输出,但不传递labels参数。
# 从输出中获取语言模型的logits。
# 如果提供了labels,则将labels移动到与logits相同的设备上,以支持模型并行。
# 对logits和labels进行移位,使得第i个token预测第i+1个token。
# 将移位后的logits和labels展平,并使用PyTorch的CrossEntropyLoss计算损失函数。
# 如果输出是字典形式,则将损失函数添加到输出字典中;否则,将损失函数与其他输出组合成一个元组返回。
# 保存模型的原始前向传播函数,并将模型的前向传播函数替换为新的causal_lm_forward函数。

# 通过这种方式,该函数实现了以下目标:
# 确保损失函数的计算使用fp32精度,而不受混合精度训练设置的影响。
# 支持模型并行,将labels移动到与logits相同的设备上。
# 处理序列生成任务中的标签移位问题。
# 保持模型的其他输出不变,只修改了损失函数的计算方式。
# 该函数可用于在使用DeepSpeed进行大型语言模型训练时,确保损失函数的计算精度,并支持模型并行等优化技术。
def causal_lm_model_to_fp32_loss(model):
    """ Convert CausalLM model to calculate loss in fp32 """
    # 1. 该函数的目的是将CausalLM模型转换为在fp32精度下计算损失函数。

    def causal_lm_forward(
        input_ids=None,  # 2. 输入的token ids
        past_key_values=None,  # 3. 用于序列并行的过去的key/value状态
        attention_mask=None,   # 4. 注意力掩码,指定要忽略的位置
        head_mask=None,        # 5. 头部掩码,指定要忽略的注意力头
        inputs_embeds=None,    # 6. 输入的嵌入,可用于绕过嵌入层
        labels=None,           # 7. 目标标签,用于计算损失函数
        use_cache=None,         # 8. 是否使用缓存key/value状态
        output_attentions=None,     # 9. 是否输出注意力权重
        output_hidden_states=None,  # 10. 是否输出隐藏状态
        return_dict=None,           # 11. 是否以字典形式返回输出    
        **deprecated_arguments,     # 12. 废弃的参数
    ):
        # 13. 根据模型类型设置kwargs,对于llama模型,不需要head_mask参数
        kwargs = dict() if model.config.model_type == "llama" else dict(
            head_mask=head_mask)
        
        # 14. 调用模型的原始前向传播函数,获得输出
        output = model.__original_forward__(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,        # 15. 在这里不传递labels,因为需要在后面进行特殊处理
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs)

        # 16. 检查输出是否为字典形式
        return_dict = isinstance(output, dict)
        lm_logits = output.logits if return_dict else output[0]
        loss = None

        # 18. 如果提供了标签,则计算损失函数
        if labels is not None:
            # move labels to correct device to enable model parallelism
            # 19. 将标签移动到与logits相同的设备上,以支持模型并行
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            # 20. 对logits进行移位,使得第i个token预测第i+1个token, .float()转换为fp32
            shift_logits = lm_logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 21. 获取批次大小,序列长度和词表大小
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            # 22. 将logits和标签展平,以便计算交叉熵损失函数
            # TODO 疑问: CrossEntropyLoss 是否需要给定ignore_index. 
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length))

        # 23. 如果输出不是字典形式,则以元组的形式返回损失和其他输出
        if not return_dict:
            # re-pack output with fp32 loss
            return ((loss, ) + output) if loss is not None else output

        # 24. 如果输出是字典形式,则将损失函数添加到输出字典中
        output.loss = loss
        return output
    # 25. 保存模型的原始前向传播函数
    model.__original_forward__ = model.forward
    # 26. 将模型的前向传播函数替换为新的causal_lm_forward函数
    model.forward = causal_lm_forward


# 该代码使用了以下技术:
# Hugging Face Transformers:该代码使用Hugging Face Transformers库加载和创建预训练的transformer模型,如GPT-2和T5。
# DeepSpeed:DeepSpeed是一个深度学习优化库,用于提高大型模型训练的效率和性能。该代码使用DeepSpeed的ZeRO优化技术,通过分片和优化内存利用来减少模型训练所需的内存占用。
# RLHF (Reinforcement Learning from Human Feedback):RLHF是一种通过人类反馈进行强化学习的技术,用于微调大型语言模型以获得更好的行为。该代码支持使用RLHF进行训练。
# Dropout:Dropout是一种正则化技术,通过在训练期间随机丢弃一些神经元来防止过拟合。该代码允许指定Dropout比例来控制正则化强度。
# 词表调整:为了优化内存利用率,该代码会调整模型的词表大小,使其为8的倍数,这是DeepSpeed优化所需的。
# 该代码的主要目的是创建一个预训练的transformer模型,同时支持以下功能:

# 从不同来源(Hugging Face或TensorFlow)加载预训练权重。
# 启用DeepSpeed的ZeRO优化,以减少训练所需的内存占用。
# 支持RLHF训练,通过人类反馈进行微调。
# 配置Dropout正则化强度。
# 调整词表大小以优化内存利用率。
# 通过这些功能,该代码为大型语言模型的高效训练提供了便利,并支持了一些最新的技术,如DeepSpeed和RLHF。
# def create_hf_model(model_class,  # Hugging Face模型类
#                     model_name_or_path,  # 模型权重文件的路径或名称
#                     tokenizer,   # 用于文本TokenizeToken的分词器
#                     ds_config=None,  # DeepSpeed配置,默认为None
#                     rlhf_training=False,  # 是否进行RLHF训练,默认为False
#                     dropout=None):  # Dropout率,默认为None
def create_hf_model(model_class,  # 1. model_class是指定的Hugging Face Transformers模型类,例如GPT2LMHeadModel或T5ForConditionalGeneration,用于创建特定类型的预训练语言模型
                    model_name_or_path,  # 2. model_name_or_path是预训练模型的名称或本地路径,用于从中加载模型权重和配置
                    tokenizer,  # 3. tokenizer是用于文本标记化的tokenizer对象,通常与所使用的预训练模型相对应
                    ds_config=None,  # 4. ds_config是DeepSpeed配置字典,用于启用DeepSpeed相关优化,如ZeRO优化等,以减少大型模型训练所需的内存占用
                    rlhf_training=False,  # 5. rlhf_training是一个布尔值,指示是否进行RLHF(Reinforcement Learning from Human Feedback)训练,RLHF是一种通过人类反馈进行强化学习的技术,用于微调大型语言模型以获得更好的行为
                    dropout=None):  # 6. dropout是指定的dropout比例,用于控制模型的正则化强度,dropout是一种通过在训练期间随机丢弃一些神经元来防止过拟合的技术
    
    # 7. 从预训练模型路径加载模型配置，AutoConfig.from_pretrained是Hugging Face Transformers库中的一个函数，用于自动加载与给定预训练模型相对应的配置
    model_config = AutoConfig.from_pretrained(model_name_or_path)

    # 8. 根据指定的dropout比例配置模型的dropout层，configure_dropout是一个自定义函数，用于在模型配置中设置dropout比例
    configure_dropout(model_config, dropout)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    # 9. 创建DeepSpeed配置对象，仅在DeepSpeed的ZeRO优化阶段为3时启用，ZeRO优化是DeepSpeed库中的一种优化技术，通过分片和优化内存利用来减少大型模型训练所需的内存占用，阶段3是ZeRO优化的最高阶段，提供最大程度的优化
    # 这是为了避免全局效果，因为DeepSpeed配置对象可能会影响其他部分的代码，这是一种防止副作用的编程实践
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        # 10. HfDeepSpeedConfig是Hugging Face Transformers库中的一个类，用于将DeepSpeed配置与Transformers模型集成
        # TODO: 疑问: HfDeepSpeedConfig 的对象 dschf 官方代码没有使用? 
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    
    # model.from_config是从一个配置文件中加载模型的参数和结构,然后初始化一个新的模型实例。只载入模型的结构和超参数,需要重新训练模型参数。
    if rlhf_training:
        # the weight loading is handled by create critic model
        # 11. 如果进行RLHF训练，则直接从模型配置创建模型实例，因为RLHF训练需要特殊的模型初始化方式，权重加载由create_critic_model函数处理
        model = model_class.from_config(model_config)
    # model.pretrained是从一个预训练好的模型checkpoint中加载模型参数,然后用这些参数初始化一个新的模型实例. 除了载入模型结构和超参数,还会载入预训练模型的参数。
    else:
        # 12. 否则，从预训练模型路径加载模型权重，用于从预训练权重创建模型实例
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path), # 13. 如果模型路径包含".ckpt"，则从TensorFlow权重加载，否则从PyTorch权重加载
            config=model_config)

    # 14. 设置模型配置中的end_token_id和pad_token_id为tokenizer的eos_token_id，这是一种常见的做法，用于确保模型和tokenizer之间的一致性
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # 15. 调整模型的词表大小，使其为8的倍数，这是DeepSpeed优化所需的，因为DeepSpeed会将模型的参数分片，每个分片的大小必须是8的倍数
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


# 在这个版本中,我为每一行代码添加了详细的注释,解释了参数、函数和使用的技术,以及它们在上下文中的作用。以下是一些关键点:

# 该函数用于创建一个奖励模型 (Reward Model),用于评估生成的文本序列的质量,并为优化语言模型提供奖励信号。
# 函数使用了 DeepSpeed 库进行模型加载和优化,并考虑了 ZeRO 优化阶段的兼容性。
# 函数支持从模型检查点加载预训练的奖励模型,以继续进行 RLHF 训练。
# 函数使用了 Hugging Face 库加载预训练模型,并创建了一个 RewardModel 实例作为奖励模型。
# 函数还支持一些其他选项,如指定输入序列开头的填充 token 数量、应用 Dropout 正则化和使用混合精度计算损失。
# 函数使用了一些自定义的函数,如 create_hf_model、snapshot_download、load_state_dict_into_model 等,这些函数可能在其他地方定义,用于处理特定的任务
def create_critic_model(model_name_or_path,   # 1. model_name_or_path 参数是预训练模型的名称或路径,用于加载预训练模型。
                        tokenizer,  # 2. tokenizer 参数是一个用于文本标记化和解码的 Tokenizer 对象,必须与预训练模型相匹配。
                        ds_config,  # 3. ds_config 参数是一个 DeepSpeed 配置对象,用于设置 DeepSpeed 库的各种参数,如优化策略、混合精度等。
                        num_padding_at_beginning=0,  # 4. num_padding_at_beginning 参数指定了输入序列开头的填充 token 数量,默认为 0。有些模型(如 OPT)在序列开头添加了特殊的填充 token,这个参数用于处理这种情况。
                        rlhf_training=False,  # 5. rlhf_training 参数是一个布尔值,指示是否进行 RLHF训练。RLHF 是一种通过人类反馈来微调和优化语言模型的技术。
                        dropout=None,   # 6. dropout 参数用于指定应用于模型的 Dropout 比例,可以用于正则化和防止过拟合。如果未提供,则使用模型的默认 Dropout 设置。
                        zero_stage=0,    # 7. zero_stage 参数指定了 DeepSpeed 的 ZeRO 优化阶段,用于控制模型参数的分片和优化策略。0 表示不使用 ZeRO 优化,1/2/3 表示不同的优化策略。
                        compute_fp32_loss=False):  # 8. compute_fp32_loss 参数是一个布尔值,指示是否使用 32 位浮点数计算损失,默认为 False。使用 32 位浮点数可以提高计算精度,但会占用更多内存。
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule
    # 9. OPT 模型系列在序列开头总是添加一个填充 token 的情况,但在其他模型中未观察到这种情况,因此不确定这是否是一个普遍规则。

    import time

    start = time.time()
    # 12. 调用 create_hf_model 函数创建一个 Hugging Face 模型实例。
    # create_hf_model 是一个自定义函数,它利用 Hugging Face 的 AutoModel 类从预训练模型创建一个模型对象。
    critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, dropout)
    end = time.time()
    # 14. 使用 print_rank_0 函数打印模型创建所用时间,该函数确保只在单个进程中打印,以避免在多 GPU 训练环境中的重复打印。
    print_rank_0(f">Creating model from_config took {end - start} seconds",
                 None)

    # 15. 创建一个 RewardModel 实例,作为奖励模型。
    # RewardModel 是一个自定义类,用于评估生成的文本序列的质量,并为优化语言模型提供奖励信号。
    critic_model = RewardModel(critic_model,
                                tokenizer,
                                num_padding_at_beginning=num_padding_at_beginning,
                                compute_fp32_loss=compute_fp32_loss)
    # 20. 如果进行 RLHF 训练,则加载预训练的奖励模型。
    if rlhf_training:
        # load critic model from checkpoint
        # 21. 从检查点加载预训练的奖励模型。
        # 22. 检查 model_name_or_path 是否是一个目录路径。
        # 如果不是目录路径,则假设它是一个远程模型路径,需要使用 snapshot_download 函数下载模型。
        # snapshot_download 是一个函数,用于从远程位置下载模型快照。
        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        # 23. 构造模型检查点文件的路径。
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        # 26. 使用 torch.load 函数加载模型检查点的状态字典,并将其映射到 CPU 以节省内存。
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')
        end = time.time()
        # 28. 使用 print_rank_0 函数打印加载模型检查点所用时间。
        print_rank_0(f">Creating model from_config took {end - start} seconds",
                     None)

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        # 29. 下面的代码是用于从模型检查点加载奖励模型,并确保与 DeepSpeed 的 ZeRO 优化阶段 3 兼容。
        #这个功能未来可能会被移动到 DeepSpeed 的检查点加载 API 中。
        start = time.time()
        # 31. 调用 load_state_dict_into_model 函数,将模型检查点的状态字典加载到奖励模型中,并考虑了 ZeRO 优化阶段的兼容性。
        # load_state_dict_into_model 是一个自定义函数,用于将模型状态字典加载到目标模型中,并处理 ZeRO 优化阶段带来的一些复杂性。
        load_state_dict_into_model(critic_model,
                                   model_ckpt_state_dict,
                                   "",
                                   zero_stage=zero_stage)
        end = time.time()
        # 33. 使用 print_rank_0 函数打印加载模型检查点并更新到奖励模型所用时间。
        print_rank_0(f">Creating model from_config took {end - start} seconds",
                     None)
    # 34. 返回创建的奖励模型实例。
    return critic_model
