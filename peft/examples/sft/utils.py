import os
from enum import Enum

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig

# DEFAULT_CHATML_CHAT_TEMPLATE是一个用于格式化聊天消息的jinja2模板字符串
# jinja2是一种流行的Python模板引擎,它允许在模板中嵌入Python代码,使模板更加动态和可编程
# 在这个模板中,{% for message in messages %} 是一个jinja2的for循环语句,用于遍历messages列表中的每个消息
# {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
# 这一部分定义了每条消息的格式化方式,包括:
#   1. <|im_start|>: 一个特殊标记,表示消息角色(如user、system或assistant)的开始
#   2. message['role']: 当前消息的角色,如user、system或assistant
#   3. \n: 换行符,用于在角色和消息内容之间添加新行
#   4. message['content']: 当前消息的实际内容
#   5. <|im_end|>: 一个特殊标记,表示消息内容的结束
#   6. \n: 换行符,用于在每条消息之后添加新行
# {% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}
# 这一部分是一个jinja2的条件语句,当循环遍历到最后一条消息时,如果add_generation_prompt为True,
# 则会在最后一条消息后添加'<|im_start|>assistant\n'作为提示,表示需要模型生成助手的回复
# 这种模板格式化方式的目的是将原始的聊天记录转换为适合语言模型输入的格式,以便进行对话生成任务
DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"


# DEFAULT_ZEPHYR_CHAT_TEMPLATE与DEFAULT_CHATML_CHAT_TEMPLATE类似,也是一个用于格式化聊天消息的jinja2模板
# 不同之处在于格式化方式和使用的特殊标记
# {% for message in messages %} 同样是一个用于遍历消息列表的for循环
# {% if message['role'] == 'user' %} 是一个条件语句,用于判断当前消息的角色是否为user
# 如果是user,则使用{{ '<|user|>\n' + message['content'] + eos_token }}将消息格式化为:
#   1. <|user|>: 用户角色的特殊标记
#   2. \n: 换行符
#   3. message['content']: 消息内容
#   4. eos_token: 句尾标记,如</s>
# {% elif message['role'] == 'system' %} 是另一个条件分支,用于判断当前消息的角色是否为system
# 如果是system,则使用{{ '<|system|>\n' + message['content'] + eos_token }}进行格式化
# {% elif message['role'] == 'assistant' %} 是第三个条件分支,用于判断当前消息的角色是否为assistant
# 如果是assistant,则使用{{ '<|assistant|>\n'  + message['content'] + eos_token }}进行格式化
# {% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}
# 这一部分与DEFAULT_CHATML_CHAT_TEMPLATE类似,当遍历到最后一条消息时,如果add_generation_prompt为True,
# 则会添加'<|assistant|>\n'作为提示,表示需要模型生成助手的回复
# 总的来说,这种格式化方式将原始聊天记录转换为适合语言模型输入的形式,但使用了不同的特殊标记
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

# ZephyrSpecialTokens是一个继承自str和Enum的枚举类
# 它定义了Zephyr聊天格式中使用的各种特殊标记,如用户标记、助手标记、系统标记等
# 枚举类的好处是可以将一组相关的常量组织在一起,并提供更好的可读性和类型安全性
# 每个特殊标记都被定义为一个类属性,其值为对应的字符串形式
# 例如,user = "<|user|>"表示用户标记的字符串形式为"<|user|>"
class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"      # 句尾标记,表示一个句子或序列的结束
    bos_token = "<s>"       # 句首标记,表示一个句子或序列的开始
    pad_token = "<pad>"     # 填充标记,用于将序列填充至指定长度

    # list方法是一个类方法,它返回一个列表,包含了该枚举类中所有特殊标记的字符串形式
    # 这个方法常用于初始化分词器(tokenizer)时,将这些特殊标记添加到词表中
    @classmethod
    def list(cls):
        return [c.value for c in cls]

# ChatmlSpecialTokens与ZephyrSpecialTokens类似,也是一个定义了Chatml聊天格式中使用的特殊标记的枚举类
# 不同之处在于具体的特殊标记字符串形式
# 例如,user标记在Chatml格式中为"<|im_start|>user",而在Zephyr格式中为"<|user|>"
class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

# create_datasets函数用于创建训练和测试数据集
# 参数包括:
#   tokenizer: 用于对文本进行分词(tokenization)和编码(encoding)的分词器对象
#   data_args: 包含数据相关配置的参数对象,如数据集名称、切分方式等
#   training_args: 包含训练相关配置的参数对象
#   apply_chat_template (bool): 是否应用聊天模板对数据进行预处理,默认为False
def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    # preprocess是一个内部函数,用于对数据样本进行预处理
    # 它接受一个字典样本作为输入,其中"messages"键对应一个列表,列表中的每个元素都是一个对话(conversation)
    def preprocess(samples):
        batch = []     # 初始化一个空列表,用于存储预处理后的对话
        # TODO 修改源码
        batch_tokens = []
        # 遍历样本中的每个对话
        for conversation in samples["messages"]:
            # 对每个对话应用tokenizer.apply_chat_template方法进行预处理
            # tokenize=False表示不执行分词操作,只进行格式化
            # https://huggingface.co/docs/transformers/main/zh/chat_templating
            # TODO 对源码进行修改
            chat_tmp = tokenizer.apply_chat_template(conversation, tokenize=False)
            batch.append(chat_tmp)
            chat_tmp_tokens = tokenizer.tokenize(chat_tmp)
            batch_tokens.append(chat_tmp_tokens)
        # 返回一个字典,其中"content"键对应预处理后的对话列表
        return {"content": batch, "content_tokens":batch_tokens}

    raw_datasets = DatasetDict()   # 初始化一个空的DatasetDict对象,用于存储数据集
    # 遍历data_args.splits指定的数据集切分(如train、test等)
    for split in data_args.splits.split(","):
        try:
            # Try first if dataset on a Hub repo, 首先尝试从Hugging Face Hub上加载指定的数据集
            dataset = load_dataset(data_args.dataset_name, split=split)
        except DatasetGenerationError:
            # If not, check local dataset, 如果从Hub上加载失败,则尝试从本地磁盘加载数据集
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

        # 根据切分类型,将数据集存入raw_datasets的对应键值中
        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(f"Split type {split} not recognized as one of test or train.")

    # 如果apply_chat_template为True,则对数据集应用preprocess函数进行预处理
    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,         # 表示对样本进行批处理,提高效率
            remove_columns=raw_datasets["train"].column_names,
            # TODO 新增代码, 取消缓存, 用于调试
            load_from_cache_file = False
        )

    train_data = raw_datasets["train"]  # 获取训练数据集
    valid_data = raw_datasets["test"]   # 获取测试数据集

    # TODO 只有主进程打印
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")  # 打印数据集大小
        print(f"A sample of train dataset: {train_data[0]}")  # 打印训练数据集的第一个样本

    return train_data, valid_data


# create_and_prepare_model函数用于创建和准备模型
# 参数包括:
#   args: 包含模型相关配置的参数对象,如模型名称、是否使用量化等
#   data_args: 包含数据相关配置的参数对象,如最大序列长度等
#   training_args: 包含训练相关配置的参数对象,如是否使用梯度检查点等
def create_and_prepare_model(args, data_args, training_args):
    if args.use_unsloth:
        # 如果使用Unsloth库(一种用于加速语言模型的库),则导入FastLanguageModel类
        from unsloth import FastLanguageModel
    bnb_config = None    # 初始化BitsAndBytesConfig为None,用于量化配置
    quant_storage_dtype = None   # 初始化量化存储数据类型为None

    # 检查是否为分布式训练且使用Unsloth库,如果是则抛出NotImplementedError
    # 因为当前版本的Unsloth不支持分布式训练
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    # 如果使用4位量化,则设置计算数据类型和量化存储数据类型
    if args.use_4bit_quantization:
        # 获取指定的计算数据类型, getattr 会将字符串 bfloat16 ---> torch.bfloat16
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        # 获取指定的量化存储数据类型, getattr 会将字符串 bfloat16 ---> torch.bfloat16
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        # 创建BitsAndBytesConfig对象,用于配置量化相关参数
        # BitsAndBytesConfig是一个用于管理量化配置的类,可以指定量化类型、计算数据类型、存储数据类型等
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,          # 是否使用4位量化
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,     # 4位量化的类型, 如 nf4
            bnb_4bit_compute_dtype=compute_dtype,             # 计算数据类型
            bnb_4bit_use_double_quant=args.use_nested_quant,  # 是否使用双量化
            # TODO Qlora + zero3 修改的代码
            bnb_4bit_quant_storage=quant_storage_dtype,       # 量化存储数据类型
        )

        # 如果计算数据类型为float16且使用4位量化,则打印GPU是否支持bfloat16的提示
        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        # 如果使用8位量化,则创建相应的BitsAndBytesConfig对象
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    # 如果使用Unsloth库
    if args.use_unsloth:
        # Load model, 使用FastLanguageModel.from_pretrained方法加载模型, 传入模型名称路径、最大序列长度、是否使用4位量化等参数
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else: # 如果不使用Unsloth库,则使用AutoModelForCausalLM.from_pretrained方法加载模型
        # TODO Qlora + zero3 修改的代码
        # 如果指定了quant_storage_dtype且是浮点数类型,则使用quant_storage_dtype, 否则使用默认的torch.float32
        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )
        # 使用AutoModelForCausalLM.from_pretrained方法加载语言模型, 传入模型路径、量化配置、是否信任远程代码、注意力实现方式和数据类型等参数
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            # 注意力实现方式,flash_attention_2或eager
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            # TODO Qlora + zero3 修改的代码
                # 注意 torch_dtype 对于 AutoModelForCausalLM 与 bnb_4bit_quant_storage 数据类型相同。就是这样。其他所有事情都由 Trainer 和 TRL 处理。
            torch_dtype=torch_dtype,
        )

    peft_config = None      # 初始化PEFT配置为None
    chat_template = None    # 初始化聊天模板为None
    # 如果使用PEFT LoRA且不使用Unsloth库,则创建LoraConfig对象
    # PEFT (Parameter-Efficient Fine-Tuning)是一种模型微调技术,可以在保持大部分模型参数不变的情况下,只微调一小部分参数
    # LoRA (Low-Rank Adaptation)是PEFT的一种实现,通过添加低秩矩阵来适应新任务
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,         # LoRA的alpha参数,控制LoRA层的重要性
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",                       # 是否对偏置项应用LoRA
            task_type="CAUSAL_LM",             # 任务类型,这里是因果语言模型
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    special_tokens = None   # 初始化特殊标记为None
    chat_template = None    # 初始化聊天模板为None
    # 根据args.chat_template_format参数,设置特殊标记和聊天模板
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens              # 使用Chatml格式的特殊标记
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE      # 使用Chatml聊天模板
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens            # 使用Zephyr格式的特殊标记
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE    # 使用Zephyr聊天模板

    # 如果特殊标记不为None
    if special_tokens is not None:
        # 使用AutoTokenizer.from_pretrained方法加载分词器
        # 设置填充标记、句首标记、句尾标记和其他特殊标记
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,     # 填充标记
            bos_token=special_tokens.bos_token.value,     # 句首标记
            eos_token=special_tokens.eos_token.value,     # 句尾标记
            additional_special_tokens=special_tokens.list(),  # 其他特殊标记
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template           # 设置聊天模板
        # make embedding resizing configurable?
        # 调整tokenizer的嵌入大小,使其能够容纳新增的特殊标记
        # pad_to_multiple_of=8用于对齐,提高GPU计算效率
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        # 如果特殊标记为None,则直接加载分词器
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token     # 设置填充标记为句尾标记


    # 如果使用Unsloth库
    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        # 使用FastLanguageModel.get_peft_model方法对模型进行修补,并添加快速LoRA权重
        # 传入LoRA相关参数,如alpha、dropout、rank等,以及是否使用梯度检查点、随机种子和最大序列长度
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    return model, peft_config, tokenizer       # 返回模型、PEFT配置和分词器
