# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from trl import SFTTrainer
from trl.import_utils import is_npu_available, is_xpu_available
from trl.trainer import ConstantLengthDataset

# field 装饰器用于为类属性指定元数据,如默认值、类型等
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    # streaming 参数指定是否以流式方式加载数据集,默认为 True,流式加载可以减少内存占用,适用于大型数据集
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    # shuffle_buffer 参数指定了 shuffle 缓冲区的大小,默认为 5000, 这个参数仅在 streaming=True 时有效,用于控制数据集的随机打乱程度
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    # num_workers 参数指定了用于数据预处理的工作线程数量,默认为 4,增加工作线程数量可以加快数据预处理的速度,但也会消耗更多资源
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    use_bnb: Optional[bool] = field(default=True, metadata={"help": "whether to use BitsAndBytes"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


# group_by_length 选项用于将数据按长度分组,以提高训练效率, 当数据按长度分组时,每个批次中的序列长度都相近,可以减少由于填充导致的计算浪费
# packing 选项用于对小批量数据进行打包,以避免内存浪费, 当批量数据较小时,将多个小批量打包在一起,可以更有效地利用GPU/TPU资源,减少内存开销
# 但是这两个选项是互斥的,因为 group_by_length 已经将相似长度的数据分组,再使用 packing 会破坏这种分组
if training_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")

# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
if training_args.gradient_checkpointing:
    raise ValueError("gradient_checkpointing not supported")

set_seed(training_args.seed)


# 定义 chars_token_ratio 函数,用于估计数据集中每个 token 平均包含的字符数, 这个信息对于后续的数据处理很有用
def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
     # 遍历数据集的前 nb_examples 个样本
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)  # 调用 prepare_sample_text 函数准备样本文本
        total_characters += len(text)         # 统计文本的字符数
        if tokenizer.is_fast:                 # 根据 tokenizer 的类型,统计文本的 token 数
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    # 返回字符数和 token 数的比值,作为字符-token 比率的估计值
    return total_characters / total_tokens

# 定义 print_trainable_parameters 函数,用于打印模型的可训练参数数量, 这个信息对于监控训练过程很有用
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def create_datasets(tokenizer, args, seed=None):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,    # split 参数指定要加载的数据集分割
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,   # streaming 参数指定是否以流式方式加载数据集
    )
    # 如果以流式方式加载数据集
    if args.streaming:
        print("Loading the dataset in streaming mode")
        # 使用 take 方法获取验证集
        valid_data = dataset.take(args.size_valid_set) 
        train_data = dataset.skip(args.size_valid_set)  # 使用 skip 方法获取训练集
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=seed)  # 对训练集进行 shuffle,buffer_size 参数控制 shuffle 的程度
    else:
        # 如果不是流式加载,则使用 train_test_split 方法划分训练集和验证集
        dataset = dataset.train_test_split(test_size=0.005, seed=seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
     # 计算数据集的字符-token 比率
    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    # 创建 ConstantLengthDataset 实例作为训练集
    # formatting_func 参数指定文本预处理函数
    # infinite 参数指定是否无限循环数据集
    # seq_length 参数指定输入序列的最大长度
    # chars_per_token 参数指定字符-token 比率
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


bnb_config = None
if script_args.use_bnb:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# 修复使用fp16训练时的奇怪溢出问题
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

train_dataset, eval_dataset = create_datasets(tokenizer, script_args, seed=training_args.seed)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=script_args.packing,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
trainer.save_model(training_args.output_dir)

output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
if is_xpu_available():
    torch.xpu.empty_cache()
elif is_npu_available():
    torch.npu.empty_cache()
else:
    torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
