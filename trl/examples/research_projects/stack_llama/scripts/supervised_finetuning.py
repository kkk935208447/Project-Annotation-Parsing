import argparse
import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


"""
Fine-Tune Llama-7b on SE paired dataset
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="lvwerra/stack-exchange-paired")
    parser.add_argument("--subset", type=str, default="data/finetune")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    return parser.parse_args()

# 这段代码定义了几个与自然语言处理相关的函数,用于估计字符与token之间的比率、计算可训练参数数量、准备样本文本等。主要用途是在微调大型语言模型时对数据进行预处理和评估
def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0  # 3. 初始化字符数和token数的计数器
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):  # 4. 遍历指定数量的样本
        text = prepare_sample_text(example)   # 5. 准备样本文本
        total_characters += len(text)         # 6. 累加样本文本的字符数
        if tokenizer.is_fast:                # 7. 如果是快速tokenizer
            total_tokens += len(tokenizer(text).tokens())       # 8. 使用快速tokenizer统计token数
        else:
            total_tokens += len(tokenizer.tokenize(text))         # 9. 否则使用普通tokenize函数统计token数

    return total_characters / total_tokens


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


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,          # 26. 要加载的数据集分割(train/test等)
        use_auth_token=True,       # 27. 是否使用认证令牌
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,  # 29. 是否以流式方式加载数据
    )
    if args.streaming:      # 30. 如果以流式方式加载数据
        print("Loading the dataset in streaming mode")      # 31. 打印提示信息
        valid_data = dataset.take(args.size_valid_set)      # 32. 从数据流中取出指定数量的样本作为验证集
        train_data = dataset.skip(args.size_valid_set)      # 33. 跳过验证集,剩余部分作为训练集
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)   # 34. 对训练集进行随机打乱
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=args.seed)    # 36. 从数据集中划分出训练集和测试集
        train_data = dataset["train"]       # 37. 获取训练集
        valid_data = dataset["test"]        # 38. 获取测试集(作为验证集)
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)    # 40. 估计训练集中每个token包含的平均字符数
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,      # 44. 用于准备样本文本的函数
        infinite=True,                     # 45. 是否以无限模式加载数据(循环加载)
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,          # 47. 字符与token的比率
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,                  # 48. 验证集以普通模式加载
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0          # 56. 设置训练数据集的迭代起始点为0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,       # 58. 模型输出目录
        dataloader_drop_last=True,         # 59. 是否丢弃最后一个不完整的批次
        evaluation_strategy="steps",      # 60. 评估策略,这里是每隔一定步数评估一次
        max_steps=args.max_steps,         # 61. 最大训练步数
        eval_steps=args.eval_freq,        # 62. 评估频率(步数)
        save_steps=args.save_freq,        # 63. 保存模型的频率(步数)
        logging_steps=args.log_freq,         # 64. 记录日志的频率(步数)
        per_device_train_batch_size=args.batch_size,     # 65. 每个设备的训练批次大小
        per_device_eval_batch_size=args.batch_size,      # 66. 每个设备的评估批次大小
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,               # 69. warmup步数
        gradient_accumulation_steps=args.gradient_accumulation_steps,      # 70. 梯度累积步数
        gradient_checkpointing=args.gradient_checkpointing,       # 71. 是否使用梯度检查点
        fp16=args.fp16,         # 72. 是否使用FP16精度
        bf16=args.bf16,         # 73. 是否使用BF16精度
        weight_decay=args.weight_decay,    # 74. 权重衰减
        run_name="llama-7b-finetuned",      # 75. 运行名称
        report_to="wandb",
        ddp_find_unused_parameters=False,    # 77. 是否查找未使用的参数
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, load_in_8bit=True, device_map={"": Accelerator().process_index}
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=True,  # 84. 是否对模型进行打包
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the llama model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()      # 97. 设置日志级别为错误级别

    main(args)
