#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy,math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
# TODO 新增代码
import os 
os.environ["WANDB_DISABLED"] = "true" # 关闭 wandb

# TODO 新增代码, logger
def set_logger_with_seed(training_args: transformers.Seq2SeqTrainingArguments):
    import logging,sys
    from transformers import set_seed
    logger = logging.getLogger(__name__)  # 7. 创建一个日志记录器,用于记录脚本运行时的日志信息。
    # Setup logging
    # 10. 设置日志记录的基本配置,包括日志格式、日期格式和输出处理器(在这里是将日志输出到标准输出流)。
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        # 11. 如果训练参数指定应该记录日志,则将Transformers库的日志级别设置为info(信息级别)。默认情况下,训练参数的日志级别是被动的,所以这里将其设置为信息级别。
        transformers.utils.logging.set_verbosity_info()
 
    # 12. 获取训练参数指定的进程日志级别,并将该级别设置为当前日志记录器的级别。
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
     # 13. 设置Transformers库的日志级别为训练参数指定的进程日志级别,启用默认的日志处理器和显式的日志格式。
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # 设置一切相关的随机种子,保证训练结果的可重现性
    set_seed(training_args.seed)
    return logger

# TODO 新增代码, 打印模型的是否参与训练的参数名和数据类型
def print_model_allarguments_name_dtype(logger,model):
    for n,v in model.named_parameters():
        if v.requires_grad:
            logger.info(f"trainable model arguments: {n} - {v.dtype} - {v.shape}")
        else:
            logger.info(f"not trainable model arguments: {n} - {v.dtype} - {v.shape}")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict() # 从 Trainer 中获取当前模型的状态字典
    if trainer.args.should_save: # 如果训练器的参数 should_save 为 True,表示需要保存模型
        # 将状态字典中的所有参数张量转移到 CPU 设备上,以减少内存占用
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

# smart_tokenizer_and_embedding_resize 函数用于调整 tokenizer 和模型 embedding 的大小
# 以适应新增的特殊 token,确保模型能够正确处理这些特殊 token
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict) # 将特殊 token 添加到 tokenizer 中,获取添加的新 token 数量
    model.resize_token_embeddings(len(tokenizer))  # 调整模型 embedding 层的大小,使其与 tokenizer 的词汇表大小一致
 
    if num_new_tokens > 0:
        # 获取模型的输入 embedding 和输出 embedding
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        # 计算输入 embedding 和输出 embedding 的平均值,作为新 token 的初始化值
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        # 将新增 token 的 embedding 初始化为之前 token 的平均值
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    
    # TODO 新增代码, 将resize_token_embeddings设置为8的倍数, 为了提升性能
    T,E = input_embeddings = model.get_input_embeddings().weight.shape
    model.resize_token_embeddings(int(8 * math.ceil(T / 8.0)))

# _tokenize_fn 函数用于对一个字符串列表进行分词
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings   # 一次一个样本
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

# preprocess 函数用于对输入和目标数据进行预处理和tokenization
# 它将输入和目标拼接成完整的序列,并使用 _tokenize_fn 进行分词
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]  # 将输入和目标拼接成完整的序列
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]  # 使用 _tokenize_fn 对输入序列和源序列进行分词
    input_ids = examples_tokenized["input_ids"]  # 从分词结果中提取 input_ids
    labels = copy.deepcopy(input_ids)
    # 创建 labels 序列,将输入部分设置为 IGNORE_INDEX,只需预测输出部分
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)   # 返回包含输入 ID 和标签的字典

# SupervisedDataset 类继承自 PyTorch 的 Dataset 类,用于加载和管理监督式学习任务的数据
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        # 加载指定路径下的数据
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)
        # 格式化输入文本
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        # 对输入和目标进行tokenization
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        # 将处理后的 input_ids 和 labels 保存为实例属性
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# DataCollatorForSupervisedDataset 类用于对监督式学习任务的数据进行批处理
# 它将一个样本序列转换为一个批次的输入和标签张量
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 从样本字典中提取 input_ids 和 labels
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # 使用 PyTorch 的 rnn.pad_sequence 函数对 input_ids 和 labels 进行填充
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

# make_supervised_data_module 函数用于创建监督式学习任务的数据模块
# 它返回一个字典,包含训练集、验证集和数据批处理器
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    # 使用 HuggingFace 的参数解析器,解析模型参数、数据参数和训练参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # TODO 新增代码
    logger = set_logger_with_seed(training_args=training_args)
    torch.distributed.barrier() # 进程阻塞,同步进程

    # TODO 新增代码,训练阶段, 将 use_cache=False 
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.use_cache=False

    # 从指定的预训练模型路径加载模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # TODO 新增代码, 加载时使用 bf16, 节省内存
        torch_dtype=torch.bfloat16,
        # TODO 新增代码, 添加修改后的 config
        config = config
    )
    
    # 从指定的预训练模型路径加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",     # 右填充
        use_fast=False,
    )
    # 如果 tokenizer 没有 pad_token,则添加 pad_token 并调整模型 embedding 大小
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    # 如果模型是 LLaMA 模型,则添加 eos_token、bos_token 和 unk_token
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    # TODO 新增代码, model 打印参数
    if training_args.local_rank == 0: 
        logger.info(f"Model {model}")
        logger.info(f"Training/evaluation parameters {training_args}")
        logger.info(f"Model parameters {model.config}")
        print_model_allarguments_name_dtype(logger=logger,model=model)
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # TODO 新增代码, 同步所有进程,等待所有进程到达这一点
    torch.distributed.barrier()
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
