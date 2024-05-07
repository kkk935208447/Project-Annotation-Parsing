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

import copy
import logging
# 1. dataclass是Python 3.7中引入的一种语法糖,用于自动为数据类添加特殊方法,如__init__、__repr__等。
#    它可以减少样板代码的数量,提高代码的可读性和维护性。
# 2. field是dataclass中用于定义字段的描述符。
#    它允许为字段设置默认值、元数据等。
#    例如,在代码中的DataArguments类中,data_path字段使用field描述符设置了默认值None,并添加了元数据{"help": "Path to the training data."}。
from dataclasses import dataclass, field
# 从typing模块导入Dict、Optional和Sequence类型提示,用于类型注解。
# 1. Dict表示一个字典类型,可以指定键和值的类型,如Dict[str, int]表示键为字符串,值为整数的字典。
# 2. Optional表示一个可选类型,可以是该类型或None。例如,Optional[str]表示类型为字符串或None。
# 3. Sequence表示一个序列类型,如列表或元组。它是typing.Reversible和typing.Container的子类。
# 使用类型注解可以提高代码的可读性和可维护性,并在开发过程中捕获潜在的类型错误。
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer

# 定义一个常量IGNORE_INDEX,其值为-100。
# 在计算损失函数时,这个值表示需要忽略对应位置的标签。例如,在进行SFT任务时,输入序列部分的标签通常会被标记为IGNORE_INDEX,以避免对输入序列的token进行惩罚。
# 而定义为 -100 是为了契合transformers, transformers库默认 -100 为忽略loss计算的标签
IGNORE_INDEX = -100
# 定义了一些默认的特殊标记(token),如填充标记(PAD)、结束标记(EOS)、开始标记(BOS)和未知标记(UNK)。
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
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


# 1. 这些是使用dataclass定义的数据类,用于存储模型参数、数据参数和训练参数。
# 2. ModelArguments类存储了预训练模型的路径或名称,默认为"facebook/opt-125m"。
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
# 3. DataArguments类存储了训练数据的路径,是一个必需的字段。
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
# 4. TrainingArguments类继承自transformers.TrainingArguments,用于存储训练相关的参数。
#    它包括了缓存目录、优化器、最大序列长度等字段。
# 5. 使用dataclass可以方便地定义这些参数类,并提供默认值和元数据。
# 6. 在训练过程中,这些参数将被解析并用于配置模型、数据和训练过程。
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # cache_dir是一个可选字符串字段,用于指定缓存目录
    cache_dir: Optional[str] = field(default=None)
    # optim是一个字符串字段,用于指定优化器,默认为"adamw_torch"
    optim: str = field(default="adamw_torch")
    # model_max_length是一个整数字段,用于指定模型的最大序列长度
    # 超过该长度的序列将被截断,不足的部分将被右侧填充
    model_max_length: int = field(default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict,
                                         tokenizer: transformers.PreTrainedTokenizer,
                                         model: transformers.PreTrainedModel,
):
    # 1. 这是一个函数,用于调整tokenizer和embedding的大小。
    # 2. 它接受三个参数:special_tokens_dict(特殊标记字典)、tokenizer(tokenizer实例)和model(模型实例)。
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # 3. 首先,它使用tokenizer.add_special_tokens方法向tokenizer添加特殊标记,并获取新增标记的数量。
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # 4. 然后,它调用model.resize_token_embeddings方法,根据新的词表大小调整embedding的大小。
    model.resize_token_embeddings(len(tokenizer))

    # 5. 如果新增了特殊标记,函数会计算原有embedding的平均值,并将新增的embedding初始化为该平均值。
    # 6. 这个函数的作用是确保tokenizer和模型之间的一致性,并正确处理新添加的特殊标记。
    # 7. 注释中提到,这是一个未经优化的版本,可能会导致embedding的大小不被64整除,这可能会影响性能。
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    # 2. 它接受两个参数:strings(字符串序列)和tokenizer(tokenizer实例)。
    # 3. 函数使用tokenizer的__call__方法对每个字符串进行tokenize,并将结果存储在tokenized_list中。
    """Tokenize a list of strings."""
    # 对每个字符串进行tokenize,并将结果存储在tokenized_list中, padding="longest"表示将序列填充到最长长度
    # max_length指定最大序列长度,超过该长度的部分将被截断, truncation=True表示允许截断
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    # 从tokenize结果中提取input_ids和labels
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # 计算每个序列的实际长度,不包括填充标记
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

# 定义了一个函数,用于对输入和目标序列进行预处理
def preprocess(
    sources: Sequence[str],   # 用户模版输入, instruct + input
    targets: Sequence[str],   # 这里面的传入的 targets 已经包含了 EOS
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    # 将输入序列和目标序列拼接成一个序列
    examples = [s + t for s, t in zip(sources, targets)]
    # 对拼接后的序列和输入序列进行tokenize
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    # 将input_ids作为初始labels
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        # 将labels中对应模版输入部分即 instruct + input 的标签设置为IGNORE_INDEX,表示在训练时忽略这些标签
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

# 6. 定义了一个Dataset类,用于加载和处理监督式微调的数据集
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # 从指定路径加载数据,得到一个字典列表
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        # 根据数据中是否包含输入(input),生成对应的提示(prompt)序列
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        # 生成目标序列,在输出(output)后添加eos_token
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        # 对输入和目标序列进行预处理(tokenize)
        data_dict = preprocess(sources, targets, tokenizer)
        # 将处理后的input_ids和labels存储在Dataset实例中
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        # 返回Dataset实例的长度
        return len(self.input_ids)   

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 根据索引返回对应的input_ids和labels
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    # 1. 这是一个数据collator类,用于在每个训练批次中组合样本。
    # 2. 它继承自object,并使用dataclass进行定义。
    """Collate examples for supervised fine-tuning."""

    # 3. 构造函数接受一个tokenizer实例作为参数。
    tokenizer: transformers.PreTrainedTokenizer

    # 4. __call__方法实现了样本组合的逻辑, 它接受一个实例序列(instances),并从中提取input_ids和labels。
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 从输入的样本实例中提取input_ids和labels
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # 对input_ids进行填充,使它们在批次内具有相同的长度, 填充的id是tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # labels填充为IGNORE_INDEX, 我们使用transformers Trainer进行训练, 其内部会自动忽略掉 IGNORE_INDEX的loss的计算
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        # 返回一个字典,包含填充后的input_ids、labels和attention_mask
        return dict(
            input_ids=input_ids,
            labels=labels,
            # .ne(self.tokenizer.pad_token_id)：这是一个逐元素比较的操作，意味着“不等于（not equal）”。self.tokenizer.pad_token_id 是tokenizer中定义的填充（padding）token的ID。这个操作会对 input_ids 中的每个元素进行检查，看它是否不等于 pad_token_id。
            # 结果是一个与 input_ids 形状相同的Boolean（True/False）Tensor，
            # 其中 True（或1）表示该位置的token不是填充token，False（或0）表示该位置的token是填充token。
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

# 8. 定义了一个函数,用于创建监督式微调所需的数据模块
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # 1. 创建一个SupervisedDataset实例作为训练数据集,传入tokenizer和数据路径。
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    # 2. 创建一个DataCollatorForSupervisedDataset实例作为数据collator,传入tokenizer。
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # 3. 返回一个字典,包含训练数据集、评估数据集(这里为None)和数据collator。
    # 这个字典将被用于初始化Trainer。
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

# 9. 定义了一个train函数,用于进行监督式微调
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # 1. 使用transformers.HfArgumentParser解析命令行参数,并将它们转换为相应的数据类实例。
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. 根据模型参数,从预训练模型中加载causaL LM模型。
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # 3. 根据模型参数,加载对应的tokenizer。
    # 设置了一些tokenizer参数,如最大长度、填充方向
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right"
    )
    # 4. 检查tokenizer中是否缺少一些特殊标记,如果缺少则添加默认值。
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # 5. 调用smart_tokenizer_and_embedding_resize函数,根据新添加的特殊标记调整tokenizer和embedding的大小。
    # 并为新加入的token初始化embedding为旧embedding的平均值。
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    # 6. 创建监督式微调所需的数据模块,包括训练数据集、评估数据集(为None)和数据collator。
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # 7. 创建Trainer实例,传入模型、tokenizer、训练参数和数据模块。
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    # trainer.train()执行模型的训练过程。在训练过程中,Trainer会使用传入的数据集、collator和训练参数进行模型训练。
    # trainer.save_state()保存模型的当前状态,包括优化器状态、训练步数等。
    # trainer.save_model(output_dir=training_args.output_dir)将最终训练好的模型保存到指定的输出目录中。
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    """
    1. trainer.save_state()保存的是模型的当前状态,包括:
    模型参数(model.state_dict())
    优化器状态(optimizer.state_dict())
    训练状态,如当前训练步数(self.state.global_step)
    保存这些状态的目的是为了允许从上次中断的地方恢复训练。如果训练过程中断(比如由于硬件故障或者手动停止),可以使用trainer.train(resume_from_checkpoint=...)从保存的状态继续训练。

    2. 而trainer.save_model(output_dir=training_args.output_dir)保存的是最终训练好的模型权重。它调用了model.save_pretrained(output_dir)方法,将模型参数和配置保存到指定的输出目录中。
    保存最终模型权重的目的是为了将来可以加载并使用这个训练好的模型进行推理或其他任务,而不需要再次训练。

    3. 因此,save_state()和save_model()分别代表了不同的保存目标:
        save_state()保存当前训练状态,用于恢复训练
        save_model()保存最终模型权重,用于部署模型
        通常,我们需要同时调用这两个方法,以确保训练状态和最终模型权重都能够被正确保存。
    """


if __name__ == "__main__":
    train()
