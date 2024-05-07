# -*- coding: utf-8 -*-
import os
import jieba
import dataclasses as dc
import functools            # functools模块提供了一些高阶函数和装饰器,可以用于简化代码,提高函数的复用性和灵活性。
from collections.abc import Callable, Mapping, Sequence  # collections.abc 模块定义了一些抽象基类,如Callable、Mapping和Sequence,用于描述集合类型的接口。这些接口可以帮助我们编写更加健壮和通用的代码。
from pathlib import Path     # pathlib 模块提供了一个面向对象的路径操作API,更加方便和安全地处理文件系统路径。
from typing import Annotated, Any, Optional, Union
import numpy as np
import ruamel.yaml as yaml     # ruamel.yaml 是一个 YAML 解析库,相比标准的 yaml 模块,它提供了更好的保留注释和格式的功能,适合用于读写配置文件。
import torch
import torch.distributed
import typer  # typer 是一个基于 Click 库的命令行应用程序开发框架,可以帮助我们快速构建功能丰富的命令行工具。
from datasets import (
    Dataset, 
    DatasetDict, 
    NamedSplit, 
    Split, 
    load_dataset
)
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import (
    PeftConfig,
    PeftModelForCausalLM,
    get_peft_config,
    get_peft_model
)
from rouge_chinese import Rouge
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments, AutoConfig,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq

from transformers import Seq2SeqTrainer as _Seq2SeqTrainer

# 定义一些类型别名,方便后续代码中的类型注解
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# 使用typer创建命令行应用程序,Typer是一个用于构建命令行应用程序的库,在本代码中可能会用于创建一个命令行工具,方便用户使用。
# pretty_exceptions_show_locals=False表示不显示本地变量
app = typer.Typer(pretty_exceptions_show_locals=False)

# TODO 新增代码
def set_logger_with_seed(training_args: Seq2SeqTrainingArguments):
    import logging,sys
    import transformers
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

# TODO 新增代码
def print_model_allarguments_name_dtype(model):
    print('--> Model arguments')
    for n,v in model.named_parameters():
        if v.requires_grad:
            print(f"trainable model arguments: {n} - {v.dtype} - {v.shape}")
        else:
            print(f"not trainable model arguments: {n} - {v.dtype} - {v.shape}")


# DataCollatorForSeq2Seq 是一个用于处理序列到序列任务输入数据的数据收集器
# 它继承自 _DataCollatorForSeq2Seq 类,并重写了 __call__ 方法, ChatGlM input_ids labels 都是左填充, 同时加入 attention_mask(左填充) 和 position_ids ( 0,0,0,0,0,0,1,2,3,4....)
class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = (  # 从 features 中提取 output_ids,如果不存在则设置为 None
            [feature['output_ids'] for feature in features]
            if 'output_ids' in features[0].keys()
            else None
        )
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)  # 计算 output_ids 的最大长度
            # 如果指定了 pad_to_multiple_of,则将最大长度向上取整到该值的整数倍
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
             # 遍历 features,为每个 output_ids 填充至最大长度, 右填充, 这是为了 eval/ predict
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        # 调用父类 _DataCollatorForSeq2Seq 的 __call__ 方法,返回处理后的特征
        return super().__call__(features, return_tensors)

# Seq2SeqTrainer 是一个用于序列到序列任务训练的 Trainer 类
# 它继承自 _Seq2SeqTrainer,并重写了 prediction_step 方法
class Seq2SeqTrainer(_Seq2SeqTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # 如果 predict_with_generate 为 True,则从 inputs 中提取 output_ids
        if self.args.predict_with_generate:
            output_ids = inputs.pop('output_ids')
        input_ids = inputs['input_ids']
        # 调用父类 _Seq2SeqTrainer 的 prediction_step 方法
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        # 从生成的 tokens 中去除输入 token 部分,仅保留生成的部分
        generated_tokens = generated_tokens[:, input_ids.size()[1]:]
        # 如果 predict_with_generate 为 True,则使用 output_ids 作为 labels
        if self.args.predict_with_generate:
            labels = output_ids
        return loss, generated_tokens, labels
    # For P-Tuning a new save_model function is fine for the prefix_encoder model
    # but may cost problems for the whole model loading

    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     if output_dir is None:
    #         output_dir = self.args.output_dir
    #     os.makedirs(output_dir, exist_ok=True)
    #     ptuning_params = {k: v for k, v in self.model.transformer.prefix_encoder.state_dict().items()}
    #
    #     torch.save(ptuning_params, os.path.join(output_dir, 'pytorch_model.bin'))
    #
    #     print(f"P-Tuning model weights saved in {output_dir}")
    #
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)


def _resolve_path(path: Union[str, Path]) -> Path:
    # _resolve_path 函数用于将输入路径规范化为一个 Path 对象, 它会将路径展开为绝对路径,并解析任何符号链接
    return Path(path).expanduser().resolve()

# _sanity_check 函数用于对输入和输出 token ID 进行简单的检查和打印
# 它会遍历输入和输出 token ID 序列,并根据 tokenizer 将 token ID 转换为对应的文本
def _sanity_check(
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        tokenizer: PreTrainedTokenizer,
):
    print('--> Sanity check')
    for in_id, out_id in zip(input_ids, output_ids):
        if in_id == 0:
            continue # 跳过输入 ID 为 0 的情况,因为 0 通常表示填充 token
        if in_id in tokenizer.tokenizer.index_special_tokens:
            # 如果输入 ID 对应的是特殊标记,则直接获取其文本表示
            in_text = tokenizer.tokenizer.index_special_tokens[in_id]
        else:
            # 否则,使用 tokenizer 的 decode 方法将 ID 转换为文本
            in_text = tokenizer.decode([in_id])
        print(f'{repr(in_text):>20}: {in_id} -> {out_id}')  # 打印输入 token、输入 ID 和对应的输出 ID

# _get_yaml_parser 函数使用缓存装饰器 @functools.cache 缓存 YAML 解析器的实例
# 这样可以避免每次调用时都重新创建 YAML 解析器,提高性能
@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    # 创建一个 YAML 解析器实例,设置一些格式化参数
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser

# DataConfig 类用于存储训练、验证和测试数据文件的路径, dict[NamedSplit, str] 类型
@dc.dataclass
class DataConfig(object):
    train_file: str                     # 训练数据文件路径
    val_file: Optional[str] = None      # 验证数据文件路径(可选)
    test_file: Optional[str] = None
    # 处理数据时使用的进程数(可选)
    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:  # 数据格式属性,返回训练数据文件的后缀作为数据格式
        return Path(self.train_file).suffix

    @property  
    def data_files(self) -> dict[NamedSplit, str]:  # 数据文件字典属性,返回包含训练、验证和测试数据文件路径的字典
        # 使用 zip 函数将三种数据类型(训练、验证和测试)与对应的文件路径关联起来
        # 并构建一个字典,key 为数据类型(通过 NamedSplit 枚举定义),value 为文件路径
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],  
                [self.train_file, self.val_file, self.test_file], 
            )
            if data_file is not None    # 只包含不为 None 的文件路径
        }

# FinetuningConfig 类用于封装微调模型的配置信息
# 它包含了数据配置、输入输出长度限制、训练参数和 PEFT 配置等
@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig     # 数据配置

    max_input_length: int        # 最大输入长度
    max_output_length: int       # 最大输出长度

    # 如果在创建 FinetuningConfig 实例时未提供 training_args 参数,则会自动调用这个 lambda 函数来生成一个默认的 Seq2SeqTrainingArguments 实例,并将其赋值给 training_args 属性。
    # 这个属性用于存储训练模型时使用的参数配置信息
    # 它使用 Seq2SeqTrainingArguments 类型,该类是 Hugging Face Transformers 库提供的一个用于序列到序列任务训练的参数配置类
    # 在实例化 FinetuningConfig 时,如果未提供 training_args 参数,则会使用默认工厂函数创建一个新的 Seq2SeqTrainingArguments 实例
    # 默认情况下,将 output_dir 设置为 './output'，表示训练过程中产生的模型、日志等文件将被保存在当前目录下的 output 文件夹中
    # Seq2SeqTrainingArguments 类包含了许多常用的训练参数,如:
    #   - do_train: 是否执行训练
    #   - do_eval: 是否执行评估
    #   - evaluation_strategy: 评估策略,如每个 epoch 评估一次
    #   - per_device_train_batch_size: 训练时每个设备的批量大小
    #   - per_device_eval_batch_size: 评估时每个设备的批量大小
    #   - num_train_epochs: 训练的 epoch 数
    #   - learning_rate: 训练时使用的学习率
    #   - weight_decay: 权重衰减系数
    #   - lr_scheduler_type: 学习率调度器类型
    # 通过配置这些参数,可以很方便地控制模型的训练过程,提高训练效果
    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    # PEFT 配置(可选)
    peft_config: Optional[PeftConfig] = None

    # 在实例化后自动执行的初始化方法
    def __post_init__(self):
        # 如果未提供验证数据集或评估功能未开启,则跳过验证阶段
        if not self.training_args.do_eval or self.data_config.val_file is None:
            # skips the evaluation stage when `do_eval` or `eval_file` is not provided
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
            self.data_config.val_file = None
        else:  # 否则,设置验证批量大小为训练批量大小或其默认值
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':  # 从字典创建 FinetuningConfig 实例的类方法
        # 处理 training_args 参数
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(training_args, Seq2SeqTrainingArguments):
            gen_config = training_args.get('generation_config')
            # TODO: a bit hacky
            if not isinstance(gen_config, GenerationConfig): # 如果 generation_config 不是 GenerationConfig 类型,将其转化为 GenerationConfig 类型
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            # 如果 training_args 不是 Seq2SeqTrainingArguments 类型,将其转化为 Seq2SeqTrainingArguments 类型
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)
        # 处理 data_config 参数
        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)
        # 处理 peft_config 参数
        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(peft_config)
        
        # FinetuningConfig 类中,from_dict 类方法就使用了 cls(**kwargs) 的方式来创建新的 FinetuningConfig 实例。开发者可以通过传入包含各种配置信息的字典,来快速地创建所需的 FinetuningConfig 对象
        return cls(**kwargs) 

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':  
        # 从yaml文件创建 FinetuningConfig 实例的类方法
        path = _resolve_path(path)
        kwargs = _get_yaml_parser().load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
        data_dir: Path,
        data_format: str,
        data_files: dict[NamedSplit, str],
        num_proc: Optional[int],
) -> DatasetDict:
    if data_format in ('.csv', '.json', '.jsonl'):
        dataset_dct = load_dataset(  # load_dataset 加载数据集
            data_format[1:],
            data_dir=data_dir,
            data_files=data_files,   # dict[NamedSplit, str] 格式的数据文件
            num_proc=num_proc,       # 可选的进程数
        )
    else:
        # 如果数据格式不支持,抛出 NotImplementedError 异常
        err_msg = f"Cannot load dataset in the '{data_format}' format."
        raise NotImplementedError(err_msg)

    return dataset_dct


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc # 从 DataConfig 中获取并保存进程数

        self._dataset_dct = _load_datasets(   # 调用 _load_datasets 函数加载数据集
            _resolve_path(data_dir),
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        # 根据指定的分割名称,从数据集字典中获取对应的数据集对象
        # 如果不存在,返回 None
        return self._dataset_dct.get(split, None)

    def get_dataset(
            self,
            split: NamedSplit,
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],
            batched: bool = True,
            remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)   # 获取指定分割的原始数据集
        if orig_dataset is None:
            return
        # TODO 调试新增代码: 使用少量的数据
        orig_dataset = orig_dataset.select(range(80))

        if remove_orig_columns:   # 是否在预处理之后删除原始数据集中的列
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,   # 自定义的数据处理的map函数
            batched=batched,      # 使用批处理的方式进行预处理
            remove_columns=remove_columns,  # 在预处理之后删除原始数据集中的列
            num_proc=self._num_proc,   # 启用的进程数
            load_from_cache_file = False # load_from_cache_file 表示不使用原本数据处理的缓存, 常用于数据变动, 也方便调试
        )

# TODO 修改此函数
def print_model_size(model: PreTrainedModel):
    print("--> Model")
    # 计算模型中可训练参数的总数, 使用 sum 函数遍历模型的所有参数,并累加可训练参数的数量, p.numel() 返回参数 p 的元素数量, p.requires_grad 用于判断参数是否需要梯度更新
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6}M trainable params\n")

# process_batch 函数用于将一个批次的输入数据转换为模型可以接受的格式。它从批次数据中提取对话信息,并使用预训练的 tokenizer 将其转换为 token 序列和标签序列。这个函数主要解决了如何将原始的对话数据转换为模型可以处理的格式的问题。
def process_batch(
        batch: Mapping[str, Sequence], # 这个函数用于处理一个batch的输入数据,将其转换为模型可以接受的格式
        tokenizer: PreTrainedTokenizer, # 一个预训练的tokenizer,用于将文本转换为token序列
        max_input_length: int,          # 输入序列的最大长度
        max_output_length: int,         # 输出序列的最大长度
) -> dict[str, list]:    # 请注意, 这些数据没有经过 pad
    # 从batch数据中获取 'tools' 和 'conversations' 两个键对应的值
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    # 初始化输入ID序列和标签列表
    batched_input_ids = []
    batched_labels = []

    if batched_tools is None:  # 如果 'tools' 键不存在,则用 None 填充
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv): # 遍历 'tools' 和 'conversations' 的每个元素
        # 初始化输入ID序列和loss mask列表
        input_ids, loss_masks = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ], [False, False]

        if tools is not None:   # 如果存在 'tools',则抛出NotImplementedError异常,表示尚未实现该功能
            raise NotImplementedError()

        for message in conv:  # 遍历对话中的每个消息
            # 根据消息的角色设置loss mask
            if message['role'] in ('system', 'user'):
                loss_mask_val = False     # labels 掩盖掉为false的部分, 这些token的标签将用 -100 的标签来代替
            else:
                loss_mask_val = True
            # 如果消息来自'tool',则抛出NotImplementedError异常
            if message['role'] == 'tool':
                raise NotImplementedError()
            else:
                # 使用tokenizer构建单个消息的token序列和loss mask
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content']
                )
                new_loss_masks = [loss_mask_val] * len(new_input_ids)
            # 将新的token序列和loss mask添加到input_ids和loss_masks中
            input_ids += new_input_ids
            loss_masks += new_loss_masks
        # 多轮的 role - content 合并在一起后再添加句子结束标记
        input_ids.append(tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        # 构建标签序列
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        # 截断输入序列和标签序列到最大长度(如果超过最大的长度, EOS会被截断掉, 表示样本未结束)
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    # 返回一个字典,包含输入ID序列和标签序列
    return {'input_ids': batched_input_ids, 'labels': batched_labels}

# process_batch_eval 函数与 process_batch 类似,但它是用于评估模型,不需要计算损失。它仅构建输入 ID 序列和输出 ID 序列,而不需要生成标签序列。
def process_batch_eval(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:  # 在eval阶段, 会把多轮的 user 与 assistant 的 分离开来, 也就是说一个样本可能会拆分成多个过程 
    # 从batch数据中获取 'tools' 和 'conversations' 两个键对应的值
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    # To avoid computing loss, we do not provide the `labels` field in the input dictionary.
    batched_output_ids = []
    # 如果 'tools' 键不存在,则用 None 填充
    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        # 初始化输入ID序列
        input_ids = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ]
        # 如果存在 'tools',则抛出NotImplementedError异常,表示尚未实现该功能
        if tools is not None:
            raise NotImplementedError()

        for message in conv:  # 遍历对话中的每个消息
            # 如果输入ID序列的长度已经达到最大输入长度,则停止添加
            if len(input_ids) >= max_input_length:
                break
            if message['role'] == 'tool':      # 如果消息来自'tool',则抛出NotImplementedError异常
                raise NotImplementedError()
            else:
                # 使用tokenizer构建单个消息的token序列
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content']
                )
                # 对于assistant生成的消息,将其分为prompt和output两部分
                if message['role'] == 'assistant':
                    output_prompt, output_ids = (
                        new_input_ids[:1],
                        new_input_ids[1:],
                    )
                    # 每一组 assistant的content 都会添加句子结束标记
                    output_ids.append(tokenizer.eos_token_id)
                    # 将prompt添加到输入ID序列,将output添加到输出ID序列
                    batched_input_ids.append(
                        input_ids[:max_input_length] + output_prompt[:1]  # :max_input_length 截取时 不会截取 output_prompt[:1] 这个特殊标识
                    )
                    batched_output_ids.append(output_ids[:max_output_length])  # :max_input_length 截取时 可能会截取掉 EOS 标识, 没有EOS标识表示句子长度未结束
                # 将新的token序列添加到input_ids中
                input_ids += new_input_ids
    # 返回一个字典,包含输入ID序列和输出ID序列
    return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}


# Not sure if this is necessary, can set it to half.
# If train with cpu, cast all params to fp32 instead of trainable ones.
# 用于准备模型进行训练。它将模型的所有参数或可训练参数转换为 float32 精度
def _prepare_model_for_training(model: nn.Module, use_cpu: bool):
    for param in model.parameters():
        # 如果 use_cpu 为 True,则将所有参数转换为 float32 精度
        # 否则,只将可训练参数转换为 float32 精度
        if param.requires_grad or use_cpu:  # 解决了在 CPU 上训练模型时,由于精度问题可能导致的问题
            param.data = param.data.to(torch.float32)


def load_tokenizer_and_model(
        model_dir: str,             # model_dir: 预训练模型的保存路径
        peft_config: Optional[PeftConfig] = None,   # peft_config: 可选的PEFT配置信息,用于在微调模型时设置相关参数
) -> tuple[PreTrainedTokenizer, nn.Module]:
    # trust_remote_code=True表示允许加载远程托管的模型代码,提高灵活性
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if peft_config is not None:   # 2. 如果提供了peft_config参数
        # 3. 根据PEFT配置的类型,设置不同的模型加载方式
        if peft_config.peft_type.name == "PREFIX_TUNING":
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            config.pre_seq_len = peft_config.num_virtual_tokens # 设置配置中的 pre_seq_len 参数为 PEFT 中的虚拟 token 数量
            config.use_cache = False   # 禁用模型的缓存功能
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                config=config,
            )
        if peft_config.peft_type.name == "LORA":
            # 4.1 从model_dir加载预训练的自回归语言模型,并设置empty_init=False和use_cache=False
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                # empty_init=False 是 from_pretrained 方法的默认行为，即默认情况下，from_pretrained 方法会加载预训练权重。如果明确指定 empty_init=False，其实是在重申这个默认行为，可能是为了代码的可读性，让读者或使用者清楚地知道正在加载预训练权重
                empty_init=False,  # 不使用空初始化
                use_cache=False    # use_cache=False表示不使用缓存
            )
            # 应用PEFT配置到peft模型上
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()  # 打印可训练参数的信息
    else:  # 如果没有传入PEFT配置，直接加载预训练的因果语言模型 
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,   # 不使用空初始化
            use_cache=False    # use_cache=False表示不使用缓存
        )
    print_model_size(model)
    return tokenizer, model


def compute_metrics(eval_preds: EvalPrediction, tokenizer: PreTrainedTokenizer):
    batched_pred_ids, batched_label_ids = eval_preds

    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode(pred_ids).strip() # -100 的标签会被转化为空字符串 ""
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        metrics_dct['bleu-4'].append(
            sentence_bleu(
                [label_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method3,
            )
        )
    return {k: np.mean(v) for k, v in metrics_dct.items()}

# 使用 Typer 库中的 @app.command() 装饰器定义一个命令行入口函数,这样可以通过命令行参数调用这个函数
@app.command()
def main(
        # data_dir 是数据集所在的目录路径,使用 Annotated 类型注解和 typer.Argument 指定参数属性,help 参数提供了参数的帮助信息
        data_dir: Annotated[
                            str
                            , typer.Argument(help='')],
        # model_dir 是预训练模型所在的目录路径,使用 Annotated 类型注解和 typer.Argument 指定参数属性及帮助信息
        model_dir: Annotated[
                            str,
                            typer.Argument(
                                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
                            ),
        ],
        # config_file 是配置文件的路径,使用 Annotated 类型注解和 typer.Argument 指定参数属性
        config_file: Annotated[str, typer.Argument(help='')],
        # auto_resume_from_checkpoint 是一个字符串参数,用于指定是否自动恢复训练checkpoint,使用 typer.Argument 指定参数属性及帮助信息,默认值为空字符串
        auto_resume_from_checkpoint: str = typer.Argument(
            default='',
            help='If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training'
        ),

):
    ft_config = FinetuningConfig.from_file(config_file)  # 1. 从配置文件中加载 FinetuningConfig 实例
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config) # 如果提供了 PEFT 配置,则会根据配置对模型进行微调
    data_manager = DataManager(data_dir, ft_config.data_config) # 3. 创建 DataManager 实例,用于加载和处理数据集
    # TODO 调试
    print(torch.distributed.get_rank())
    # TODO 新增代码
    logger = set_logger_with_seed(training_args=ft_config.training_args)
    logger.info(f"Training/evaluation parameters {ft_config.training_args}")
    logger.info(f"model parameters {model.config}")
    logger.info(f"PEFT parameters {ft_config.peft_config}")

    # 4. 使用 data_manager 获取训练、验证和测试数据集
        # get_dataset 仅仅得到token id, 没有进行pad
    train_dataset = data_manager.get_dataset(  
        Split.TRAIN,
        functools.partial(  # 使用 functools.partial 部分应用 process_batch 函数,传入 tokenizer 和长度限制参数
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,  # 以批量的方式获取数据集
    )
    print('train_dataset:', train_dataset)
    print("one sample of train_dataset:", train_dataset[0])

        # get_dataset 仅仅得到token id, 没有进行pad
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
        print('one sample of val_dataset:', val_dataset[0])

        # get_dataset 仅仅得到token id, 没有进行pad
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print('test_dataset:', test_dataset)
        print('one sample of test_dataset:', test_dataset[0])

    # checks encoded dataset,  查看数据集的格式
    _sanity_check(
        train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer
    )

    # turn model to fp32,  根据是否使用cpu将模型转换为 fp32, 将训练的参数转化为 float32
    _prepare_model_for_training(model, ft_config.training_args.use_cpu)
    print_model_allarguments_name_dtype(model=model)

    # 7. 设置生成任务的参数,如 pad_token_id、eos_token_id 等
    ft_config.training_args.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>'),
    ]
    # 8. 启用梯度检查点和输入梯度要求, 
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset.select(list(range(50))),    # 选择验证集的前50个样本
        # LORA 微调不需要 tokenizer
        tokenizer=tokenizer if ft_config.peft_config.peft_type != "LORA" else None,  # LORA does not need tokenizer, 
        # functools.partial是Python标准库中的一个函数，它允许我们创建一个新的可调用对象，并将部分参数预先绑定到原始函数上。这样，在调用新的可调用对象时，只需要提供剩余的参数即可。
        # 使用部分应用创建 compute_metrics 函数
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer), 
    )

    # 10. 根据 auto_resume_from_checkpoint 参数决定是否从指定的 checkpoint 恢复训练
    # TODO 修改代码
    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint.upper() == "NO" or auto_resume_from_checkpoint is None:
        # 如果未指定 checkpoint,则从头开始训练
        trainer.train()
    else:
        # 根据 auto_resume_from_checkpoint 参数决定从哪个 checkpoint 恢复训练
        output_dir = ft_config.training_args.output_dir
        dirlist = os.listdir(output_dir)
        checkpoint_sn = 0
        for checkpoint_str in dirlist:
            # 获得最新的checkpoint id
            if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                if checkpoint > checkpoint_sn:
                    checkpoint_sn = checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            if checkpoint_sn > 0:
                # 如果设置为 "YES",则自动使用最近的 checkpoint 恢复训练
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                print("resume checkpoint from  checkpoint-" + str(checkpoint_sn))
                trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                # 如果没有找到 checkpoint,则从头开始训练
                trainer.train()
        else:
            if auto_resume_from_checkpoint.isdigit():
                # 如果输入了具体的 checkpoint 序号,则从该 checkpoint 恢复训练
                if int(auto_resume_from_checkpoint) > 0:
                    checkpoint_sn = int(auto_resume_from_checkpoint)
                    model.gradient_checkpointing_enable()
                    model.enable_input_require_grads()
                    checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                    print("resume checkpoint from  checkpoint-" + str(checkpoint_sn))
                    trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                # 如果输入的 checkpoint 序号无效,则打印错误信息
                print(auto_resume_from_checkpoint,
                      "The specified checkpoint sn(" + auto_resume_from_checkpoint + ") has not been saved. Please search for the correct chkeckpoint in the model output directory")

    # test stage, 如果存在测试集,则使用训练好的模型进行预测
    if test_dataset is not None:
        trainer.predict(test_dataset)


if __name__ == '__main__':
    app()
