#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# 你也可以根据需要修改这个脚本来适应自己的序列到序列任务,相关的指示会以注释的形式留在代码中。

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset  # Hugging Face的datasets库用于加载数据集
import jieba 
# 导入rouge_chinese和nltk.translate.bleu_score用于计算ROUGE和BLEU评估指标
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,  # DataCollatorForSeq2Seq用于批处理序列到序列任务的数据
    HfArgumentParser,        # HfArgumentParser用于解析命令行参数
    Seq2SeqTrainingArguments, # Seq2SeqTrainingArguments用于定义训练参数
    set_seed,                 # set_seed用于设置随机种子
)
from trainer_seq2seq import Seq2SeqTrainer      # 自定义的Seq2SeqTrainer类用于训练序列到序列模型

# 从自定义的arguments模块导入ModelArguments和DataTrainingArguments类,用于设置模型和数据相关的参数。
from arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)  # 7. 创建一个日志记录器,用于记录脚本运行时的日志信息。

# TODO 新增: 打印模型的是否参与训练的参数名和数据类型
def print_model_allarguments_name_dtype(logger,model):
    for n,v in model.named_parameters():
        if v.requires_grad:
            logger.info(f"model trainable args: {n} - {v.dtype} - {v.shape}")
        else:
            logger.info(f"model not trainalbe args: {n} - {v.dtype} - {v.shape}")

def main():
    # 8. 创建一个HfArgumentParser对象,用于解析命令行参数。它接受一个元组,包含三个自定义的参数类:ModelArguments、DataTrainingArguments和Seq2SeqTrainingArguments。
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # 9. 检查命令行参数的数量和格式。如果只传递了一个参数,并且该参数是一个JSON文件,则使用parse_json_file方法从JSON文件中解析出模型参数、数据参数和训练参数。
        #    否则,使用parse_args_into_dataclasses方法从命令行参数中解析出这些参数。
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    # Log on each process the small summary:
    # 14. 记录一些重要的训练参数信息,包括进程等级、设备、GPU数量、是否进行分布式训练、是否使用fp16位浮点数训练等。
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    # 15. 在初始化模型之前,使用训练参数中指定的种子设置随机种子,以确保实验的可重复性。
    set_seed(training_args.seed)


    # Load dataset
    # 16. 创建一个字典data_files,用于存储训练集、验证集和测试集文件的路径。如果相应的文件参数不为None,则将文件路径添加到字典中。同时,根据文件扩展名推断出数据集的格式(如txt、csv等)。
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    # 17. 使用Hugging Face的load_dataset函数加载原始数据集。该函数接受以下参数:数据集格式(extension)、数据文件路径字典(data_files)、缓存目录(cache_dir)和是否使用身份验证令牌(use_auth_token)。
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load pretrained model and tokenizer
    # trust_remote_code=True, 是否允许在其自己的建模文件中定义Hub上的自定义模型。
    # 只有对你信任并且已阅读了代码的存储库，才应将此选项设置为“True”，因为它会在本地机器上执行Hub上存在的代码。
    # 并且huggingface 会把这些代码放入家目录中的 .cache 文件夹中
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # 同时,设置配置中的pre_seq_len和prefix_projection参数,这两个参数与P-tuning技术(一种用于微调大型语言模型的技术)相关。
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection
    # ChatGLM Tokenizer 默认pad的方式是left pad, 这一点可以看 tokenization_chatglm.py 的实现方式
    # trust_remote_code=True, 是否允许在其自己的建模文件中定义Hub上的自定义模型。
    # 只有对你信任并且已阅读了代码的存储库，才应将此选项设置为“True”，因为它会在本地机器上执行Hub上存在的代码。
    # 并且huggingface 会把这些代码放入家目录中的 .cache 文件夹中
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    # 20. 如果提供了P-tuning检查点路径,则进行评估并加载额外的前缀编码器状态字典。首先使用AutoModel.from_pretrained从预训练模型路径加载模型,然后从P-tuning检查点路径加载前缀编码器的状态字典。接着,过滤出前缀编码器相关的键值对,并将它们加载到模型的前缀编码器中。这一步骤是为了在评估时使用经过P-tuning的模型。
    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        # TODO 新增代码
        logger.info(f"loading extra state dict of prefix encoder from {model_args.ptuning_checkpoint}")
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        # TODO 新增代码
        logger.info(f"loaded extra state dict of prefix encoder from {model_args.ptuning_checkpoint}")
    else:  # 21. 如果没有提供P-tuning检查点路径,则直接从预训练模型路径加载模型。
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        # 22. 如果指定了量化位数,则对模型进行量化。量化是一种压缩模型的技术,可以减小模型的大小和内存占用,但可能会略微降低模型的精度。
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    # TODO 疑问:以下两个操作,将会取消上面的量化操作, 这是何意呢?
            # 答: 下面的操作会把量化层的 weight_sclar 转化为对应的精度, 被量化的 weight 依然是 int8 类型
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        # 23. 如果指定了pre_seq_len参数(与P-tuning技术相关),则将模型转换为半精度(16位浮点数)表示,但前缀编码器仍使用全精度(32位浮点数)表示。这是P-tuning v2技术的一种优化方式。
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        # Finetune
        # 24. 如果没有指定pre_seq_len参数,则将模型转换为全精度(32位浮点数)表示,用于普通的微调。
        model = model.float()  
    

    # TODO 新增:打印模型的是否参与训练的参数名和数据类型
    print_model_allarguments_name_dtype(logger, model)

    # 25. 获取数据参数中指定的源前缀(source_prefix),如果没有指定,则将其设置为空字符串。
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # 26. 预处理数据集。根据训练参数指定的操作(训练、评估或预测),从相应的原始数据集中获取列名。如果没有指定任何操作,则打印提示信息并返回。
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    # 27. 从数据参数中获取输入/目标列的列名,包括prompt_column(提示列)、response_column(响应列)和history_column(历史列)。
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    
    # Temporarily set max_target_length for training.
    # 28. 临时设置训练时的最大目标长度(max_target_length),用于在预处理函数中截断目标序列。
    max_target_length = data_args.max_target_length

    def preprocess_function_eval(examples):
        # 用于在评估和预测时预处理数据集。
        inputs, targets = [], []
        #     1) 遍历数据集,提取有效的输入(prompt)和目标(response)。
        #     2) 为每个输入构建提示(prompt),其中包括查询(query)和历史信息(history,如果存在)。
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query = examples[prompt_column][i]
                history = examples[history_column][i] if history_column is not None else None
                prompt = tokenizer.build_prompt(query, history)
                inputs.append(prompt)
                targets.append(examples[response_column][i])

        #     3) 将源前缀(prefix)添加到每个输入的开头。
        #     4) TODO, 与训练时不同的是, 输入序列进行前向pad, 同时得到attention mask 与 position ids, 截断为最大的长度
        #     5) 目标序列进行编码, 不进行pad , 同时得到attention mask 与 position ids, 并截断到指定的最大目标长度。
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
        # 指定了忽略填充令牌的选项
        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_function_train(examples):
        # 30. 定义一个名为preprocess_function_train的函数,用于在训练时预处理数据集。
        #     1) 计算最大序列长度,等于最大源长度、最大目标长度和一个特殊标记(通常是EOS标记)的长度之和。
        max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
        # 31. 创建一个字典model_inputs,用于存储输入序列的token id和对应的标签序列的token id。
        model_inputs = {
            "input_ids": [],
            "labels": [],
            # # TODO 修改: 加入attention mask
              # # 这里取消了修改, 因为 ChatGLMPreTrainedModel 中 get_masks 方法会计算新的 attention mask, 而且训练阶段通过将无效的标签设置为-100, 屏蔽了无效信息的loss
            # "attention_mask": []
        }
        for i in range(len(examples[prompt_column])):  # 32. 遍历数据集,提取有效的输入(prompt/query)和目标(response/answer)。
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]
                # 33. 如果存在历史列,则从中提取历史信息,并与查询一起构建提示(prompt)。
                history = examples[history_column][i] if history_column is not None else None
                prompt = tokenizer.build_prompt(query, history)
                # 34. 将源前缀(prefix)添加到提示的开头,然后使用分词器对提示进行编码,生成输入序列的token id(a_ids)。编码时添加特殊标记,并截断到指定的最大源长度。
                prompt = prefix + prompt
                # XXX 这里面由于没有添加 padding, 因此tokenizer不会对序列进行pad,但是会截断,且样本每次仅仅输入一个
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                         max_length=data_args.max_source_length)
                # 35. 使用分词器对目标序列(answer)进行编码,生成目标序列的token id(b_ids)。编码时不添加特殊标记,并截断到指定的最大目标长度。
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                         max_length=data_args.max_target_length)
                # 36. 计算上下文长度(等于输入序列的长度),然后将输入序列(a_ids)和目标序列(b_ids)连接,并在末尾添加EOS标记,形成最终的输入序列(input_ids)。
                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
                # 37. 构建标签序列(labels)。首先使用分词器的填充标记填充一个等于上下文长度的序列,然后将目标序列(b_ids)和EOS标记附加到后面。
                labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
                # 38. 计算需要填充的长度(pad_len),然后将输入序列(input_ids)和标签序列(labels)填充到最大序列长度。
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                # # TODO 修改: 添加 attention mask
                # attention_mask = [1] * (max_seq_length - pad_len) + [0] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                # 39. 如果指定了忽略填充标记的选项,则将标签序列中的填充标记替换为一个特殊值(-100),以在计算损失时忽略这些位置。
                if data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
                # 40. 将处理好的输入序列和标签序列添加到model_inputs字典中。
                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
                # # TODO 修改: 添加 attention mask
                # model_inputs["attention_mask"].append(attention_mask)

        return model_inputs  # 41. 返回model_inputs字典,它包含了预处理后的输入序列和标签序列。
    
    # 42. 定义一个打印数据集示例的函数,用于调试和查看预处理后的数据格式。
    def print_dataset_example(example):
        print("input_ids:\n", example["input_ids"])
        print("inputs:\n", tokenizer.decode(example["input_ids"]))
        print("label_ids:\n", example["labels"])
        print("labels:\n", tokenizer.decode(example["labels"]))
    
    if training_args.do_train:  # 43. 如果指定了进行训练(do_train=True),则检查原始数据集中是否包含训练集。
        if "train" not in raw_datasets:  #     1) 如果没有训练集,则引发ValueError异常。
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:    # 如果指定了最大训练样本数(max_train_samples),则从训练集中选择指定数量的样本。
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        # 44. 使用train_dataset.map方法对训练集进行预处理。
        #     1) 应用preprocess_function_train函数对数据集进行预处理。
        #     2) 指定batched=True,表示对整个批次进行预处理,而不是单个样本。
        #     3) 指定num_proc参数,控制使用的进程数量。
        #     4) 指定remove_columns参数,从数据集中删除原始的列名。
        #     5) 指定load_from_cache_file参数,决定是否从缓存文件加载预处理结果(如果存在)。
        #     6) 指定desc参数,为预处理过程添加描述。
        # 使用context manager training_args.main_process_first确保预处理只在主进程中执行一次。
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # 45. 打印训练集的第一个示例,用于查看预处理后的数据格式。
        print_dataset_example(train_dataset[0])

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length  # 将最大目标长度设置为val_max_target_length参数的值。
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:   #  如果指定了最大评估样本数(max_eval_samples),则从验证集中选择指定数量的样本。
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length  # 将最大目标长度设置为val_max_target_length参数的值。
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:  # 如果指定了最大预测样本数(max_predict_samples),则从测试集中选择指定数量的样本。
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(predict_dataset[0])

    # Data collator
    # 52. 确定标签的填充标记ID。如果指定了忽略填充标记的选项,则使用-100作为填充标记ID,否则使用分词器的填充标记ID。
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # 会为训练数据添加attention mask和position ids，用于模型的前向传播
    # 而对于验证的数据,由于这些数据中原本就存在 attention mask 与 position ids, 故这些值会保留并转化为tensor
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False       # 预处理阶段已经完成,不需要padding
    )

    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds   # preds: array (num_eval_samples, max_target_length) 需要注意的是,里面包含pad, labels: array(num_eval_samples, max_target_length)
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # 54. 创建一个字典score_dict,用于存储ROUGE-1、ROUGE-2、ROUGE-L和BLEU-4四种评估指标的分数。
        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        # 55. 对于每个预测序列(pred)和标签序列(label),使用jieba进行分词,分别得到假设(hypothesis)和参考(reference)。
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            # 56. 创建Rouge对象,计算假设和参考之间的ROUGE分数,得到包含ROUGE-1、ROUGE-2和ROUGE-L三种指标的结果字典(result)。
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            # 57. 遍历结果字典,将ROUGE-1、ROUGE-2和ROUGE-L三种指标的F1值(乘以100后四舍五入到小数点后四位)添加到score_dict中对应的列表中。
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            # 58. 使用nltk的sentence_bleu函数计算BLEU-4分数,并将其(乘以100后四舍五入到小数点后四位)添加到score_dict的"bleu-4"列表中。
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))
        # 59. 对于score_dict中每种指标的分数列表,计算它们的均值,并将结果转换为浮点数存储在score_dict中。
        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # Override the decoding parameters of Seq2SeqTrainer
    # 61. 覆盖Seq2SeqTrainer的解码参数。
    #     1) 将生成的最大长度(generation_max_length)设置为训练参数中指定的值,如果没有指定,则使用数据参数中的val_max_target_length。
    #     2) 将生成的beam数量(generation_num_beams)设置为数据参数中指定的值,如果没有指定,则使用训练参数中的默认值。
    # 这些参数控制了模型在生成输出序列时使用的策略和限制,对结果的质量和效率有重要影响。
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    # Initialize our Trainer
    # 62. 初始化Seq2SeqTrainer对象。
    #     1) 传入模型(model)、训练参数(training_args)、训练数据集(train_dataset)、评估数据集(eval_dataset)、分词器(tokenizer)和数据集器(data_collator)。
    #     2) 如果需要进行生成式预测(predict_with_generate=True),则传入评估指标计算函数compute_metrics。
    #     3) 如果使用了P-tuning技术(pre_seq_len不为None),则将save_changed设置为True,表示需要保存模型的变化。
    # Seq2SeqTrainer是一个定制的训练器,专门用于训练序列到序列模型,如机器翻译、文本摘要等任务。
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        save_changed=model_args.pre_seq_len is not None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:  #     1) 如果提供了resume_from_checkpoint参数,则从指定的检查点继续训练。
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        # 64. 启用梯度检查点和输入需要梯度的功能,这些技术可以减少训练过程中的内存占用,但可能会导致计算速度略微降低。
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        # 65. 调用trainer.train方法开始训练,如果提供了检查点,则从检查点继续训练。
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload
        # 66. 获取训练过程中的指标(如损失、准确率等),并计算训练样本数量。
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
         # 67. 记录训练指标,保存指标到文件,并保存模型状态。
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        #     1) 调用trainer.evaluate方法进行评估,传入评估数据集、评估指标前缀、是否进行采样(do_sample)、top_p采样参数、最大序列长度(max_length)和温度参数(temperature)。
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.85, max_length=max_seq_length, temperature=0.95)
        #     2) 计算评估样本数量。
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        #     3) 将评估指标存储在metrics字典中。
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
         # 69. 记录评估指标,并将其保存到文件中。
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        #     1) 调用trainer.predict方法进行预测,传入预测数据集、指标前缀、最大序列长度、是否进行采样、top_p采样参数和温度参数。
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_seq_length, do_sample=True, top_p=0.85, temperature=0.95)
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        # 71. 记录预测指标,并将其保存到文件中。
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():         # 仅主进程
            if training_args.predict_with_generate:  # 72. 如果进行了生成式预测,则将预测结果和标签序列解码为文本,并将它们保存到文件中。
                predictions = tokenizer.batch_decode(    #     1) 使用tokenizer.batch_decode将预测结果和标签序列解码为文本。
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]  #     2) 去除文本中的特殊标记和多余空格。
                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                #     3) 将预测结果和标签序列以JSON格式写入文件。
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for p, l in zip(predictions, labels):
                        res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                        writer.write(f"{res}\n")
    return results   # 73. 返回results字典,它包含了训练、评估和预测过程的一些结果和指标。


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
