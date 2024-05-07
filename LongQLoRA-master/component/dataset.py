import sys
from loguru import logger
# 重定义终端logger显示颜色
logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} |<cyan><lvl>{level:8}</></>| {name} : {module}:{line:4} | <cyan>mymodule</> | - <lvl>{message}</>",
        "colorize": True
    },
])
import json
import datasets
from typing import Annotated, Any, Optional, Union
from torch.utils.data import Dataset
import numpy as np
import functools
from transformers import PreTrainedTokenizerBase

# TODO ++ 新增一些代码, 定义SFT模版, prompt = SYSTEM_PROMPT + INPUT_PROMPT + RESPONSE_PROMPT
instruction = """You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteria: ** importance and novelty **, ** potential reasons for acceptance **, ** potential reasons for rejection **, and ** suggestions for improvement **. The "Input" is the given paper, and the "Response" is your review that you need to provide.\n\n\n""".strip()
SYSTEM_PROMPT = ("Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n")
INPUT_PROMPT = "### Instruction:\n{{instruction}}\n\n### Input:\n{{input}}".replace("{{instruction}}", instruction)
RESPONSE_PROMPT = "\n\n### Response:\n"


# print(SYSTEM_PROMPT + INPUT_PROMPT + RESPONSE_PROMPT + "用于测试")
# 模版格式如下
"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a professional machine learning conference reviewer who reviews a given paper and considers 4 criteria: ** importance and novelty **, ** potential reasons for acceptance **, ** potential reasons for rejection **, and ** suggestions for improvement **. The "Input" is the given paper, and the "Response" is your review that you need to provide.

### Input:
{{input}}

### Response:
用于测试
"""


class PretrainDataset(Dataset):
    def __init__(self, 
                 file, # jsonl 格式
                 tokenizer, 
                 max_seq_length, 
                 ignore_index=-100):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        # josnl 文件
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()

        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        # json 格式, 取里面的text是文本
        text = json.loads(data)['text']
        return text

# # NOTE 弃用
# class EvalDataset(Dataset):
#     """
#     用于评测ppl
#     """
#     def __init__(self, file, tokenizer, max_seq_length, ignore_index=-100, sliding_window=256):
#         self.tokenizer = tokenizer
#         self.ignore_index = ignore_index
#         self.pad_token_id = tokenizer.pad_token_id
#         self.max_seq_length = max_seq_length
#         logger.info('Loading data: {}'.format(file))
#         token_list = np.memmap(file, dtype=np.uint16, mode='r').tolist()

#         # 以滑动窗口截取评测数据
#         eval_data_list = []
#         for i in range(0, len(token_list), sliding_window):
#             input_ids = token_list[i: i+max_seq_length]
#             labels = token_list[i: i+max_seq_length]
#             # padding
#             padding_len = self.max_seq_length - len(input_ids)
#             input_ids += [self.pad_token_id]*padding_len
#             labels += [self.ignore_index]*padding_len
#             eval_data_list.append({
#                 'input_ids': input_ids,
#                 'labels': labels
#             })
#         logger.info("there are {} data in eval dataset".format(len(eval_data_list)))
#         self.data_list = eval_data_list

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
#         data = self.data_list[index]
#         return data

# # NOTE 弃用
# class VicunaSFTDataset(Dataset):
#     def __init__(self, 
#                  file, # jsonl 格式
#                  tokenizer, 
#                  max_seq_length, 
#                  ignore_index=-100
#                 ):
#         self.tokenizer = tokenizer
#         self.ignore_index = ignore_index
#         self.max_seq_length = max_seq_length
#         self.pad_token_id = tokenizer.pad_token_id
#         self.eos_token_id = tokenizer.eos_token_id
#         logger.info('Loading data: {}'.format(file))
#         with open(file, 'r', encoding='utf8') as f:
#             data_list = f.readlines()

#         logger.info("there are {} data in dataset".format(len(data_list)))
#         self.data_list = data_list
#         self.input_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: {input}\nASSISTANT: "

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
#         """
#         沿袭Vicuna的的格式。
#         A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
#         USER: xxx
#         ASSISTANT: xxx
#         """
#         data = self.data_list[index]
#         data = json.loads(data)
#         inputs = data['input'].strip()
#         output = data['output'].strip()
#         # 输入部分
#         input_format = self.input_template.format(input=inputs)

#         input_format_ids = self.tokenizer(input_format, add_special_tokens=False).input_ids
#         output_ids = self.tokenizer(output, add_special_tokens=False).input_ids + [self.eos_token_id]

#         input_ids = input_format_ids + output_ids
#         labels = [self.ignore_index] * len(input_format_ids) + output_ids
#         assert len(input_ids) == len(labels)

#         # 对长度进行截断
#         input_ids = input_ids[:self.max_seq_length]
#         labels = labels[:self.max_seq_length]
#         attention_mask = [1] * len(input_ids)
#         # padding(右填充)
#         padding_len = self.max_seq_length - len(input_ids)
#         input_ids += [self.pad_token_id] * padding_len
#         labels += [self.ignore_index] * padding_len
#         attention_mask += [0] * padding_len

#         inputs = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'labels': labels
#         }
#         return inputs



# TODO ++ 新增代码, 该类可能会用到截取,但不进行pad, 里面的对象是 datasets 库中的对象
class CustomDataset():
    def __init__(self, 
                 file:Union[str,list], # jsonl 格式, keys: input, output
                 tokenizer:PreTrainedTokenizerBase, 
                 max_seq_length:int,      # max_seq_length 为 prompt + response 最大长度
                 # max_prompt_length + max_response_length == max_seq_length, 里面包含 <s> 和 </s> 等这些特殊符号
                 max_prompt_length:int,  
                 max_response_length:int, # max_response_length 为 response 最大长度, 超过将会被截取
                 ignore_index:int = -100,
                 num_proc:int = 10,           # 数据处理的进程数
                ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.ignore_index = ignore_index
        if isinstance(file, str):  # 如果 file 为 str 类型, 将其转为 list
            file = [file]
        self.file = file
        self.num_proc = num_proc

        logger.success(f'Loading data: {file}')
        self.dataset = datasets.load_dataset('json', data_files={"train":file},
                                        num_proc=1)       # 加载时使用一个进程
        # 仅用于代码流程debug
        # 取其中一部分数据作为验证集与测试集跑通流程
        self.dataset['validation'] = self.dataset['train'].shuffle(42).select(range(5))
        self.dataset['test'] = self.dataset['train'].shuffle(42).select(range(5, 8))

        # train 与 validation 使用相同的数据处理过程
            # 因为validation 我们仅计算验证集loss, 因此和train阶段的过程是相同的
        self.dataset["train"] = self.dataset["train"].map(
            function=functools.partial(
                self.process_batch,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                ignore_index=ignore_index
            ),
            batched=True,
            remove_columns=self.dataset["train"].column_names,
            num_proc=num_proc,              
            load_from_cache_file = True  # load_from_cache_file = False, 取消使用缓存, 用于调试
        )
        self.dataset["validation"] = self.dataset["validation"].map(
            function=functools.partial(
                self.process_batch,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                ignore_index=ignore_index
            ),
            batched=True,
            remove_columns=self.dataset["validation"].column_names,
            num_proc=num_proc,
            load_from_cache_file = True  # load_from_cache_file = False, 取消使用缓存, 用于调试
        )

        # TODO: 待完成: test集 由于存在 generate 推理截断(即prompt需要 left pad), 因此需要单独处理
        self.dataset["test"] = self.dataset["test"].map(
            function=functools.partial(
                self.process_predict_batch,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                ignore_index=ignore_index
            ),
            batched=True,
            remove_columns=self.dataset["test"].column_names,
            num_proc=num_proc,
            load_from_cache_file = False  # 取消缓存, 用于调试
        )

        logger.info(
            f"Number of train samples: {len(self.dataset['train'])} \n"
            f"Number of validation samples: {len(self.dataset['validation'])} \n"
            f"Number of test samples: {len(self.dataset['test'])} \n"
        )
        logger.success(f"Loading data done!")

    # 获得训练集/验证集/测试集的 dataset
    def get_train_dataset(self):
        return self.dataset["train"]
    def get_validation_dataset(self):
        # 如果为空, 返回 None
        return self.dataset["validation"] if len(self.dataset["validation"]) else None
    def get_test_dataset(self):
        return self.dataset["test"] if len(self.dataset["test"]) else None

    # 用于 datasets库 的map方法, 用于批处理SFT训练集/验证集(驗證集仅计算验证loss, 故也可以按照这个数据处理的方法)
    # 返回的数据格式为: {"input_ids": ..., "labels": ...}
    def process_batch(self,
                      batch, 
                      tokenizer:PreTrainedTokenizerBase, 
                      max_seq_length:int,
                      max_prompt_length:int,
                      max_response_length:int, 
                      ignore_index:int=-100):
        
        batch_input_ids = []
        batch_labels = []
        for item in zip(batch['input'],batch['output']):

            input = item[0].strip()
            output = item[1].strip()

            # 插入SFT prompt
            tmp = INPUT_PROMPT.replace("{{input}}", input)
            prompt_prefix = SYSTEM_PROMPT + tmp  # prompt前缀, 与后缀分开是为了 截取 Input 方便
            prompt_suffix = RESPONSE_PROMPT   # prompt后缀, 与前缀分开为了后面截取 Input 方便

            # 转化为id, 同时添加 bos eos
            prompt_prefix_id = [tokenizer.bos_token_id] + tokenizer.encode(prompt_prefix, add_special_tokens=False)
            prompt_suffix_id = tokenizer.encode(prompt_suffix, add_special_tokens=False)
            response_id = tokenizer.encode(output, add_special_tokens=False) + [tokenizer.eos_token_id]

            input_id = prompt_prefix_id + prompt_suffix_id + response_id
            # 数据截断
            if len(input_id) > max_seq_length:
                # 先判断 response_id 是否超出长度
                if len(response_id) > max_response_length:
                    response_id = response_id[:max_response_length]
                    # 从尾部截断 prompt_prefix_id
                    prompt_prefix_id = prompt_prefix_id[:max_seq_length - len(response_id) - len(prompt_suffix_id)]
                    input_id = prompt_prefix_id + prompt_suffix_id + response_id
                
                else: 
                    # 如果 response_id 没有超出长度, 则将 prompt 超出长度的部分会从prompt_prefix_id尾部截取掉
                    # 这种情况prompt可能会超过max_prompt_length, 训练阶段为了更长的上下文, 允许prompt超出max_prompt_length
                    prompt_prefix_id = prompt_prefix_id[:max_seq_length-len(response_id)-len(prompt_suffix_id)]
                    input_id = prompt_prefix_id + prompt_suffix_id + response_id

            # 最后再进行判断一次, 确保 input_id 不超出 max_seq_length
            if len(input_id) > max_seq_length:
                truncated_length = len(input_id) - max_seq_length
                prompt_prefix_id = prompt_prefix_id[:-truncated_length]
                input_id = prompt_prefix_id + prompt_suffix_id + response_id
            # 获得label, 不需要的位置设置为 -100
            label = [ignore_index] * (len(prompt_prefix_id) + len(prompt_suffix_id)) + response_id
            # 保存
            batch_input_ids.append(input_id)
            batch_labels.append(label)
        return {
            'input_ids': batch_input_ids,
            'labels': batch_labels
        }

    # TODO 待写, 用于处理SFT支持 predict 数据格式, 将使用于 datasets库 的map方法, 用于批处理SFT预测集
    def process_predict_batch(self,
                              batch, 
                              tokenizer:PreTrainedTokenizerBase, 
                              max_seq_length:int,
                              max_prompt_length:int,
                              max_response_length:int, 
                              ignore_index:int=-100):
        """待完成"""
        return batch


if __name__ == '__main__':
    CustomDataset()
    pass