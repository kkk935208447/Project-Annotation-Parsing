# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from itertools import chain
from dschat.utils.data import raw_datasets
from deepspeed.accelerator import get_accelerator


def get_raw_dataset(dataset_name, output_path, seed, local_rank):

    if "Dahoas/rm-static" in dataset_name:
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "Dahoas/full-hh-rlhf" in dataset_name:
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "yitingxie/rlhf-reward-datasets" in dataset_name:
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "openai/webgpt_comparisons" in dataset_name:
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "stanfordnlp/SHP" in dataset_name:
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "pvduy/sharegpt_alpaca_oa_vicuna_format" in dataset_name:
        return raw_datasets.PvduySharegptalpacaoavicunaformatDataset(
            output_path, seed, local_rank, dataset_name)
    elif "wangrui6/Zhihu-KOL" in dataset_name:
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "mkqa-Chinese" in dataset_name:
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank,
                                               "mkqa")
    elif "mkqa-Japanese" in dataset_name:
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank,
                                                "mkqa")
    elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "lmqg/qg_jaquad" in dataset_name:
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "lmqg/qag_jaquad" in dataset_name:
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank,
                                                 dataset_name)
    
    # TODO:注释官方代码
    # elif "local/jsonfile" in dataset_name:
    #     chat_path = os.path.abspath(
    #         os.path.join(os.path.dirname(__file__), os.path.pardir,
    #                      os.path.pardir, os.path.pardir))
    #     if not (os.path.isfile(chat_path + '/data/train.json')
    #             and os.path.isfile(chat_path + '/data/eval.json')):
    #         raise RuntimeError(
    #             f"Please check both the train.json and eval.json files in your applications/DeepSpeed-Chat/data directory."
    #         )
    #     return raw_datasets.LocalJsonFileDataset(output_path, seed, local_rank,
    #                                              dataset_name, chat_path)
    # else:
    #     raise RuntimeError(
    #         f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
    #     )
    
    # TODO: 添加代码,自定义本地数据集
    else:
        return raw_datasets.LocalMyDataset(output_path, seed, local_rank,
                                                 dataset_name)


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx

"""
该函数的主要作用是获取原始数据集的分割索引,用于将原始数据集划分为训练集和评估集。它涉及以下几个关键技术:
数据集分割: 该函数根据提供的data_split参数将原始数据集划分为多个部分。data_split参数是一个字符串,表示每个分割的比例。例如,"0.8,0.2"表示将数据集划分为80%的训练集和20%的评估集。
分割索引计算: 该函数计算每个分割的开始和结束索引,以便从原始数据集中提取相应的子集。它首先将data_split字符串转换为浮点数列表,然后归一化这些比例,并根据数据大小计算每个分割的开始和结束索引。
随机化: 该函数使用get_shuffle_idx函数获取随机化的索引。这可以确保数据在分割时是随机的,有助于提高模型的泛化能力。
可重建性: 该函数通过rebuild参数支持重新构建分割索引。如果rebuild为True,或者分割索引文件不存在,或者原始数据集是一个jsonfile,则会重新计算并保存分割索引。这确保了分割索引的最新性和正确性。
分割索引文件: 该函数将计算出的分割索引保存到文件中,文件名包含了数据集名称、种子值、分割名称、分割比例和分割索引。这使得分割索引可以在需要时被重用,提高了效率。
jsonfile处理: 该函数特别处理了原始数据集是jsonfile的情况。由于jsonfile更容易被修改,因此每次都需要重新计算分割索引。
通过使用这些技术,该函数可以灵活地将原始数据集划分为训练集和评估集,同时确保分割的可重复性和随机性。它还支持不同的分割比例和数据格式,如jsonfile等。
"""
def get_raw_dataset_split_index(local_rank,
                                output_path,
                                dataset_name,
                                seed,
                                split_name,
                                data_split,
                                split_index,   # 判断加载对应阶段的split_index, RLHF有三个阶段, 数据集需要分开
                                data_size,
                                rebuild=False):
    # 1. 该函数的目的是获取原始数据集的分割索引,用于将原始数据集划分为训练集和评估集。

    # 2. 构建分割索引文件的文件名,用于存储或加载分割索引。
    # TODO: 修改官方代码
    # index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    index_file_name = os.path.join(f"{output_path}",f"{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy")
    # reindex each time when using local jsonfile since it's more likely to get modified
    # 3. 如果满足以下任一条件,则需要重新计算并保存分割索引:
    #    a) rebuild参数为True,强制重建分割索引
    #    b) 分割索引文件不存在
    #    c) 原始数据集是一个jsonfile,每次都需要重新索引,因为它更可能被修改
    if rebuild or (not os.path.isfile(index_file_name)) or (dataset_name
                                                            == 'jsonfile'):
        # 4. 将data_split字符串转换为浮点数列表,表示每个分割的比例
        splits = [float(s) for s in data_split.split(',')]
        # 5. 计算分割比例的总和
        splits_sum = sum(splits)   
        # 6. 将分割比例归一化
        splits = [split / splits_sum for split in splits]
        # 7. 初始化分割索引列表
        splits_index = [0]
        # 8. 计算每个分割的开始和结束索引
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        # 9. 如果分割索引的最后一个值与数据大小不相等,则对所有分割索引进行调整
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
            # 10. 确保最后一个分割索引等于数据大小
        assert splits_index[-1] == data_size

        # 11. 获取随机化的索引
        shuffle_idx = get_shuffle_idx(seed, data_size)

        # 12. 将随机化的索引按照分割比例保存到文件中
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = os.path.join(f"{output_path}",f"{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy")
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    # 13. 加载并返回指定分割索引
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id

"""
关键点包括:

该函数支持RLHF(Reinforcement Learning from Human Feedback)训练策略,通过创建不同阶段所需的数据集。
利用了DeepSpeed的ZeRO优化技术,以高效地训练大型语言模型。
处理了端到端对话数据,并添加了对话结束标记。
使用了tokenization(分词)、padding(填充)、截断和反转张量等数据预处理技术。
可能涉及词表调整,以处理特殊标记或特定的词汇表。
返回了一个封装了prompt、chosen和reject三个数据集的PromptDataset对象。
"""
def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    # 1. 该函数的作用是从原始数据集中创建用于不同训练阶段的数据集,包括提示(prompt)、选择的响应(chosen)和拒绝的响应(reject)三个部分。
    # 2. 它将利用DeepSpeed的ZeRO优化技术,以及RLHF(Reinforcement Learning from Human Feedback)训练策略,对大型语言模型进行高效的训练。
    # 3. 以下是参数的解释:
    #    current_dataset: 当前的数据集,通常是经过分割和处理的子集
    #    raw_dataset: 原始的完整数据集,包含所有需要的数据
    #    train_phase: 整数,表示当前的训练阶段,1、2或3
    #    tokenizer: 用于对文本数据进行tokenization(分词)的tokenizer对象,通常是预训练的tokenizer
    #    end_of_conversation_token: 字符串,表示对话结束的特殊标记,用于处理端到端对话数据
    #    max_seq_len: 整数,表示输入序列的最大长度,超过该长度的序列将被截断
    prompt_dataset = []  # 4. 用于存储提示(prompt)数据集
    chosen_dataset = []  # 5. 用于存储选择的响应(chosen)数据集
    reject_dataset = []  # 6. 用于存储拒绝的响应(reject)数据集
    # 7. 第一阶段训练:仅使用选择的响应数据
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            # 8. 从原始数据集中获取选择的响应
            chosen_sentence = raw_dataset.get_prompt_and_chosen(tmp_data)  # the accept response
            if chosen_sentence is not None:
                # 9. 在响应末尾添加对话结束标记,用于处理端到端对话数据
                chosen_sentence += end_of_conversation_token
                # 10. 使用tokenizer对响应进行tokenization(分词)
                # 11. max_length参数用于设置最大序列长度,序列将被截断到该长度
                # 12. padding参数指定填充策略,以确保所有序列具有相同长度
                # 13. truncation参数指定是否允许截断过长的序列
                # 14. return_tensors参数指定返回的数据格式,这里使用PyTorch张量
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                # 15. 移除tokenization结果中的batch维度,因为这里只处理单个样本
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(0)
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                # 16. 将tokenization结果添加到chosen_dataset
                chosen_dataset.append(chosen_token)
        # 17. 打印创建的数据集信息
        print(
            f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
        )

    # 18. 第二阶段训练:使用选择的响应和拒绝的响应数据
    elif train_phase == 2:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            # 19. 从原始数据集中获取选择的响应和拒绝的响应
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                # 20. 在响应末尾添加对话结束标记
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                # 21. 使用tokenizer对选择的响应和拒绝的响应进行tokenization
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                # 23. 将tokenization结果添加到chosen_dataset和reject_dataset
                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]
                chosen_dataset.append(chosen_token)

                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]
                reject_dataset.append(reject_token)
        # 24. 打印创建的数据集信息
        print(
            f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
        )

    # 25. 第三阶段训练:phase3用到prompt，prompt将被用来生成经验数据
    elif train_phase == 3:
        filtered = 0   # 26. 统计过滤掉的提示数量
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            # 27. 从原始数据集中获取提示
            prompt = raw_dataset.get_prompt(tmp_data)
            if prompt is not None:
                # 28. 使用tokenizer对提示进行tokenization
                prompt_token = tokenizer(prompt, return_tensors="pt")
                # 29. 如果提示长度不超过max_seq_len,则保留该提示
                if prompt_token["input_ids"].size()[-1] <= max_seq_len:
                    # 30. 反转input_ids和attention_mask张量, 用于后面pad后再反转实现前面pad
                    for key_word in ["input_ids", "attention_mask"]:
                        prompt_token[key_word] = prompt_token[
                            key_word].squeeze(0).flip(0)
                    # 32. 将tokenization结果添加到prompt_dataset
                    prompt_dataset.append(prompt_token)
                else:
                    filtered += 1   # 33. 统计过滤掉的提示数量
        # 34. 打印创建的数据集信息,包括过滤掉的提示数量
        print(f'Creating dataset {raw_dataset.dataset_name_clean} '
              f'for {train_phase=} size={len(prompt_dataset)} {filtered=}')

    # 35. 返回PromptDataset对象,包含prompt、chosen和reject三个数据集
    # 36. PromptDataset可能是一个自定义的数据集类,用于封装和处理这些数据
    # 37. tokenizer.pad_token_id可能与词表调整相关,用于处理特殊标记或特定的词汇表
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, rebuild):
    # 1. 该函数的目的是创建用于训练和评估的数据集。
    # 它从原始数据集中获取训练和评估数据,对其进行分割和处理,并返回最终的训练和评估数据集。
    # local_rank: 当前进程在分布式环境中的本地rank,用于确定数据分割和处理
    # dataset_name: 原始数据集的名称或标识符
    # data_split: 数据集分割策略,例如按比例或固定大小分割
    # output_path: 存储处理后数据集的输出路径
    # train_phase: 当前训练的阶段,用于确定分割索引
    # seed: 用于确保数据集分割的可重复性
    # tokenizer: 用于对文本数据进行tokenization(分词)的tokenizer对象
    # end_of_conversation_token: 表示对话结束的特殊标记,用于处理端到端对话数据
    # max_seq_len: 输入序列的最大长度,用于截断过长的序列
    # rebuild: 布尔值,指示是否需要重新构建数据集分割索引

    # 2. 获取原始数据集
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    # 3. 从原始数据集中获取训练数据
    train_dataset = raw_dataset.get_train_data()
    # 4. 获取训练数据的分割索引
    # 这将根据本地rank、数据分割策略和训练阶段对训练数据进行分割
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset), rebuild)
    # 5. 根据分割索引从训练数据中创建子集
    train_dataset = Subset(train_dataset, train_index)
    # 6. 对训练数据进行进一步处理,包括tokenization、最大长度截断等
    # 这一步可能会涉及一些数据预处理技术,创建RLHF不同阶段所需的数据集。
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)
    # 7. 从原始数据集中获取评估数据
    eval_dataset = raw_dataset.get_eval_data()
    # 8. 获取评估数据的分割索引
    # 这将根据本地rank、数据分割策略和训练阶段对评估数据进行分割
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset), rebuild)
    # 9. 根据分割索引从评估数据中创建子集
    eval_dataset = Subset(eval_dataset, eval_index)
    # 10. 对评估数据进行进一步处理,包括tokenization、最大长度截断等
    # 这一步可能会涉及一些数据预处理技术,创建RLHF不同阶段所需的数据集。
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    # 11. 返回处理后的训练和评估数据集
    return train_dataset, eval_dataset

"""
该函数的主要目的是创建用于prompt-based微调的数据集,包括训练数据集和评估数据集。它采用了以下关键技术:

分布式数据并行:通过local_rank参数和torch.distributed模块,支持在多个进程或多个GPU上并行创建和加载数据集,以加速数据预处理过程。
数据集缓存:根据数据路径、分割比例、训练阶段、随机种子、tokenizer和序列长度等信息生成一个唯一的文件名,将创建的数据集缓存到文件中,避免重复计算。
数据集混合:支持将多个数据集合并为一个数据集,以便在训练时同时使用多个数据源。
RLHF训练:在RLHF训练阶段,可以额外添加专门用于RLHF的数据集(SFT数据集),并与原始数据集合并。
数据随机化:使用get_shuffle_idx函数生成随机索引,通过Subset对数据集进行随机打乱,增加数据多样性。
序列截断:通过max_seq_len参数控制序列的最大长度,超出该长度的序列将被截断,以控制内存占用。
该函数的主要步骤如下:

根据输入参数生成一个唯一的文件名,用于缓存数据集。
检查是否已经存在缓存的数据集文件,如果不存在或需要重新生成,则进行以下步骤:
如果只有一个数据路径,则直接创建数据集。
如果有多个数据路径,则将多个数据集合并,并进行随机打乱。
如果是RLHF训练阶段且提供了SFT数据路径,则将SFT数据集与原始数据集合并。
将生成的数据集保存到文件。
等待所有进程完成数据集创建或加载。
加载并返回训练数据集和评估数据集。
该函数的设计考虑了分布式训练、数据缓存、数据混合、RLHF训练和数据随机化等多种需求,旨在高效地创建用于prompt-based微调的数据集。通过使用这些技术,可以加速数据预处理过程,提高整体训练效率,同时支持灵活的数据组合和调整。
"""
def create_prompt_dataset(local_rank,  # 1. 本地进程的rank,用于分布式训练。在分布式训练中,每个进程都会被分配一个唯一的rank,用于区分不同的进程。通过指定local_rank,可以确保每个进程只处理自己应该处理的数据,从而实现并行化计算,提高训练效率。
                          data_path,   # 2. 一个列表,包含了训练数据集的路径。可以提供多个数据集的路径,函数会将它们合并成一个训练数据集。这种混合数据集的方式可以提高数据的多样性,从而提升模型的泛化能力。
                          data_split,  # 3. 数据集的分割比例,如"10,0,0"表示10%用于训练,0%用于验证,0%用于测试。这个参数决定了数据集如何被划分为训练集、验证集和测试集。合理的数据划分对于模型的训练和评估至关重要。
                          output_path, # 4. 生成的数据集文件的输出路径。为了加快数据加载速度,函数会将处理好的数据集缓存到磁盘文件中。output_path指定了这些缓存文件的存储位置。
                          train_phase, # 5. RLHF训练的阶段
                          seed, # 6. 用于打乱数据集的随机种子。为了增加数据的多样性,函数会对数据集进行随机打乱。通过设置相同的随机种子,可以确保在不同的运行中得到相同的打乱顺序,从而保证实验的可复现性。
                          tokenizer,   # 7. 用于对数据进行编码的tokenizer。
                          max_seq_len, # 8. 序列的最大长度,超出该长度的序列将被截断。
                          end_of_conversation_token="<|endoftext|>",  # 9. 对话结束的标记符。在对话数据中,通常需要一个特殊的标记符来表示对话的结束。end_of_conversation_token参数指定了这个标记符的值,默认为"<|endoftext|>"。
                          sft_only_data_path=[],  # sft_only_data_path参数提供了SFT数据集的路径,函数会将这些数据集与普通数据集合并
                          reload=True):  # 11. 是否重新生成数据集文件,即使已经存在也会重新生成。通常情况下,如果数据集文件已经存在,函数会直接加载这些文件,以节省时间。但是如果设置reload为True,函数会忽略已有的文件,重新生成数据集。这在需要更新数据集时非常有用。
    """
    Creates the prompt dataset
    """
    # 12. 创建输出路径,如果不存在则创建。确保输出路径存在,以便函数可以在该路径下保存生成的数据集文件。
    os.makedirs(output_path, exist_ok=True)
    # 13. 生成一个文件名,用于缓存数据集文件,文件名包含了数据路径、分割比例、训练阶段、随机种子、tokenizer名称、序列长度和SFT数据路径等信息。
    # 这个文件名被设计为唯一标识一个特定的数据集配置,以确保每个配置对应的数据集都被正确缓存。通过包含所有相关参数,可以避免不同配置之间的数据集被混淆。
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    # 14. 对文件名进行哈希,避免文件名过长。某些文件系统对文件名长度有限制,因此对文件名进行哈希可以确保文件名的长度在合理范围内。
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name

    # 15. 构造训练数据集和评估数据集的文件名。将生成的数据集分别保存为两个文件,一个用于训练,一个用于评估。
    # TODO: 修改官方代码
    # train_fname = f"{output_path}/traindata_{fname}.pt"
    # eval_fname = f"{output_path}/evaldata_{fname}.pt"
    train_fname = os.path.join(f"{output_path}",f"traindata_{fname}.pt")
    eval_fname = os.path.join(f"{output_path}",f"evaldata_{fname}.pt")


    # 16. 检查是否已经存在缓存的数据集文件。如果缓存文件存在,就可以直接加载,从而节省数据预处理的时间。
    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)

    # 17. 将cache_found广播到所有进程,确保所有进程的cache_found值相同。
    # 在分布式训练中,每个进程只能访问本地文件系统,因此需要在所有进程之间同步cache_found的值,以确保一致性。
    buf_create_cache = torch.ByteTensor([not cache_found]).to(
        get_accelerator().current_device_name())
    torch.distributed.all_reduce(buf_create_cache)

    # 18. 如果本地rank为0,并且需要创建缓存或强制重新生成缓存。
    # 通常只有rank为0的进程负责创建或重新生成缓存,以避免多个进程重复执行相同的操作。
    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        print(f'Creating prompt dataset {data_path}, {reload=}')

        # 19. 如果只有一个数据路径,则直接创建数据集。
        # 当只有一个数据源时,无需合并多个数据集,直接从该数据源创建训练集和评估集。
        if len(data_path) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank,
                data_path[0],
                data_split,
                output_path,
                train_phase,
                seed,
                tokenizer,
                end_of_conversation_token,
                max_seq_len,
                rebuild=reload)
            
        # 20. 如果有多个数据路径,则将多个数据集合并。
        # 当提供了多个数据源时,需要将它们合并为一个统一的数据集。这种数据混合有助于提高数据的多样性,从而提升模型的泛化能力。
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path in data_path:
                train_dataset, eval_dataset = create_dataset(
                    local_rank,
                    d_path,
                    data_split,
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                    rebuild=reload)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            # 21. 使用ConcatDataset将多个训练数据集合并为一个数据集。
            train_dataset = ConcatDataset(train_datasets)
            # 22. 生成一个打乱顺序的索引。为了增加数据的多样性,函数会对数据集进行随机打乱。get_shuffle_idx函数根据给定的随机种子和数据集大小生成一个随机的索引序列,用于打乱数据集。
            shuffle_idx = get_shuffle_idx(seed, train_size)
            # 23. 使用Subset按照打乱的顺序创建新的训练数据集。Subset是PyTorch中的一个数据集包装器,它可以根据给定的索引从原始数据集中取出一个子集,从而实现数据的随机打乱。
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            # 24. 使用ConcatDataset将多个评估数据集合并为一个数据集。
            eval_dataset = ConcatDataset(eval_datasets)
            # 25. 生成一个打乱顺序的索引,用于打乱评估数据集。
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            # 26. 使用Subset按照打乱的顺序创建新的评估数据集。
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
        # 27. 如果当前是RLHF的SFT训练阶段,除了现有的数据,还可以提供了其他SFT数据路径,增加模型的鲁棒性。
        if train_phase == 1 and sft_only_data_path:
            sft_train_datasets = []
            sft_eval_datasets = []
            sft_train_size = 0
            sft_eval_size = 0
            for sft_path in sft_only_data_path:
                sft_train_dataset, sft_eval_dataset = create_dataset(
                    local_rank,
                    sft_path,
                    # 28. 对于其他的SFT数据集,使用固定的分割比例"10,0,0"。这些SFT数据集通常只用于SFT训练,因此将其完全划分。
                    "10,0,0",  
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                    rebuild=reload)
                sft_train_datasets.append(sft_train_dataset)
                sft_eval_datasets.append(sft_eval_dataset)
                sft_train_size += len(sft_train_dataset)
                sft_eval_size += len(sft_eval_dataset)

            # 29. 如果存在新的SFT训练数据集,则将其与原始SFT训练数据集合并。
            # 形成一个更大的训练数据集,用于RLHF训练。这种数据合并可以让模型同时利用两种类型的数据,提高训练效果。
            if sft_train_datasets:  # Check if sft_train_datasets is not empty
                sft_train_dataset = ConcatDataset(sft_train_datasets)
                train_dataset = ConcatDataset(
                    [train_dataset, sft_train_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
                train_dataset = Subset(train_dataset, shuffle_idx.tolist())

            # 30. 如果存在新的SFT评估数据集,则将其与原始SFT评估数据集合并。
            # 形成一个更大的评估数据集,用于评估RLHF训练后的模型。
            if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
                sft_eval_dataset = ConcatDataset(sft_eval_datasets)
                eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
                eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
        # 31. 将生成的训练数据集和评估数据集保存到文件。
        # 将处理好的数据集(已处理后的)缓存到磁盘文件中,以便后续快速加载,避免重复计算。
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    # 32. 等待所有进程完成数据集创建或加载。
    # 在分布式训练中,需要确保所有进程都完成了数据集的创建或加载,以避免进程之间的不同步。
    torch.distributed.barrier()
    # 33. 加载并返回训练数据集和评估数据集。
    # 如果数据集文件已经存在,就直接加载这些文件;否则,就从前面的步骤中获取新创建的数据集。
    return torch.load(train_fname), torch.load(eval_fname)


class DataCollatorReward:
    # 1. DataCollatorReward 是一个自定义的数据收集器类,用于处理奖励函数训练数据的批处理。
    # 4. __call__ 方法是数据收集器的主要逻辑,在每次调用时处理一个数据批次。
    def __call__(self, data):
        batch = {} # 3. 创建一个字典,用于存储处理后的批次数据。
        # 4. 将提示序列和响应序列的 input_ids 拼接到一个张量中,形成批次输入。
        batch["input_ids"] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        # 5. 将提示序列和响应序列的注意力掩码拼接到一个张量中,形成批次注意力掩码。
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        # 6. 返回处理后的批次数据。
        return batch


class DataCollatorRLHF:
    # 1. DataCollatorRLHF 是一个自定义的数据收集器类,用于处理训练数据的批处理。
    # 2. max_token_len 参数指定了最大序列长度,用于对输入数据进行补齐。
    # 3. inference_tp_size 参数指定了推理时的 tensor parallelism 大小,这是一种分布式训练优化技术。
    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    # 4. __call__ 方法是数据收集器的主要逻辑,在每次调用时处理一个数据批次。
    def __call__(self, data):
        # 5. 创建一个字典,用于存储处理后的批次数据。
        batch = {}
        # 6. 获取填充 token 的 ID,这是用于补齐序列的特殊 token。
        pad_token_id = data[-1][-1]
        # 7. 使用 pad_sequence 函数对提示序列和提示注意力掩码进行补齐。
        # pad_sequence 是一个 PyTorch 工具函数,用于对批次样本进行填充,确保所有样本长度相同。
        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        # 8. 检查提示序列的长度是否小于最大序列长度,如果小于则进行填充。
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            # 9. 使用 F.pad 函数对提示序列和注意力掩码进行填充,填充值为 pad_token_id 和 0。
            batch["prompt"] = F.pad(prompt,
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(0, pad_length),
                                             mode='constant',
                                             value=0)
        else:
            # 10. 如果提示序列长度已经达到最大长度,则直接将其添加到批次数据中。
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        # 11. 对提示序列和注意力掩码进行反转,这是为了适应某些模型的输入要求。
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


# 1. args 参数是一个命令行参数对象,包含了各种配置选项。
# 2. tokenizer 参数是一个用于文本标记化和解码的 Tokenizer 对象。
def get_unsupervised_data(args, tokenizer):
    # 3. 使用 load_dataset 函数从给定的数据集名称和配置加载未监督训练数据。
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    # 4. 获取原始训练数据集的列名。
    column_names = unsupervised_raw_datasets["train"].column_names
    # 5. 确定文本列的名称,如果有 "text" 列则使用它,否则使用第一个列。
    text_column_name = "text" if "text" in column_names else column_names[0]
    # 6. 定义一个 tokenize_function 函数,使用 tokenizer 对数据集进行标记化。
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    # 7. 使用 map 函数对原始数据集进行标记化处理,并缓存结果。
    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,  # 8. 使用批处理加速标记化过程。
        num_proc=args.preprocessing_num_workers,  # 9. 使用多进程加速标记化。
        remove_columns=column_names,        # 10. 在处理后删除原始列,仅保留标记化后的数据。
        load_from_cache_file=True,      # 11. 从缓存文件加载数据,避免重复处理。
        desc="Running tokenizer on dataset",  # 12. 为进度条添加描述信息。
    )
    # 13. 计算最大输入序列长度。
    block_size = args.max_prompt_seq_len + args.max_answer_seq_len
    # 14. 定义一个 group_texts 函数,用于将标记化后的数据集划分为固定长度的块。
    def group_texts(examples):
        # Concatenate all texts.
        # 15. 将所有文本拼接到一个列表中。
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        # 16. 舍弃最后一个不足 block_size 的部分,可以根据需求进行修改。
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        # 17. 将拼接后的文本划分为固定长度的块。
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # 18. 将输入 ID 复制到 labels 中,用于无监督预训练。
        result["labels"] = result["input_ids"].copy()
        return result
    # 19. 使用 map 函数对标记化后的数据集进行分块处理,并缓存结果。
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,  # 20. 使用批处理加速分块过程。
        num_proc=args.preprocessing_num_workers,  # 21. 使用多进程加速分块。
        load_from_cache_file=True,  # 22. 从缓存文件加载数据,避免重复处理。
        desc=f"Grouping texts in chunks of {block_size}",  # 23. 为进度条添加描述信息。
    )
    # 24. 从处理后的数据集中获取训练数据集。
    train_dataset = lm_datasets["train"]

    return train_dataset # 25. 返回经过预处理的无监督训练数据集。


class MiniDataset:
    # 1. MiniDataset 是一个自定义的小型数据集类,用于管理和处理模型训练中的批次数据。

    # 2. max_size 参数指定了数据集的最大容量。
    # 3. small_batch_size 参数指定了小批次的大小。
    def __init__(self, max_size, small_batch_size):
        self.dataset = [] #4. 创建一个列表,用于存储大批次数据。
        self.max_size = max_size
        self.small_batch_size = small_batch_size
    # 5. seperate 方法用于将大批次数据拆分为小批次。
    def seperate(self):
        small_dataset = []  # 6. 创建一个新的列表,用于存储拆分后的小批次数据。
        for large_batch in self.dataset:  # 7. 遍历所有大批次数据。
            if type(large_batch) == list or type(large_batch) == tuple:  # 8. 判断大批次数据的类型,如果是列表或元组。
                large_size = len(large_batch[0])   # 9. 获取大批次数据的长度。
            elif type(large_batch) == dict:   # 10. 如果大批次数据是字典。
                large_size = len(large_batch[list(large_batch.keys())[0]])  # 11. 获取字典中第一个值的长度。
            else:   # 12. 如果大批次数据是单个张量。
                large_size = len(large_batch)  # 13. 获取张量的长度。
            # 14. 将大批次数据拆分为多个小批次,并添加到 small_dataset 中。
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()  # 15. 清空原始的大批次数据集。

        return small_dataset   # 16. 返回拆分后的小批次数据集。

    def add(self, data):   # 17. add 方法用于向数据集中添加大批次数据。
        if len(self.dataset) < self.max_size:   # 18. 检查数据集是否未满。
            self.dataset.append(data)   # 19. 将大批次数据添加到数据集中。
            if len(self.dataset) == self.max_size:  # 20. 检查是否已达到最大容量。
                return self.seperate()   # 21. 如果已达到最大容量,则拆分大批次数据为小批次。
            else:
                return None   # 22. 如果未达到最大容量,则返回 None。
        else:
            # 23. 如果数据集已满,则抛出异常。
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )
    # 24. free 方法用于清空原始的大批次数据集。
    def free(self):
        self.dataset = []
