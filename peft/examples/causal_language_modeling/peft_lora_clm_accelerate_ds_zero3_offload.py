import gc
import os
import sys
import threading

import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import (
    LoraConfig, 
    TaskType,  # PEFT 的 TaskType, 例如: TaskType.CAUSAL_LM
    get_peft_model
)

# levenshtein_distance 函数用于计算两个字符串之间的编辑距离(Levenshtein 距离)
# 这种距离度量经常用于评估字符串的相似程度
def levenshtein_distance(str1, str2):
    # TC: O(N^2)  # 时间复杂度: O(N^2)
    # SC: O(N)    # 空间复杂度: O(N)
    if str1 == str2:
        return 0  # 如果两个字符串相同,直接返回0
    num_rows = len(str1) + 1    # 行数为第一个字符串长度加1
    num_cols = len(str2) + 1    # 列数为第二个字符串长度加1
    dp_matrix = list(range(num_cols))  # 初始化动态规划矩阵的第一行
    for i in range(1, num_rows):  # 遍历剩余的行
        prev = dp_matrix[0]       # 保存上一行的第一个元素
        dp_matrix[0] = i          # 更新当前行的第一个元素
        for j in range(1, num_cols):   # 遍历当前行的其余列
            temp = dp_matrix[j]        # 保存上一行当前列的元素
            if str1[i - 1] == str2[j - 1]:   # 如果两个字符相同
                dp_matrix[j] = prev          # 当前元素等于左上方的元素
            else:
                # 否则取左上方、左方和上方三个元素的最小值加1
                dp_matrix[j] = min(prev, dp_matrix[j], dp_matrix[j - 1]) + 1
            prev = temp   # 更新 prev 为上一行当前列的元素
    return dp_matrix[num_cols - 1]  # 返回最后一个元素,即编辑距离

# get_closest_label 函数用于从给定的类别列表中找到与预测结果最相近的类别标签
# 它使用 levenshtein_distance 函数来计算编辑距离,返回距离最小的类别标签
def get_closest_label(eval_pred, classes):
    min_id = sys.maxsize                    # 初始化最小ID为最大值
    min_edit_distance = sys.maxsize         # 初始化最小编辑距离为最大值
    for i, class_label in enumerate(classes):   # 遍历所有类别标签
        edit_distance = levenshtein_distance(eval_pred.strip(), class_label)  # 计算编辑距离
        if edit_distance < min_edit_distance:   # 如果当前距离更小
            min_id = i    # 更新最小ID
            min_edit_distance = edit_distance   # 更新最小编辑距离
    return classes[min_id]     # 返回距离最小的类别标签


# Converting Bytes to Megabytes, b2mb 函数用于将字节数转换为兆字节数
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
# TorchTracemalloc 是一个上下文管理器,用于跟踪进程的内存使用情况
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()                                 # 执行垃圾回收
        torch.cuda.empty_cache()                     # 清空 CUDA 缓存
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero, 重置最大分配内存计数器
        self.begin = torch.cuda.memory_allocated()   # 记录初始 CUDA 内存使用量
        self.process = psutil.Process()              # 获取当前进程信息

        self.cpu_begin = self.cpu_mem_used()         # 记录初始 CPU 内存使用量
        self.peak_monitoring = True                  # 设置为监控内存峰值模式
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)   # 创建监控线程
        peak_monitor_thread.daemon = True            # 设置为守护线程
        peak_monitor_thread.start()                  # 启动监控线程
        return self   # 返回实例本身

    # cpu_mem_used 方法用于获取当前进程的常驻内存集大小
    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    # peak_monitor_func 方法是一个循环,用于持续监控和更新 CPU 内存峰值
    def peak_monitor_func(self):
        self.cpu_peak = -1   # 初始化 CPU 内存峰值为-1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)   # 更新 CPU 内存峰值

            # can't sleep or will not catch the peak right (this comment is here on purpose), 不能使用 sleep,否则无法捕获内存峰值
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:   # 如果不再监控内存峰值
                break   # 退出循环

    def __exit__(self, *exc):
        self.peak_monitoring = False         # 关闭内存峰值监控模式

        gc.collect()                          # 执行垃圾回收
        torch.cuda.empty_cache()              # 清空 CUDA 缓存
        self.end = torch.cuda.memory_allocated()              # 记录结束时的 CUDA 内存使用量
        self.peak = torch.cuda.max_memory_allocated()         # 记录 CUDA 内存峰值
        self.used = b2mb(self.end - self.begin)               # 计算 CUDA 内存使用量(MB)
        self.peaked = b2mb(self.peak - self.begin)            # 计算 CUDA 内存峰值(MB)

        self.cpu_end = self.cpu_mem_used()                    # 记录结束时的 CPU 内存使用量
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)         # 计算 CPU 内存使用量(MB)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)         # 计算 CPU 内存峰值(MB)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def main():
    # 创建 Accelerator 实例,用于在多 GPU 和分布式环境下加速训练和推理过程
    # Accelerator 是 Hugging Face 提供的一个工具,可以自动处理数据并行、模型并行等复杂操作,大大简化了分布式训练的难度
    accelerator = Accelerator()

    # 指定预训练的大型语言模型路径,这里使用的是 BigScience 开源的 Bloom 模型
    model_name_or_path = "bigscience/bloomz-7b1"

    # 指定数据集名称,这里使用的是 RAFT (Roadmap for Alleviating Flawed Testing) 中的 "twitter_complaints" 数据集
    dataset_name = "twitter_complaints"

    # 创建 LoraConfig 实例,用于配置 PEFT (Parameter-Efficient Fine-Tuning) 的参数
    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,  # TaskType.CAUSAL_LM 表示这是一个因果语言模型任务,即根据给定的文本生成连贯的后续内容
                        inference_mode=False,          # inference_mode=False 表示该配置用于微调训练,而不是推理
                        r=8,                           # r=8 表示 Lora 矩阵的秩为 8,秩越大,模型能表示的知识就越多,但也需要更多参数
                        lora_alpha=32,                 # lora_alpha=32 是 Lora 缩放系数,用于控制 Lora 矩阵的大小
                        lora_dropout=0.1               # lora_dropout=0.1 是 Lora 矩阵的 Dropout 率,用于防止过拟合
                    )
    
    # 指定数据集中文本列和标签列的名称
    text_column = "Tweet text"
    label_column = "text_label"

    # 设置训练超参数
    lr = 3e-3
    num_epochs = 20   # 训练轮数
    batch_size = 8
    seed = 42
    max_length = 64   # 最大序列长度,超过该长度的输入将被截断
    do_test = False   # 是否在训练后进行测试集评估,这里设置为 False
    set_seed(seed)

    dataset = load_dataset("ought/raft", dataset_name)
    # 获取标签列表,并将下划线替换为空格,使标签更加可读
    classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
    # 将数值标签转换为文本形式,并添加到数据集中
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["Label"]]},
        batched=True,
        num_proc=1,
    )

    # 加载分词器,用于将文本转换为模型可接受的输入形式
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # 定义预处理函数,用于将文本和标签转换为模型输入
    # 该函数将输入文本和标签拼接,并使用分词器进行编码,同时执行填充和截断操作
    def preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 定义测试预处理函数,用于将测试集文本转换为模型输入
    # 与预处理函数类似,但不需要处理标签
    def test_preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        model_inputs = tokenizer(inputs)
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        return model_inputs

    # with accelerator.main_process_first(): 是一个上下文管理器,用于确保在多进程环境下,某些操作只在主进程中执行,而不是在所有进程中重复执行。
    # 只在主进程中执行数据预处理,然后将结果广播给其他进程使用。
    # 在使用 accelerator.main_process_first() 上下文管理器的情况下,广播的过程隐含在了 accelerator.wait_for_everyone() 的调用中。
    # 在内部实现中, accelerator.wait_for_everyone() 会负责将主进程中预处理好的数据广播到其他进程,确保所有进程都可以使用相同的预处理结果。
    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    # 获取处理后的训练集
    train_dataset = processed_datasets["train"]

    # 对训练集和测试集应用测试预处理函数,并将结果缓存
    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            test_preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    # 获取处理后的评估集和测试集    
    eval_dataset = processed_datasets["train"]
    test_dataset = processed_datasets["test"]

    # 创建数据加载器,用于从数据集中按批次取出数据
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )

    print(next(iter(train_dataloader)))

    # creating model, 从预训练模型加载 AutoModelForCausalLM 实例
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)  # 使用 PEFT 对模型进行参数高效微调
    model.print_trainable_parameters()    # 打印可训练参数的数量

    # optimizer, 使用 AdamW 优化器,将模型的所有参数作为优化对象,学习率为 lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 创建学习率调度器
    # 使用线性学习率warmup调度器,warmup步数为0,总训练步数为训练集大小乘以轮数
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # 使用 Accelerator 准备模型、数据加载器、优化器和学习率调度器
    # 这一步会自动处理分布式训练所需的数据并行、梯度同步等操作
    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
                                                            model, train_dataloader, 
                                                            eval_dataloader, test_dataloader, 
                                                            optimizer, lr_scheduler
                                                            )
    accelerator.print(model)  # 打印模型信息

    # 检查是否使用了 DeepSpeed ZeRO-3,这是一种大规模模型并行技术
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    # 训练循环
    for epoch in range(num_epochs):
        # 启动内存跟踪器,用于记录 GPU 和 CPU 内存的使用情况
        with TorchTracemalloc() as tracemalloc:
            model.train()     # 将模型设置为训练模式
            total_loss = 0    # 初始化总损失为 0
            # 遍历训练数据加载器中的批次
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)  # 将批次数据传入模型,获取输出
                loss = outputs.loss       # 获取损失值
                total_loss += loss.detach().float()  # 累加损失值
                accelerator.backward(loss)           # 计算梯度
                optimizer.step()                     # 更新模型参数
                lr_scheduler.step()            # 更新学习率
                optimizer.zero_grad()          # 清空梯度
        
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        # 打印当前 epoch 下的 GPU 和 CPU 内存使用情况
        accelerator.print(f"GPU Memory before entering the train : {b2mb(tracemalloc.begin)}")
        accelerator.print(f"GPU Memory consumed at the end of the train (end-begin): {tracemalloc.used}")
        accelerator.print(f"GPU Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}")
        accelerator.print(
            f"GPU Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
        )

        accelerator.print(f"CPU Memory before entering the train : {b2mb(tracemalloc.cpu_begin)}")
        accelerator.print(f"CPU Memory consumed at the end of the train (end-begin): {tracemalloc.cpu_used}")
        accelerator.print(f"CPU Peak Memory consumed during the train (max-begin): {tracemalloc.cpu_peaked}")
        accelerator.print(
            f"CPU Total Peak Memory consumed during the train (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}"
        )
        # 计算当前 epoch 的训练集损失和困惑度(PerplexityLoss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        # 打印结果(单进程)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        model.eval()       # 将模型设置为评估模式
        eval_preds = []     # 初始化预测结果列表
        with TorchTracemalloc() as tracemalloc:   # 启动内存跟踪器
            # 遍历评估数据加载器中的批次
            for _, batch in enumerate(tqdm(eval_dataloader)):
                # 移除批次数据中的标签,因为这是生成任务
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():    # 不计算梯度,加速推理
                    # 使用 generate 方法生成文本
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                    )  # synced_gpus=True for DS-stage 3
                # 在分布式环境下,需要对输出进行填充以保持一致
                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                preds = accelerator.gather_for_metrics(outputs)       # 收集预测结果

                # 对预测结果进行解码,并移除特殊标记
                preds = preds[:, max_length:].detach().cpu().numpy()
                eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        # 打印当前 epoch 下的 GPU 和 CPU 内存使用情况
        accelerator.print(f"GPU Memory before entering the eval : {b2mb(tracemalloc.begin)}")
        accelerator.print(f"GPU Memory consumed at the end of the eval (end-begin): {tracemalloc.used}")
        accelerator.print(f"GPU Peak Memory consumed during the eval (max-begin): {tracemalloc.peaked}")
        accelerator.print(
            f"GPU Total Peak Memory consumed during the eval (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
        )

        accelerator.print(f"CPU Memory before entering the eval : {b2mb(tracemalloc.cpu_begin)}")
        accelerator.print(f"CPU Memory consumed at the end of the eval (end-begin): {tracemalloc.cpu_used}")
        accelerator.print(f"CPU Peak Memory consumed during the eval (max-begin): {tracemalloc.cpu_peaked}")
        accelerator.print(
            f"CPU Total Peak Memory consumed during the eval (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}"
        )

        # 计算准确率
        correct = 0
        total = 0
        assert len(eval_preds) == len(
            dataset["train"][label_column]
        ), f"{len(eval_preds)} != {len(dataset['train'][label_column])}"
        for pred, true in zip(eval_preds, dataset["train"][label_column]):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        accelerator.print(f"{accuracy=}")
        accelerator.print(f"{eval_preds[:10]=}")
        accelerator.print(f"{dataset['train'][label_column][:10]=}")


    # 如果 do_test 为 True,则在测试集上进行评估
    if do_test:
        model.eval()
        test_preds = []
        for _, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                )  # synced_gpus=True for DS-stage 3
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather(outputs)
            preds = preds[:, max_length:].detach().cpu().numpy()
            test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        # 使用 get_closest_label 函数找到与预测结果最相近的类别标签
        test_preds_cleaned = []
        for _, pred in enumerate(test_preds):
            test_preds_cleaned.append(get_closest_label(pred, classes))

        test_df = dataset["test"].to_pandas()
        assert len(test_preds_cleaned) == len(test_df), f"{len(test_preds_cleaned)} != {len(test_df)}"
        test_df[label_column] = test_preds_cleaned

        # 将预测结果和原始预测结果添加到测试集数据框中
        test_df["text_labels_orig"] = test_preds
        accelerator.print(test_df[[text_column, label_column]].sample(20))   # 打印部分预测结果,用于检查

        # 创建提交文件
        pred_df = test_df[["ID", label_column]]
        pred_df.columns = ["ID", "Label"]
        os.makedirs(f"data/{dataset_name}", exist_ok=True)   # 创建存储目录
        pred_df.to_csv(f"data/{dataset_name}/predictions.csv", index=False)  # 将预测结果保存为 CSV 文件

    accelerator.wait_for_everyone()   # 等待所有进程完成
    # Option1: Pushing the model to Hugging Face Hub
    # model.push_to_hub(
    #     f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_"),
    #     token = "hf_..."
    # )
    # token (`bool` or `str`, *optional*):
    #     `token` is to be used for HTTP Bearer authorization when accessing remote files. If `True`, will use the token generated
    #     when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
    #     is not specified.
    #     Or you can get your token from https://huggingface.co/settings/token
    # Option2: Saving the model locally
    # 选项2: 将模型保存在本地
    peft_model_id = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace(
        "/", "_"
    )
    # 保存模型
    model.save_pretrained(peft_model_id)
    # TODO 思考, 是否将 tokenizer 也 save_pretrained 
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
