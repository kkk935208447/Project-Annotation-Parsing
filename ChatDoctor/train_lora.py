# 导入模型所需的包
import os
import sys
from typing import List

import fire
import torch
import torch.distributed
import transformers
from datasets import load_dataset
from typing import List, Optional, Union

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
# 从peft框架中导入相关配置文件
from peft import (  # noqa: E402
    LoraConfig,
    # BottleneckConfig, # TODO 修改官方代码
    get_peft_model,
    get_peft_model_state_dict,
    # TODO 修改官方代码
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
# 导入加载LlaMA模型所需的库
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    set_seed
)  # noqa: F402

# TODO 新增代码
import os 
set_seed(42)
os.environ["WANDB_DISABLED"] = "true" # 关闭 wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用并行性：如果您想禁用并行处理以避免死锁，可以将TOKENIZERS_PARALLELISM设置为false。


# TODO 新增代码, 打印模型的是否参与训练的参数名和数据类型
def print_model_allarguments_name_dtype(model):
    for n,v in model.named_parameters():
        if v.requires_grad:
            print(f"trainable model arguments: {n} - {v.dtype} - {v.shape}")
        else:
            print(f"not trainable model arguments: {n} - {v.dtype} - {v.shape}")


def train(
        # model/data params
        base_model: str = "",  # 必需的模型参数
        data_path: str = "yahma/alpaca-cleaned",  # 数据路径
        output_dir: str = "./lora-alpaca",  # 输出模型的目录
        adapter_name: str = "lora",  # 选择使用LoRA或Bottleneck Adapter进行微调
        # training hyperparams
        batch_size: int = 128,  # 批处理大小
        micro_batch_size: int = 4,  # 微批处理大小（可以根据batch_size与micro_batch_size的比值得到gradient_accumulation_steps）
        num_epochs: int = 3,  # 训练轮数
        learning_rate: float = 3e-4,  # 学习率
        cutoff_len: int = 256,  # 截断输入文本的最大长度
        val_set_size: int = 2000,  # 验证集大小
        use_gradient_checkpointing: bool = False,  # 是否使用梯度检查点(一种用时间换显存的方式)
        eval_step: int = 200,  # 每多少步进行一次验证
        save_step: int = 200,  # 每多少步保存一次模型
        # lora hyperparams
        lora_r: int = 8,  # LoRA模型的R参数，矩阵的秩
        lora_alpha: int = 16,  # LoRA模型的alpha参数
        lora_dropout: float = 0.05,  # LoRA模型的dropout率
        lora_target_modules: List[str] = None,  # LoRA模型的目标模块列表
        # bottleneck adapter hyperparams
        # TODO 修改代码, 删除此适配器
        bottleneck_size: int = 256,  # Bottleneck适配器的大小
        non_linearity: str = "tanh",  # 非线性激活函数
        adapter_dropout: float = 0.0,  # 适配器的dropout率
        use_parallel_adapter: bool = False,  # 是否使用并行适配器
        use_adapterp: bool = False,  # 是否使用适配器
        target_modules: List[str] = None,  # 适配器的目标模块列表
        scaling: Union[float, str] = 1.0,  # 缩放参数
        # llm hyperparams
        train_on_inputs: bool = True,  # 是否训练输入
        group_by_length: bool = False,  # 是否按长度分组
        # wandb params
        wandb_project: str = "",  # WandB项目名称
        wandb_run_name: str = "",  # WandB运行名称
        wandb_watch: str = "",  # WandB监控选项
        wandb_log_model: str = "",  # 是否记录模型到WandB
        resume_from_checkpoint: str = None,  # 从检查点恢复训练
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/LLaMA-7b-hf'"
    # 设置梯度累积的步数
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ, 检查是否设置了 Weights & Biases (wandb) 相关的环境变量或参数
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed, 如果在命令行中传入了 wandb 参数,则将其设置到环境变量中
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True, # 模型静态量化加载, 节约内容占用, 后面在训练阶段还会对模型进行再处理，没有被量化的层加载时用 fp16, 需要注意的是, 这些没有被量化层仍然会参与训练, 因此需要用prepare_model_for_kbit_training 进一步将这些层的 requires_grad 设置为 False 
        torch_dtype=torch.float16,  
        device_map=device_map,
    )

    # TODO 修改官方代码
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # 设置 tokenizer 的 pad_token_id 为 0 (unknown token),以便与 eos_token_id 区分
    tokenizer.pad_token_id = (0)
    # only-decoder 的LLM会普遍采用left padding，为了输入和输出的连续性。
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):  # 添加 EOS
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,  # None 表示返回 python 对象
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)  # 加入 EOS
            result["attention_mask"].append(1)                 # 加入 attention mask

        result["labels"] = result["input_ids"].copy()

        # TODO # 返回input_ids  labels attention_mask
            # 思考: 是否返回 position_ids, 因为tokenizer为left填充, position id 将用于计算RoPE, 格式是否是: 0,0,0,0,1,2,3,4,...
        return result  

     # 定义一个 generate_and_tokenize_prompt 函数,用于生成完整的训练/验证样本
    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt) 
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False) 
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            # 如果不训练输入部分,则将输入部分的 labels 设置为 -100 (忽略),只预测输出部分
            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    # TODO 新增代码, 打印模型参数
    print("---> model load_in_8bit or load_in_4bit 后那些没有被量化的层仍会参与训练, 具体见:")
    print_model_allarguments_name_dtype(model)

    # TODO 前面已经使用了 int8 量化, prepare_model_for_kbit_training的作用？
    """
    load_in_8bit=True是在从预训练模型加载时使用的,它将模型的参数量化成8位整数,以减少内存占用。这是一种静态量化的方法。
    需要注意的是: 静态量化中没有被量化的层仍然会参与训练(即requires_grad 为 True), 故需要进一步将这些层的 requires_grad 设置为 False 
    prepare_model_for_kbit_training 将一些未量化的层,如 lm_head, position_embedding 等层的 requires_grad 设置为 False, 并将这些层转化为32位 float, 再开启梯度检查点等一系列操作
    """
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    # TODO 新增代码, 打印模型参数
    print("---> prepare_model_for_kbit_training 后那些没有被量化的层也不会参与训练, 具体见:")
    print_model_allarguments_name_dtype(model)

    # 此处提供了两种微调方法，一种是lora，一种是bottleneck
    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    # TODO 修改注释官方代码
    # elif adapter_name == "bottleneck":
    #     config = BottleneckConfig(
    #         bottleneck_size=bottleneck_size,
    #         non_linearity=non_linearity,
    #         adapter_dropout=adapter_dropout,
    #         use_parallel_adapter=use_parallel_adapter,
    #         use_adapterp=use_adapterp,
    #         target_modules=target_modules,
    #         scaling=scaling,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    model = get_peft_model(model, config)

    # TODO 新增代码, 打印模型参数
    print("---> LORA model arguments, 具体见:")
    print_model_allarguments_name_dtype(model)
    print(f"---> model:\n{model}")

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint: # 如果指定了从检查点恢复训练,则加载检查点权重
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # 根据是否有验证集,创建训练集和验证集
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    print(f"one sample of train datas:\n{train_data[0]}")
    # 如果不是分布式训练环境,且有多个GPU,则设置模型为可并行化
    # TODO 思考, 是否存在 ddp 为False, torch.cuda.device_count() 这种情况?
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,          # 使用 bf16 混合精度训练
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",  # 存在验证集进行验证
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,  # 存在验证集训练结束时加载最优模型参数
            ddp_find_unused_parameters=False if ddp else None,    # 如果是分布式, 关闭参数检查
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    # 禁用模型的使用缓存功能,以节省显存
    model.config.use_cache = False

    # 替换模型的 state_dict 方法
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    # 在支持的平台上使用 torch.compile 加速模型推理
    # TODO peft 与 torch.compile 似乎不匹配, 是否注释
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)  # 保存训练好的模型

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


# 获得prompt
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)
