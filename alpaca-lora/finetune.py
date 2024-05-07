import os
import sys
from typing import List
# fire是一个可以将Python组件自动转换为命令行界面的库
import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
# 从peft库中导入以下函数和类:
# LoraConfig: 用于配置LoRA(Low-Rank Adaptation)参数
# get_peft_model: 获取PEFT(Parameter-Efficient Fine-Tuning)模型
# get_peft_model_state_dict: 获取PEFT模型的state_dict
# prepare_model_for_int8_training: 为int8精度训练准备模型
# set_peft_model_state_dict: 设置PEFT模型的state_dict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, AutoTokenizer
# 从本地utils.prompter模块导入Prompter类,用于生成提示(prompt)
from utils.prompter import Prompter


def train(
    # model/data params
    base_model: str = "",  # the only required argument, 唯一必需的参数,指定基础模型路径
    data_path: str = "yahma/alpaca-cleaned",  # 数据集路径,默认为"yahma/alpaca-cleaned"
    output_dir: str = "./lora-alpaca",       # 输出目录,默认为"./lora-alpaca"
    # training hyperparams
    batch_size: int = 128,                # 批次大小,默认为128
    micro_batch_size: int = 4,             # 微批次大小,默认为4
    num_epochs: int = 3,                  # 训练epoch数,默认为3
    learning_rate: float = 3e-4,          # 学习率,默认为3e-4
    cutoff_len: int = 256,                # 截断长度,默认为256
    val_set_size: int = 2000,             # 验证集大小,默认为2000
    # lora hyperparams
    lora_r: int = 8,                      # LoRA秩(rank),默认为8
    lora_alpha: int = 16,                 # LoRA的alpha值,默认为16
    lora_dropout: float = 0.05,           # LoRA的dropout率,默认为0.05
    # LoRA目标模块列表,默认为["q_proj", "v_proj"]
    lora_target_modules: List[str] = ["q_proj","v_proj"],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss, # 如果为False,则在损失函数中掩码输入
    add_eos_token: bool = False,    # 是否在输入中添加EOS token
    group_by_length: bool = False,  # faster, but produces an odd training loss curve, 是否按长度分组,更快但产生奇怪的训练损失曲线
    # wandb params
    wandb_project: str = "",       # Weights & Biases项目名称
    wandb_run_name: str = "",      # Weights & Biases运行名称
    wandb_watch: str = "",  # options: false | gradients | all, 监视选项: false | gradients | all
    wandb_log_model: str = "",  # options: false | true,  # 是否记录模型: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter, # 从检查点或最终适配器恢复训练
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.  # 使用的提示模板名称,默认为"alpaca"
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        # 只在主进程中打印训练参数
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    # 检查基础模型路径是否被指定,否则引发AssertionError
    assert (base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    # 计算梯度累积步数,用于分批训练以节省内存
    gradient_accumulation_steps = batch_size // micro_batch_size
    # 创建Prompter实例,用于生成提示(prompt)
    prompter = Prompter(prompt_template_name)

    device_map = "auto"  # 默认自动选择设备
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # 判断是否为分布式数据并行(DDP)模式
    ddp = world_size != 1
    if ddp:
        # 在DDP模式下,根据本地rank设置device_map
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        # 在DDP模式下,梯度累积步数除以世界大小
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    # 检查是否设置了wandb相关参数
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    # 如果wandb参数被传递,则覆盖环境变量
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True, # 模型静态量化加载, 节约内容占用, 后面在训练阶段还会对模型进行动态量化
        # 没有被量化的层加载时用 fp16, 需要注意的是, 这些没有被量化层仍然会参与训练, 因此需要用 prepare_model_for_kbit_training 进一步将这些层的 requires_grad 设置为 False 
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # 设置Tokenizer的填充标记(pad_token_id)和填充方向(padding_side)
    # 这是为了支持批处理推理
    tokenizer.pad_token_id = (0  # unk. we want this to be different from the eos token
    )
    # 左侧填充
    tokenizer.padding_side = "left"  # Allow batched inference,
    # 3. 定义tokenize函数,用于tokenize提示(prompt)
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        # 使用Tokenizer的__call__方法对提示进行tokenize
        # truncation=True表示截断超长序列
        # max_length指定最大长度
        # padding=False表示不进行填充
        # return_tensors=None返回Python列表而不是张量

        # llama tokenizer 默认是不添加eos的
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        # 如果最后一个token不是eos_token_id且长度小于cutoff_len,则添加eos_token_id
        # 也就是说,如果上述文章太长被截断以后是不添加 EOS 的
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        # 为tokenize结果添加labels字段,用于训练
        result["labels"] = result["input_ids"].copy()

        return result

    # 4. 定义generate_and_tokenize_prompt函数,用于生成并tokenize提示
    def generate_and_tokenize_prompt(data_point):
        # 使用Prompter生成完整的提示
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        # 对完整的提示进行tokenize
        tokenized_full_prompt = tokenize(full_prompt)
        # 如果train_on_inputs为False,则在损失函数中掩码输入部分
        if not train_on_inputs:
            # 生成用户提示(不包含输出)
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            # 对用户提示进行tokenize
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            # 如果添加了eos_token,则长度减1
            if add_eos_token:
                user_prompt_len -= 1
            # 将用户提示部分的label设置为-100(忽略损失)
            # 其余部分保留原始label
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # could be sped up, probably
        return tokenized_full_prompt

    # TODO 前面已经使用了 int8 量化, 为何这里又进行了一次量化
    """
    load_in_8bit=True是在从预训练模型加载时使用的,它将模型的参数量化成8位整数,以减少内存占用。这是一种静态量化的方法。
    需要注意的是: 静态量化中没有被量化的层仍然会参与训练(即requires_grad 为 True), 故需要用动态量化进一步将这些层的 requires_grad 设置为 False 
    prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)是一个额外的量化步骤,它使用动态量化的方法,在训练期间对模型参数进行实时量化。这样可以进一步优化模型的内存使用,并在训练过程中获得更好的性能。
    具体来说,动态量化可以根据每个batch的数据特征,对相应的权重参数进行实时的量化,从而获得更精确的量化效果。这种方法通常能够获得比静态量化更好的模型性能。
    所以这两步量化的目的是不同的:
        静态量化是为了减少加载预训练模型时的内存占用。
        动态量化是为了在训练过程中进一步优化内存使用和性能。
    """
    model = prepare_model_for_int8_training(model)

    # 6. 创建LoRA配置
    # LoraConfig类用于指定LoRA(Low-Rank Adaptation)的超参数
    config = LoraConfig(
        r=lora_r,   # LoRA秩(rank)
        lora_alpha=lora_alpha,    # LoRA的alpha值
        target_modules=lora_target_modules,   # LoRA目标模块列表
        lora_dropout=lora_dropout,   # LoRA的dropout率
        bias="none",    # 不对bias项应用LoRA
        task_type="CAUSAL_LM",   # 任务类型为因果语言模型
    )
    # 7. 获取PEFT(Parameter-Efficient Fine-Tuning)模型
    # get_peft_model函数根据基础模型和LoRA配置创建PEFT模型
    model = get_peft_model(model, config)

    # 8. 加载数据集
    # 根据data_path的文件类型,使用不同的方式加载数据集
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # 9. 如果指定了resume_from_checkpoint,则从检查点恢复训练, 同时取消Trainer检查点回复
    if resume_from_checkpoint:
        # Check the available weights and load them, 
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")  # Full checkpoint
        # 检查可用的权重文件
        if not os.path.exists(checkpoint_name):
            # 只LoRA模型的文件
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state, # 防止trainer试图加载状态
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        # 上述两个文件名存在不同,但实际上是同一个文件, 都是LORA的参数
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            # 将检查点权重加载到PEFT模型中
            set_peft_model_state_dict(model, adapters_weights)
            
        else:
            print(f"Checkpoint {checkpoint_name} not found")
            
    # 10. 打印可训练参数的比例,以提高透明度
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # 11. 根据val_set_size划分训练集和验证集
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

    # 12. 如果不是DDP模式且有多个GPU,则启用模型并行
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # 13. 创建Trainer实例
    trainer = transformers.Trainer(
        model=model,   # 传入要训练的模型
        train_dataset=train_data,   # 传入训练数据集
        eval_dataset=val_data,      # 传入评估数据集
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,      # 1. 每个设备的微批次大小,用于梯度积累
            gradient_accumulation_steps=gradient_accumulation_steps,   # 2. 梯度累积步数,可以模拟大批量训练来节省内存
            warmup_steps=100,    # 3. 学习率warmup步数,在初始几步内逐渐增加学习率,使训练更加稳定
            num_train_epochs=num_epochs,   # 4. 总的训练epoch数
            learning_rate=learning_rate,   # 5. 初始学习率
            fp16=True,    # 6. 启用fp16混合精度训练,可以节省一半的内存并提高计算速度
            logging_steps=10,   # 7. 每间隔10步记录一次训练日志
            optim="adamw_torch",   # 8. 使用AdamW优化器,是BERT等模型推荐的优化器
            evaluation_strategy="steps" if val_set_size > 0 else "no",   # 9. 如果有验证集,则每隔一定步数评估一次模型
            save_strategy="steps",   # 10. 按步数保存模型检查点
            eval_steps=200 if val_set_size > 0 else None,    # 11. 如果有验证集,则每200步评估一次
            save_steps=200,         # 12. 每200步保存一次模型
            output_dir=output_dir,   # 13. 模型输出目录
            save_total_limit=3,      # 14. 最多保存3个模型检查点
            load_best_model_at_end=True if val_set_size > 0 else False,   # 15. 如果有验证集,则在结束时加载最佳模型
            ddp_find_unused_parameters=False if ddp else None,    # 16. 在DDP(分布式数据并行)模式下忽略未使用的参数
            group_by_length=group_by_length,            # 17. 是否按序列长度分组,可加快训练但会产生奇怪的损失曲线
            report_to="wandb" if use_wandb else None,       # 18. 如果使用Weights & Biases,则向它报告训练过程
            run_name=wandb_run_name if use_wandb else None,     # 19. Weights & Biases运行名称
        ),
        # 20. 数据准备函数
        data_collator=transformers.DataCollatorForSeq2Seq(
                                                        tokenizer,   # 使用之前加载的tokenizer
                                                        pad_to_multiple_of=8,   # 将序列长度填充到8的倍数
                                                        return_tensors="pt",    # 返回PyTorch张量
                                                        padding=True            # 启用填充
        ),
    )
    # 14. 禁用模型缓存,以减少内存使用
    # 在序列生成任务中使用缓存可以加快推理速度,但会占用额外内存
    # 由于训练时不需要推理,因此禁用缓存可以减少内存使用
    model.config.use_cache = False

    # # 15. 修改模型的state_dict方法,使用PEFT的state_dict
    # # state_dict存储了模型的权重和参数
    # # 由于使用了PEFT进行高效微调,需要使用PEFT提供的state_dict方法
    # # TODO: 删除原代码, 该代码是为了将原模型的state_dict替换成仅仅包含lora的权重，新版本peft会自动仅保存lora权重，不会保存原本模型的权重
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))
    # # 16. 如果PyTorch版本 >= 2且不在Windows平台上,则编译模型
    # # PyTorch 2.0引入了编译机制,可以提高模型推理和训练的性能
    # # 但在Windows上目前还不支持,因此添加了平台判断
    # # TODO: 删除原代码,torch.compile 与 peft（0.9.0版本）目前似乎不兼容，开启此代码会导致lora权重文件保存的是空字典，推理时加载lora权重会报错
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    # 17. 开始训练
    # resume_from_checkpoint 此时 == False, 因为之前已经重加载过了
    with torch.autocast("cuda"): # 精度自动转换
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 18. 保存最终模型
    # 将微调后的模型权重保存到output_dir目录下
    model.save_pretrained(output_dir,safe_serialization=False)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    """
    这一行代码是 Python 的常见用法,它利用了 fire 库提供的命令行接口工具。
    if __name__ == "__main__": 是一种常见的 Python 习惯用法,它检查当前脚本是作为主程序运行还是被导入为模块。如果是作为主程序运行,则执行下面的代码块。这种方式可以确保脚本既可以被直接运行,也可以作为模块导入。
    fire.Fire(train) 是 fire 库的关键部分。fire 是一个 Python 库,它可以自动将任何 Python 组件(如函数、类、对象等)转换为命令行界面。
    在这里,fire.Fire(train) 的作用是将前面定义的 train 函数转换为一个命令行接口。这样,我们就可以在命令行中直接运行 train 函数,并通过命令行参数来传递不同的配置。
    例如,如果我们在命令行中运行:
    python script.py --base_model=huggyllama/llama-7b --data_path=data.json
    那么 fire 会自动将命令行参数解析并传递给 train 函数,相当于调用:
    train(base_model="huggyllama/llama-7b", data_path="data.json")
    使用 fire 库的优点是:
        简化了命令行参数的解析和处理过程。
        无需手动编写命令行解析代码,可以专注于函数的实现。
        方便地将任何 Python 组件公开为命令行工具,提高了代码的可用性和可维护性。
        自动生成帮助文档和命令行提示,提高了用户体验。
        总的来说,fire.Fire(train) 这一行代码将 train 函数转换为一个命令行工具,使得我们可以通过命令行参数来控制和运行该函数,而无需编写额外的命令行解析代码。
    """
    fire.Fire(train)
