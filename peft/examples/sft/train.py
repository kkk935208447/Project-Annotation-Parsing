import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional

import torch.distributed
from transformers import HfArgumentParser, TrainingArguments, set_seed, Seq2SeqTrainingArguments
from trl import SFTTrainer    # SFTTrainer用于序列到序列(Sequence-to-Sequence)的语言模型微调训练
from utils import create_and_prepare_model, create_datasets  # 自定义的实用函数,用于创建和准备模型、数据集

# TODO 新增代码, wandb 与 bash 重定向 log.out 冲突, 关闭掉
os.environ["WANDB_DISABLED"] = "true" # 关闭 wandb

# Define and parse arguments. 定义ModelArguments数据类,用于指定模型相关参数
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # 指定预训练语言模型的路径或在Hugging Face模型库中的标识符, 这允许您使用您选择的任何预训练模型,如GPT-2、GPT-3、BERT等
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # 指定聊天数据的格式,有以下选项:
    # 1) chatml: 使用Anthropic的chatml格式,例如: <human>: 你好 \n<assistant>: 你好,很高兴与你交谈。
    # 2) zephyr: 使用Pretrained.AI的zephyr格式,例如: Human: 你好 \nAssistant: 你好,很高兴与你交谈。 
    # 3) none: 如果数据集已经格式化为聊天模板,则设置为none
    # 这个参数可以帮助您灵活地处理不同格式的聊天数据
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)    # lora_alpha控制LoRA层的重要性,典型值为16或32
    lora_dropout: Optional[float] = field(default=0.1)  # lora_dropout设置LoRA层的dropout率,用于防止过拟合
    # lora_r指定LoRA低秩矩阵的秩(rank),较低的秩可以进一步减少参数量,但可能会影响性能, 秩越低,模型越压缩,但可能会导致性能下降
    lora_r: Optional[int] = field(default=64)
    # lora_target_modules指定应用LoRA的模块列表
    # 默认值包括注意力层的线性投影(q_proj, k_proj, v_proj, o_proj)和前馈神经网络层(down_proj, up_proj, gate_proj)
    # 也可以设置为"all-linear"以应用LoRA到所有线性层
    # 通过选择性地应用LoRA,可以在性能和参数量之间进行权衡
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    # use_nested_quant指定是否启用嵌套量化(nested quantization), 嵌套量化可以将4位量化模型进一步量化为2位或更低,从而进一步减小模型大小和内存占用,但可能会影响精度
    # 即 双量化
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    # bnb_4bit_compute_dtype指定4位量化模型的计算数据类型,例如float16或bfloat16, 使用较低的计算精度可以提高计算速度,但可能会影响模型精度
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    # bnb_4bit_quant_storage_dtype指定4位量化模型的量化存储数据类型,如uint8或float16或bf16, 使用较低的存储精度可以减小模型大小,但可能会影响模型精度
    # 您需要权衡模型大小和精度的平衡
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    # bnb_4bit_quant_type指定4位量化类型,包括fp4(浮点4位量化)或nf4(整数4位量化)
    # 不同的量化类型会影响模型精度和计算效率的权衡
    # fp4可能会保留更多精度,但nf4可能会更快
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    # use_flash_attn指定是否启用Flash注意力(Flash attention)
    # Flash注意力是一种高效的注意力实现,可以通过内存优化和并行计算提高训练速度,但可能会增加一些开销
    # 这个参数可以帮助您在训练速度和内存占用之间进行权衡
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    # use_peft_lora指定是否启用PEFT (Parameter-Efficient Fine-Tuning) LoRA
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    # use_8bit_quantization指定是否将模型加载为8位量化版本
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    # use_4bit_quantization指定是否将模型加载为4位量化版本, 4位量化可以将模型大小减小到原始大小的1/4,从而进一步节省内存和加快计算,但可能会显著影响精度
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    # use_reentrant是梯度检查点(Gradient Checkpointing)的一个参数, 梯度检查点可以通过重新计算激活值来节省内存,但会增加一些计算开销
    # use_reentrant指定是否使用可重入(reentrant)的梯度检查点实现,可能会进一步节省内存, 这个参数可以帮助您在内存占用和计算开销之间进行权衡
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    # use_unsloth指定是否使用Unsloth库进行训练
    # Unsloth是一个优化库,可以通过内存优化和并行计算加速PEFT LoRA的训练过程
    # 这个参数可以帮助您进一步提高训练效率
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )


# 定义DataTrainingArguments数据类,用于指定数据集和数据处理相关参数
@dataclass
class DataTrainingArguments:
    # 指定要使用的数据集名称或路径,默认为OpenAssistant Guanaco数据集
    # 您可以根据需要替换为其他数据集
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )

    # packing指定是否使用数据集打包(packing)
    # 数据集打包可以将多个样本打包为一个更长的序列,从而提高训练效率, 这个参数可以帮助您在训练速度和内存占用之间进行权衡
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )

    # dataset_text_field指定数据集中作为input文本的字段名, 这个参数可以帮助您灵活地处理不同格式的数据集
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    # max_seq_length指定输入序列的最大长度,超出部分将被截断, 这个参数可以帮助您在训练速度、内存占用和模型性能之间进行权衡
    max_seq_length: Optional[int] = field(default=512)

    # append_concat_token指定在打包数据集时,是否在每个样本的末尾追加一个连接标记(如<eos>),这个参数可以帮助您控制数据集的格式,从而影响模型的输出
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )

    # add_special_tokens指定在打包数据集时,是否由分词器(tokenizer)添加特殊标记(如<bos>和<eos>), 这个参数可以帮助您控制数据集的格式,从而影响模型的输出
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    # splits指定要从数据集中使用的数据分割,如train、test或val,多个分割用逗号分隔, 这个参数可以帮助您灵活地使用数据集的不同部分进行训练和评估
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )

# TODO 新增代码, 打印模型的是否参与训练的参数名和数据类型
def print_model_allarguments_name_dtype(model):
    for n,v in model.named_parameters():
        if v.requires_grad:
            print(f"trainable model arguments: {n} - {v.dtype} - {v.shape} - {v.device}")
        else:
            print(f"not trainable model arguments: {n} - {v.dtype} - {v.shape} - {v.device}")


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed) # 设置随机种子,以确保实验可重复性

    # model ,调用create_and_prepare_model函数,根据参数创建并准备模型、PEFT配置和分词器
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

    # gradient ckpt
    # 配置是否使用模型缓存和梯度检查点, 模型缓存可以加速注意力计算,但会占用更多内存, 梯度检查点可以节省内存,但会增加一些计算开销
    # 如果使用了Unsloth,则不需要梯度检查点
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    # datasets
    # 调用create_datasets函数,根据参数创建训练集和评估集, apply_chat_template指定是否对数据集应用聊天模板格式
    train_dataset, eval_dataset = create_datasets(
        tokenizer,
        data_args,
        training_args,
        apply_chat_template=model_args.chat_template_format != "none",
    )

    # TODO 新增代码, 用于方便调试, 判断是否使用分布式
    if (torch.distributed.is_available() and torch.distributed.is_initialized()):
        torch.distributed.barrier()  # 进程阻塞同步

    # trainer
    # 创建SFTTrainer对象,用于语言模型的序列到序列(Sequence-to-Sequence)微调训练, 传入模型、分词器、训练参数、训练集、评估集、PEFT配置等参数
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=data_args.packing,
        dataset_kwargs={
            "append_concat_token": data_args.append_concat_token,
            "add_special_tokens": data_args.add_special_tokens,
        },
        dataset_text_field=data_args.dataset_text_field,
        # 定义模型最大的长度
        max_seq_length=data_args.max_seq_length,    
    )
    # TODO 新增代码, 仅主进程打印模型信息和可训练参数, 这可以帮助您了解模型的结构和参数
    # SFTTrainer 会把原模型的量化层参数和非量化层参数都冻结, PEFT 的 get_peft_model 仅仅会将量化层冻结
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        print("---> model layers")
        print_model_allarguments_name_dtype(model = trainer.model)     # 这里使用 trainer.model
        print(f"---> Training/evaluation parameters:\n{training_args}")
        print(f"---> Model parameters:\n{model_args}")
        print(f"---> Datas parameters:\n{data_args}")
        print(f"---> model config:\n{trainer.model.config}")
        print(f"---> PEFT config:\n{peft_config}")

    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()
    
    # train
    # 如果指定了checkpoint路径,则从该checkpoint恢复训练
    # 这个功能可以让您在之前训练的基础上继续训练模型
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    # 如果使用FSDP (Fully Sharded Data Parallelism),则设置状态字典类型
    # FSDP是一种分布式并行训练技术,可以帮助您利用多个GPU进行训练
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # 保存训练好的模型
    # 这将把训练好的模型保存到磁盘,以便后续使用
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # TODO 新增代码, 用于方便调试, 判断是否使用分布式
    if (torch.distributed.is_available() and torch.distributed.is_initialized()):
        print(f"---> Torch distributed enable, Torch distributed initialized, This local rank is: {training_args.local_rank}, Word_size: {torch.torch.distributed.get_world_size()}")

    main(model_args, data_args, training_args)
