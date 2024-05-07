from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LongQLoRAArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})  # max_seq_length <= model_max_length
    model_max_length: int = field(metadata={"help": "模型位置编码扩展为该长度"})
    # TODO ++ 新增参数, max_prompt_length + max_response_length == max_seq_length, 里面包含 <s> 和 </s> 等这些特殊符号
    max_prompt_length: int = field(metadata={"help": "prompt最大长度"})
        # 1.5k 的数据中, 统计 output 的tokenizer长度分位数 0.99 = 637, 故默认值选择 637
    max_response_length: int = field(metadata={"help": "response最大长度,"})

    train_file: str = field(metadata={"help": "训练数据路径"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    sft: bool = field(metadata={"help": "True为sft，False则进行自回归训练"})

    target_modules: str = field(default=None, metadata={
        "help": "QLoRA插入adapter的位置，以英文逗号分隔。如果为None，则在自动搜索所有linear，并插入adapter"
    })
    eval_file: str = field(default=None, metadata={"help": "评测集路径"})
    # TODO ++ 新增参数, 是否启用 s2attention
    enable_s2attention: bool = field(default=True, metadata={"help": "训练时是否使用s2attention"})
    use_flash_attn: bool = field(default=False, metadata={"help": "训练时是否使用flash attention"})
    
    train_embedding: bool = field(default=False, metadata={"help": "词表权重是否参与训练"})
    train_norm: bool = field(default=False, metadata={"help": "norm权重是否参与训练"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})

    # TODO ++ 新增参数, 梯度检查点/量化相关
    use_reentrant: Optional[bool] = field(
        default=True,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    # 即 双量化
    use_nested_quant: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    # 4位量化模型的计算数据类型,例如float16或bfloat16, 使用较低的计算精度可以提高计算速度,但可能会影响模型精度
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    # 4位量化模型的量化存储数据类型,如uint8或float16或bfloat16, 使用较低的存储精度可以减小模型大小,但可能会影响模型精度
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    # 4位量化类型
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )

    # TODO ++ 新增代码
    def __post_init__(self):
        if self.max_seq_length > self.model_max_length:  # max_seq_length <= model_max_length
            raise ValueError("ArgumentError: max_seq_length > model_max_length")
        if self.max_prompt_length + self.max_response_length > self.max_seq_length: # max_prompt_length + max_response_length <= max_seq_length
            raise ValueError("ArgumentError: max_prompt_length + max_response_length > max_seq_length") 

