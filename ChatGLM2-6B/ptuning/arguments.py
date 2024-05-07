from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # 2. model_name_or_path 参数用于指定预训练模型的路径或标识符,从 Hugging Face 的模型库中下载。
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # 3. ptuning_checkpoint 参数用于指定 p-tuning v2 检查点的路径,p-tuning v2 是一种用于微调预训练语言模型的技术。
    ptuning_checkpoint: str = field(
        default=None, metadata={"help": "Path to p-tuning v2 checkpoints"}
    )
    # 4. config_name 参数用于指定预训练模型配置的名称或路径,如果与 model_name 不同的话。
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # 6. cache_dir 参数用于指定预训练模型缓存的路径,以避免重复下载。
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    # 7. use_fast_tokenizer 参数用于指定是否使用由 tokenizers 库提供支持的快速分词器。
    #    快速分词器通常能提高文本处理的速度。
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    # 8. model_revision 参数用于指定要使用的特定模型版本,可以是分支名、标签名或提交 ID。
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    # 9. use_auth_token 参数用于指示是否使用通过 `huggingface-cli login` 生成的认证令牌,
    #    这在使用私有模型时可能是必需的。
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    # 10. resize_position_embeddings 参数用于指示是否自动调整位置嵌入的大小,
    #     如果输入序列长度超过模型的位置嵌入长度
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    # 11. quantization_bit 参数用于指定模型量化的位宽。
    #     模型量化是一种压缩模型大小和提高推理速度的技术,
    #     它将模型的浮点数参数转换为定点数表示。
    quantization_bit: Optional[int] = field(
        default=None
    )
    # 12. pre_seq_len 参数用于指定 p-tuning v2 技术中的前缀长度。
    #     p-tuning v2 是一种用于微调预训练语言模型的技术,
    #     它通过在模型输入前添加一个可学习的前缀来实现微调。
    pre_seq_len: Optional[int] = field(
        default=None
    )
    # 指示是否使用前缀投影技术,这是 p-tuning v2 技术的一部分。
    prefix_projection: bool = field(
        default=False
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    1. 定义一个名为 DataTrainingArguments 的数据类,用于存储与数据相关的命令行参数。
       这些参数主要用于配置输入数据和数据预处理相关的选项。
    """
    # 2. lang 参数用于指定数据集的语言 ID,主要针对多语言摘要任务。
    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})
    # 3. dataset_name 参数用于指定要使用的数据集名称,这里会通过 datasets 库来加载数据集。
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    # 4. dataset_config_name 参数用于指定要使用的数据集配置名称,这也是通过 datasets 库来加载数据集的一部分。
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    # 5. prompt_column 参数用于指定数据集中包含完整文本的列名,主要针对摘要任务。
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    # 6. response_column 参数用于指定数据集中包含摘要的列名,主要针对摘要任务。
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    # 7. history_column 参数用于指定数据集中包含聊天历史的列名。
    history_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the history of chat."},
    )
    # 8. train_file 参数用于指定训练数据文件的路径,文件格式可以是 jsonlines 或 csv。
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    # 9. validation_file 参数用于指定验证数据文件的路径,文件格式可以是 jsonlines 或 csv。
    #    这个文件用于在训练过程中评估性能指标(如 ROUGE 评分)。
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    # 10. test_file 参数用于指定测试数据文件的路径,文件格式可以是 jsonlines 或 csv。
    #     这个文件用于在训练完成后评估最终的性能指标(如 ROUGE 评分)。
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    # 11. overwrite_cache 参数用于指示是否覆盖缓存的训练和验证数据集。
    #     如果设置为 True,则每次运行时都会重新生成缓存。
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    # 12. preprocessing_num_workers 参数用于指定用于数据预处理的并行进程数。
    #     这可以加速数据准备的过程。
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # 13. max_source_length 参数用于指定输入序列的最大长度,超过此长度的序列将被截断,短于此长度的序列将被填充。
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    # 14. max_target_length 参数用于指定目标文本的最大长度,超过此长度的序列将被截断,短于此长度的序列将被填充。
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    # 15. val_max_target_length 参数用于指定验证目标文本的最大长度,超过此长度的序列将被截断,短于此长度的序列将被填充。
    #     如果未指定,则默认使用 max_target_length 的值。
    #     此参数还用于覆盖 model.generate 方法中的 max_length 参数,该参数在 evaluate 和 predict 期间使用。
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    # 16. pad_to_max_length 参数用于指示是否将所有样本填充到模型的最大句子长度。
    #     如果设置为 False,则在批处理时动态填充到批次中的最大长度,这在 GPU 上更高效,但在 TPU 上表现较差。
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    # 17. max_train_samples 参数用于指定训练数据中最大样本数,可用于调试或更快的训练。
    #     如果设置了该参数,则训练集将被截断到指定的样本数。
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    # 18. max_eval_samples 参数用于指定验证数据中最大样本数,可用于调试或更快的训练。
    #     如果设置了该参数,则验证集将被截断到指定的样本数。
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    # 19. max_predict_samples 参数用于指定预测数据中最大样本数,可用于调试或更快的预测。
    #     如果设置了该参数,则预测集将被截断到指定的样本数。
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    # 20. num_beams 参数用于指定评估时使用的 beam 数量。
    #     该参数将传递给 model.generate 方法,该方法在 evaluate 和 predict 期间使用。
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    # 21. ignore_pad_token_for_loss 参数用于指示是否在损失计算中忽略与填充标签对应的令牌。
    #     这对于处理被填充的序列很有用。
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    # 22. source_prefix 参数用于在每个源文本前添加一个前缀,这在使用 T5 模型时很有用。
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    # 23. forced_bos_token 参数用于指定强制作为第一个生成令牌的令牌,
    #     这在使用多语言模型(如 mBART)时很有用,因为第一个生成的令牌需要是目标语言的令牌。
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )


    # 这部分代码是 DataTrainingArguments 类的 __post_init__ 方法,在对象初始化后执行一些检查和默认值设置。
    def __post_init__(self):
        # 24. 在对象初始化后,检查是否提供了数据集名称或训练/验证/测试文件。
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation/test file.")
        else:
            # 25. 如果提供了训练文件,检查其扩展名是否为 csv 或 json。
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."

            # 26. 如果提供了验证文件,检查其扩展名是否为 csv 或 json。
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        
        # 27. 如果未指定 val_max_target_length,则默认使用 max_target_length 的值。
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

