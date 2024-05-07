# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("/workspace/chatglm2-6b", trust_remote_code=True)
# print(tokenizer)
# model = AutoModel.from_pretrained("/workspace/chatglm2-6b", trust_remote_code=True).half().cuda()
# print(model)
# model = model.eval()
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)

# response, history = model.chat(tokenizer,"晚上睡不着应该怎么办", history=history)
# print(response)

from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    ptuning_checkpoint: str = field(
        default=None, metadata={"help": "Path to p-tuning v2 checkpoints"}
    )
    reuse: bool = field(
        default=False, metadata={"help": "Path to p-tuning v2 checkpoints"}
    )

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments))

    model_args = parser.parse_args_into_dataclasses()
    print(model_args)
    