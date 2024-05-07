# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator    # 加速库
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """
    
    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # NOTE: gpt2 模型使用了 Conv1D 而不是 Linear 层,目前还不支持 8 位模式
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    #  PPO 训练的迭代次数
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})  
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    # 是否使用 Adafactor 优化器
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    # 提前停止的 KL 目标值, RLHF 训练过程 KL散度 的大小会慢慢变大
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    # 从奖励中减去的基线值
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    # 是否使用批量文本生成
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    # 保存模型的步数间隔
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    # 训练步数
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})

    # 初始 KL 惩罚系数 (用于自适应和线性控制)
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    # 是否使用自适应 KL 控制,否则使用线性控制
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    # 是否以 8 位模式加载模型
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 8bit"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name
dataset_name = "lvwerra/stack-exchange-paired"
config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)

train_dataset = load_dataset("lvwerra/stack-exchange-paired", data_dir="data/rl", split="train")
train_dataset = train_dataset.select(range(100000))
original_columns = train_dataset.column_names

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
# 这是一段与奖励模型 (Reward Model, RM) 有关的代码, 定义传递给情感分析管道的参数
# 将 `return_all_scores` 设置为 True 以获取每个 token 的情感分数
sent_kwargs = {
    "return_all_scores": True,        # 返回每个 token 的情感分数
    "function_to_apply": "none",       # 不对情感分数应用任何函数
    "batch_size": 16,                # 批量大小
    "truncation": True,              # 是否截断过长的输入
}

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
# GPT-2 tokenizer 有一个 pad 标记,但默认情况下它不是 eos 标记。我们需要将其设置为 eos 标记
# 仅对于这个模型
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer,
    dataset_name="lvwerra/stack-exchange-paired",
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    num_proc = 24  # 用于预处理的进程数

    def preprocess_function(examples):
        new_examples = {
            "query": [],         # 存储问题
            "input_ids": [],     # 存储问题的 token ID
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "            # 构造问题的提示
            tokenized_question = tokenizer(query, truncation=True)      # 对问题进行分词和编码
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples
    # 对数据集应用预处理函数,生成新的数据集
    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    # 过滤掉长度超过 512 的样本
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")    # 将数据集格式设置为 PyTorch 张量
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer)

# 定义一个 collator 函数,用于对一批数据进行collate
def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 加载预训练模型并使用 LoRA 技术
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=script_args.load_in_8bit,
    device_map={"": current_device},  # 设备映射
    peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:  # 如果使用 Adafactor 优化器
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),   # 只优化需要梯度的参数
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
# 使用奖励模型构建情感分析管道,并设置相同的设备
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug, 避免 `pipeline` 的一个 bug
sentiment_pipe = pipeline(
    "sentiment-analysis",         # 任务类型为情感分析
    model=reward_model_name,       # 奖励模型名称
    device_map={"": current_device},   # 设备映射
    model_kwargs={"load_in_8bit": script_args.load_in_8bit}, # 是否以 8 位精度加载模型
    tokenizer=tokenizer,
    return_token_type_ids=False,    # 不返回 token 类型 ID
)

# 如果情感分析模型没有 pad_token_id,将其设置为 eos_token_id
if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
# 定义传递给 `generate` 函数的参数,这些参数将传递给 PPOTrainer 的 `generate` 函数
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32        # 输出的最小长度
output_max_length = script_args.output_max_length         # 输出的最大长度
output_length_sampler = LengthSampler(output_min_length, output_max_length)    # 长度采样器

# 开始训练循环
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch >= config.total_ppo_epochs:
        break

    question_tensors = batch["input_ids"]  # 获取问题的 token ID

    # 使用 PPOTrainer 的 generate 函数生成回答
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,    # 不返回提示
        length_sampler=output_length_sampler,  # 使用长度采样器
        **generation_kwargs,   # 生成参数
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)   # 解码回答

    # Compute reward score (using the sentiment analysis pipeline)
    # 使用情感分析管道计算奖励分数
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]  # 将问题和回答拼接
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)    # 进行情感分析
    # 计算奖励
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # 根据 save_freq 参数保存模型
    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
