from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoConfig,
    GenerationConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import argparse
import sys
import inspect
from loguru import logger
# 重定义终端logger显示颜色
logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} |<cyan><lvl>{level:8}</></>| {name} : {module}:{line:4} | <cyan>mymodule</> | - <lvl>{message}</>",
        "colorize": True
    },
])
import os
from os.path import join
import yaml
import torch
import bitsandbytes as bnb
import math
from collections import defaultdict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from component.collator import PretrainCollator, DataCollatorForSeq2Seq
from component.dataset import PretrainDataset, CustomDataset
from component.argument import LongQLoRAArguments
from component.trainer import LoRATrainer
from component.loss import CausalLMLoss
from trl.trainer.utils import peft_module_casting_to_bf16
from attention.llama_attn_replace import replace_llama_attn
from attention.s2attn_replaced_only_train import replace_model_s2attn_only_train


def verify_model_dtype(model):
    """
    功能: 查看模型中各种类型的参数的情况
    使用技术: Python 的 defaultdict、model.named_parameters()、参数遍历等
    解决问题: 帮助开发者了解模型中参数的数据类型分布,以及可训练参数的情况,从而更好地优化模型
    """
    # 查看model 结构
    logger.info(f"--> model structure: \n{model}")
    ignore_layers = [f"layers.{i}" for i in range(2,21)]  # 减少打印的层数
    logger.info(f"ignore print layers: \n{ignore_layers}")
    # 查看model layer 结构,dtype,size,device等
    for n,v in model.named_parameters():
        # 少打印一些层
        if not any([i in n for i in ignore_layers]):
            if v.requires_grad:
                logger.info(f"trainable model arguments: {n} - {v.dtype} - {v.shape} - {v.device}")
            else:
                logger.info(f"not trainable model arguments: {n} - {v.dtype} - {v.shape} - {v.device}")

    # 创建默认字典,用于存储不同数据类型的参数数量和参数名称
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype   # 获取参数的数据类型
        # 统计参数数量和参数名称
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        # 如果参数参与训练(requires_grad=True),则统计可训练参数的数量和名称
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    logger.info('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print("all params info: {}  num: {}  {:.3f}%".format(k, v, 100.0 * v / total))  # 打印每种数据类型的参数量和占比
    print()
    
    # 统计可训练参数中，各种类型参数分布
    logger.info('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print("trainable params info: {}  num: {}  {:.3f}%".format(k, v, 100.0 * v / total_trainable))
    print()
    for k, v in dtype2trainable_param_name.items():
        print("all params info: {}  trainable layers: {}".format(k, v))   # 打印每种数据类型的可训练参数名称
    print()
    # for k, v in dtype2trainable_param_num.items():
    #     print("trainable params info: {}  num: {}".format(k, v))
    # print()
    # 查看参与训练的参数情况
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total model params: %.2fM" % (total / 1e6))
    logger.info(
        f'trainable params: {trainable} || all params: {total} || trainable%: {round(trainable / total, 4)}')

# 不给定target_modules时，自动搜索所有linear
def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # "lmhead" 不使用lora
    if 'lm_head' in lora_module_names:  # needed for 16-bit,
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='./train_args/llama2-7b-pretrain.yaml', help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")  # 需要给定,否则可能使用 deespeed 调用 train.py 会报 local_rank 错误, 后续会被 TrainingArguments 覆盖
    args = parser.parse_args()
    train_args_file = args.train_args_file

    # 读取训练的参数配置
    parser = HfArgumentParser((LongQLoRAArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_yaml_file(yaml_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    logger.add(join(training_args.output_dir, 'train.log'))
    # 加载训练配置文件
    with open(train_args_file, "r") as f:
        train_args = yaml.safe_load(f)
    # 保存训练参数到输出目录
    with open(join(training_args.output_dir, 'train_args.yaml'), "w") as f:
        yaml.dump(train_args, f)
    # 设置随机种子
    set_seed(training_args.seed)
    training_args.train_embedding = args.train_embedding
    return args, training_args


def load_model_and_tokenizer(args, training_args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # config.use_cache = False
    # TODO ++ 新增代码, 梯度检查点
    config.use_cache = not training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": args.use_reentrant}

    model_type = config.model_type
    assert model_type == 'llama', "Only support llama and gpt-neox for now"
    # 修改LLaMA的attn是否使用flash s2attention  s2attention 等
    # TODO 新增代码,s2atten 替换新的实现, 控制模型加载使用 falsh attention 或 eager attention
    if args.use_flash_attn:
        attn_implementation = 'flash_attention_2'   
    else:
        # TODO 新增代码, attn_implementation 指定 eager 时, 需要额外给定 _attn_implementation_internal,
        # 详情见 https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py 中的函数 ._autoset_attn_implementation
        config._attn_implementation_internal = "eager"
        attn_implementation = "eager"
    # replace_llama_attn(args.use_flash_attn)
    replace_model_s2attn_only_train(use_flash_attn=args.use_flash_attn,model_type=config.model_type,enable_s2attention=args.enable_s2attention)  

    # 使用PI插值, 扩展RoPE的position
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    logger.info(f'Change model_max_length from {orig_ctx_len} to {args.model_max_length}')

    # 设置device_map，以适配多卡训练
    # local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    # device_map = {'': local_rank}
    # TODO 修改官方代码, 
    # device_map = "auto"  # "auto" 需要安装 accelerate

    # 参考代码: peft 官方例子, https://blog.csdn.net/qq_16555103/article/details/137677561
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        # 获取指定的量化存储数据类型, getattr 会将字符串 bfloat16 ---> torch.bfloat16
    quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)
    # 创建BitsAndBytesConfig对象,用于配置量化相关参数, 用于管理量化配置的类,可以指定量化类型、计算数据类型、存储数据类型等
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,          # 是否使用4位量化
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,     # 4位量化的类型, 如 nf4
        bnb_4bit_compute_dtype=compute_dtype,             # 计算数据类型
        bnb_4bit_use_double_quant=args.use_nested_quant,  # 是否使用双量化
        # TODO Qlora + zero3 修改的代码
        bnb_4bit_quant_storage=quant_storage_dtype,       # 量化存储数据类型
    )
    # 如果计算数据类型为float16且使用4位量化,则打印GPU是否支持bfloat16的提示
    if compute_dtype == torch.float16:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            logger.warning("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)
    # 加载模型
    logger.info(f'Loading model from: {args.model_name_or_path}')
    torch_dtype = (quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        # TODO Qlora + zero3 修改的代码
            # 注意 torch_dtype 对于 AutoModelForCausalLM 与 bnb_4bit_quant_storage 数据类型相同。就是这样。其他所有事情都由 Trainer 和 TRL 处理。
        torch_dtype=torch_dtype,  # 未量化的模型参数转换为 bfloat16 浮点类型,以减少内存占用
        trust_remote_code=True,   # 允许加载远程预训练模型文件
        quantization_config=bnb_config,
        # # TODO 新增代码, 新版的transformer需要指定一个attention 类型, 
        # # s2atten 修改的 flashattention_2 还是 eager
        attn_implementation = attn_implementation
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        # use_fast=True,
        use_fast=False if config.model_type == 'llama' else True
    )
    assert tokenizer.eos_token_id is not None
    assert tokenizer.bos_token_id is not None
    # 部分tokenizer的pad_token_id为None
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    return model, tokenizer

def insert_adapter(args, training_args, model):
    if training_args.local_rank <= 0:
        logger.info(f'--> args: {args}')
        logger.info(f'--> training_args: {training_args}')
    # 找到所有需要插入adapter的位置
    if args.target_modules is not None:
        target_modules = args.target_modules.split(',')
    else:
        target_modules = find_all_linear_names(model)

    # 初始化lora配置
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None  # (`List[str]`): 除了适配器层之外，要设置为可训练并保存在最终检查点中的模块列表。
        # modules_to_save=["embed_tokens", "lm_head"] if args.train_embedding else None
    )

    # TODO Qlora + zero1 2 3 代码
    # 参考 trl SFTTrainer 代码, https://blog.csdn.net/qq_16555103/article/details/137677561
    if not isinstance(model, PeftModel):
        _support_gc_kwargs = hasattr(
            training_args, "gradient_checkpointing_kwargs"
        ) and "gradient_checkpointing_kwargs" in list(
            inspect.signature(prepare_model_for_kbit_training).parameters
        )
        gradient_checkpointing_kwargs = getattr(training_args, "gradient_checkpointing_kwargs", None) or {}
        is_sharded_qlora = False
        # Below is to support QLoRA + FSDP / DS-Zero3 - one should never call
        # peft_module_casting_to_bf16 or prepare_model_for_kbit_training when doing
        # QLoRA + FSDP / DS-Zero3
        if getattr(model, "is_loaded_in_4bit", False):
            for _, param in model.named_parameters():
                if param.__class__.__name__ == "Params4bit":
                    # 判断 QLoRA + FSDP / DS-Zero3, is_sharded_qlora 为 True
                    is_sharded_qlora = param.data.device.type == "cpu"
                    break
        
        # 4bit 或 8bit 量化未开启 FSDP / DS-Zero3 时 模型准备与梯度检查点
        if getattr(model, "is_loaded_in_8bit", False) or (
            getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora
        ):
            prepare_model_kwargs = {
                "use_gradient_checkpointing": getattr(training_args, "gradient_checkpointing", False)
            }

            if _support_gc_kwargs:
                prepare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

            model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

            # if training_args is not None:
            #     training_args = dataclasses.replace(training_args, gradient_checkpointing=False)

        # 4bit 量化开启 FSDP / DS-Zero3 时模型准备与梯度检查点
        elif getattr(training_args, "gradient_checkpointing", False) and (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        ):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # lora
        model = get_peft_model(model, peft_config)
        # 4bit 量化使用 training_args.bf16 且非 FSDP / DS-Zero3 时 peft_module_casting_to_bf16
        if (
            training_args is not None
            and training_args.bf16
            and getattr(model, "is_loaded_in_4bit", False)
            and not is_sharded_qlora
        ):
            peft_module_casting_to_bf16(model)

    model.print_trainable_parameters()
    # # TODO 疑问: 这行代码有何作用???
    # model.config.torch_dtype = torch.float32

    # 根据配置，决定word embedding和norm是否参与训练
    for n, p in model.named_parameters():
        # 训练word embedding
        if args.train_embedding and ("embed_tokens" in n or "lm_head" in n):
            p.requires_grad = True
        # 训练norm
        if args.train_norm and "norm" in n:
            p.requires_grad = True

    if training_args.local_rank <= 0:
        # 查看模型种各种类型的参数的情况
        verify_model_dtype(model)
        logger.info(f"--> model config: {model.config}")
        logger.info(f'--> peft config: {peft_config}')

    return model


def merge_lora():
    pass


def init_components(args, training_args):
    """
    初始化各个组件
    """
    if training_args.local_rank <= 0:
        logger.info('Initializing components...')
    # 务必设为False，否则多卡训练会报错
    training_args.ddp_find_unused_parameters = False
    # 加载model和tokenizer
    model, tokenizer = load_model_and_tokenizer(args, training_args)
    # 插入adapter
    model = insert_adapter(args, training_args, model)
    # 初始化损失函数
    loss_func = CausalLMLoss(ignore_index=-100)

    # 加载训练集和验证集
    if args.sft:
        # train_dataset = VicunaSFTDataset(args.train_file, tokenizer, args.max_seq_length)
        # data_collator = SFTCollator(tokenizer, args.max_seq_length, -100)
        # TODO 修改代码
        dataset = CustomDataset(file = args.train_file, # jsonl 格式, keys: input, output
                                tokenizer = tokenizer, 
                                max_seq_length = args.max_seq_length,      # max_seq_length 为 prompt + response 最大长度
                                # max_prompt_length + max_response_length == max_seq_length, 里面包含 <s> 和 </s> 等这些特殊符号
                                max_prompt_length = args.max_prompt_length,  
                                max_response_length = args.max_response_length, 
                                ignore_index = -100,
                                num_proc = 10)  # 数据处理的进程数
        train_dataset = dataset.get_train_dataset()
        val_dataset = dataset.get_validation_dataset()
        data_collator = DataCollatorForSeq2Seq(tokenizer, 
                                               model=model, 
                                               label_pad_token_id=-100, 
                                               pad_to_multiple_of=8,  # pad为8的倍数,提升性能,同时满足s2attention分组时一定被整除
                                               padding="longest")
        # 查看一个样本
        if training_args.local_rank <= 0:
            logger.info(f"one sample of train_dataset input str: \n{tokenizer.decode(train_dataset[0]['input_ids'],skip_special_tokens=False)}")
            logger.info(f"one sample of train_dataset input ids: \n{train_dataset[0]['input_ids']}")
            logger.info(f"one sample of train_dataset labels id: \n{train_dataset[0]['labels']}")
    else:
        train_dataset = PretrainDataset(args.train_file, tokenizer, args.max_seq_length)
        data_collator = PretrainCollator(tokenizer, args.max_seq_length, -100)
    # 初始化Trainer
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # TODO 新增
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_loss=loss_func
    )
    return trainer


def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # TODO 新增代码, 用于方便调试
    if (torch.distributed.is_available() and torch.distributed.is_initialized()):
        torch.distributed.barrier()  # 进程阻塞同步
    # 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # 保存最后的checkpoint
    # trainer.save_model(training_args.output_dir)  # Save the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
