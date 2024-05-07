#!/bin/bash
# 加入当前目录的绝对路径
PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

# 模型地址, 本地模型路径或hugging face模型线上路径
MODEL_PATH=facebook/opt-350m
# 定义并创建checkpoint等文件的输出目录
OUTPUT_PATH=/root/ckeckpoint/dsc_step1/full
mkdir -p $OUTPUT_PATH
# 数据地址
DATA_PATH=Dahoas/rm-static
# 设置ZeRO 优化阶段
ZERO_STAGE=2



echo "----------------------------- 开始训练 -----------------------------"
# 开始训练
# local_rank = --num_gpus + (-1)
# --print_loss 每一个小的step都会打印loss,不需要了可以删除掉.
# --offload \ 开启offload
# --gradient_checkpointing \开启梯度检查
# --add_eot_token 这个参数需要打开,确保 <|endoftext|> 可以被分词
deepspeed --num_gpus 1 training/step1_supervised_finetuning/main.py \
    --data_path $DATA_PATH \
    --data_split 2,4,4 \
    --model_name_or_path $MODEL_PATH \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --max_seq_len 512 \
    --learning_rate 9.65e-6 \
    --weight_decay 0.1 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 42 \
    --zero_stage $ZERO_STAGE \
    --gradient_checkpointing \
    --offload \
    --add_eot_token \
    --compute_fp32_loss \
    --deepspeed \
    --enable_tensorboard \
    --tensorboard_path $OUTPUT_PATH \
    --output_dir $OUTPUT_PATH \
    2>&1 | tee -a $OUTPUT_PATH/log.out
