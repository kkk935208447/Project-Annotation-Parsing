#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"
# checkpoint文件输出路径
OUTPUT_PATH=/root/ckeckpoint/dsc_step3/lora
mkdir -p $OUTPUT_PATH
# 数据路径
DATA_PATH=/root/DSC/datas
# 模型路径
ACTOR_MODEL_PATH=/root/sim_chatgpt/opt-350m
CRITIC_MODEL_PATH=/root/sim_chatgpt/opt-125m
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3
Num_Padding_at_Beginning=1 # this is model related
Actor_Lr=5e-4
Critic_Lr=5e-6
echo "----------------------------- 开始训练 -----------------------------"
# --offload_reference_model
# 目前--enable_hybrid_engine会报一个错误, 移除 --enable-hybric-engine, 加上 --offload_reference_model 
# --add_eot_token 这个参数需要打开,确保 <|endoftext|> 可以被分词
deepspeed --num_gpus 1 training/step3_rlhf_finetuning/main.py \
   --data_path $DATA_PATH \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 8 \
   --per_device_training_batch_size 8 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --num_warmup_steps 100 \
   --seed 42 \
   --add_eot_token \
   --deepspeed \
   --offload_reference_model \
   --inference_tp_size 2 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --actor_gradient_checkpointing \
   --actor_dropout 0.0 \
   --actor_lora_dim 128 \
   --actor_lora_module_name decoder.layers. \
   --output_dir $OUTPUT_PATH \
   2>&1 | tee -a $OUTPUT_PATH/log.out