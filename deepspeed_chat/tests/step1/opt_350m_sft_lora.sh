#!/bin/bash
###
 # @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @Date: 2024-03-11 13:23:55
 # @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @LastEditTime: 2024-03-13 15:34:53
 # @FilePath: /deepspeed_chat/tests/my_tutorial/opt_350m_sft.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team


# 定于bash python 优先搜索的项目路径,以防导包失败
    # PYTHONPATH=$PWD:/home/bennie/bennie/bennie_project/AI_Lab
    # PYTHONPATH: 这是一个环境变量，Python解释器会在其中查找模块。当你尝试导入Python模块时，解释器会先在PYTHONPATH指定的目录中搜索。
    # $PWD: 表示当前工作目录(current working directory)的路径。Shell会自动展开它为执行该命令所处的文件夹路径。
    # /home/bennie/bennie/bennie_project/AI_Lab: 是一个明确指定的文件夹路径，你希望Python也能从中寻找模块。
    # 通过将以上部分组合起来（使用冒号“:”分隔），实现了把当前工作目录和“/home/bennie...AI_Lab”同时加入到了PYTHONPATH`里面。
    # export PYTHONPATH
    # 将前一步骤修改过或新建立的环境变量输出到终端session中，并使其可被后续任何由此shell启动或控制下运行程序访问和使用。如果不执行此操作，则只有直接运行该赋值语句本身或者子进程才能识别新设定值；父进程及其他并发进行则无法获取更新后内容。

# 加入当前目录的绝对路径
PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

# 模型地址, 本地模型路径或hugging face模型线上路径
MODEL_PATH=facebook/opt-350m
# 定义并创建checkpoint等文件的输出目录
OUTPUT_PATH=/root/ckeckpoint/dsc_step1/lora
mkdir -p $OUTPUT_PATH
# 数据地址
DATA_PATH=Dahoas/rm-static
# 设置ZeRO 优化阶段
ZERO_STAGE=3

echo "----------------------------- 开始训练 -----------------------------"
# 开始训练
# local_rank = --num_gpus + (-1)
# --print_loss 每一个小的step都会打印loss,不需要了可以删除掉.
# --offload \ 开启offload
# --gradient_checkpointing \开启梯度检查
# --add_eot_token 这个参数需要打开,确保 <|endoftext|> 可以被分词
deepspeed --num_gpus 1 training/step1_supervised_finetuning/main.py \
    --data_path $DATA_PATH \
    --model_name_or_path $MODEL_PATH \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --max_seq_len 512 \
    --learning_rate 1e-5 \
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
    --lora_dim 64 \
    --lora_module_name decoder.layers. \
    --only_optimize_lora \
    --lora_learning_rate 5e-4 \
    --compute_fp32_loss \
    --enable_tensorboard \
    --tensorboard_path $OUTPUT_PATH \
    --deepspeed \
    --output_dir $OUTPUT_PATH \
    2>&1 | tee -a $OUTPUT_PATH/log.out

# 2>&1 | tee -a log.out 输出到日志文件(正确错误都输入),前台也会展示