PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"


PRE_SEQ_LEN=128
# LR=2e-2
LR=1e-3
NUM_GPUS=1

# --quantization_bit 4 量化加载
# --do_eval            用于trainer 全部训练完后进行一次评估
# --save_total_limit 10 \
# --max_train_samples 100  # 限定了最大的训练样本数, 方便调试
# --max_eval_samples 50    # 限定了最大的验证样本数, 方便调试
# --save_total_limit 10 \
# --evaluation_strategy "steps" \
# --eval_steps 20 \

# --ptuning_checkpoint 加载断点续传的ptuning模型, 但需要注意的是: 这种方式不会加载optim 的状态, 学习率调度器状态等
# --resume_from_checkpoint 加载断点续传模型, 这种方式会加载trainer的State, 会加载optim 的状态, 学习率调度器状态等, 选择那种方式加载模型需要根据实际情况选择
# --weight_decay 0.1 \ ptuning 的权重衰减
# --warmup_ratio 0.05 \ ptuning 的warmup比例
# --max_grad_norm 0.5 \ ptuning 的最大梯度, 即梯度裁剪
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS ptuning/main.py \
    --do_train \
    --do_eval \
    --train_file /workspace/AdvertiseGen/train.json \
    --validation_file /workspace/AdvertiseGen/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /workspace/chatglm2-6b \
    --output_dir /workspace/output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --save_total_limit 10 \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \
    --weight_decay 0.01 \
    --max_grad_norm 0.5 \
    2>&1 | tee -a log.out

