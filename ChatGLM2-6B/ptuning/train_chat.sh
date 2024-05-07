PRE_SEQ_LEN=128
LR=1e-2
NUM_GPUS=1


# --ptuning_checkpoint 加载断点续传的ptuning模型, 但需要注意的是: 这种方式不会加载optim 的状态, 学习率调度器状态等
# --resume_from_checkpoint 加载断点续传模型, 这种方式会加载trainer的State, 会加载optim 的状态, 学习率调度器状态等, 选择那种方式加载模型需要根据实际情况选择
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file $CHAT_TRAIN_DATA \
    --validation_file $CHAT_VAL_DATA \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir $CHECKPOINT_NAME \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

