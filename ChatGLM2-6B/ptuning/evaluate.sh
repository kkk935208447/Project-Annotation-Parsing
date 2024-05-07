PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm2-6b-pt-128-1e-3
STEP=1200
NUM_GPUS=1

# --ptuning_checkpoint 加载断点续传的ptuning模型, 但需要注意的是: 这种方式不会加载optim 的状态, 学习率调度器状态等
# --resume_from_checkpoint 加载断点续传模型, 这种方式会加载trainer的State, 会加载optim 的状态, 学习率调度器状态等, 选择那种方式加载模型需要根据实际情况选择
# --max_eval_samples 100 \
# --max_predict_samples 50
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS ptuning/main.py \
    --do_predict \
    --validation_file /workspace/AdvertiseGen/dev.json \
    --test_file /workspace/AdvertiseGen/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /workspace/chatglm2-6b \
    --ptuning_checkpoint /workspace/output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir /workspace/output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \
    --max_predict_samples 10 \
    2>&1 | tee -a log_eval_ptuning.out
