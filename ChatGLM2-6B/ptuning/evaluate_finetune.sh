PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

CHECKPOINT=adgen-chatglm2-6b-ft-1e-4
STEP=3000
NUM_GPUS=1

# --ptuning_checkpoint 加载断点续传的ptuning模型, 但需要注意的是: 这种方式不会加载optim 的状态, 学习率调度器状态等
# --resume_from_checkpoint 加载断点续传模型, 这种方式会加载trainer的State, 会加载optim 的状态, 学习率调度器状态等, 选择那种方式加载模型需要根据实际情况选择
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_predict \
    --validation_file AdvertiseGen/dev.json \
    --test_file AdvertiseGen/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path ./output/$CHECKPOINT/checkpoint-$STEP  \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --fp16_full_eval
