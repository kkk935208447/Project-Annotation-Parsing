# 加入当前目录的绝对路径
PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

# TODO 修改官方, 暂时关闭 --bf16 True \ --tf32 True \
torchrun --nproc_per_node=8 train.py \
    --model_name_or_path /workspace/Llama-2-7b-chat-hf \
    --data_path ./alpaca_data.json \
    --output_dir ./alpaca_out \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json"