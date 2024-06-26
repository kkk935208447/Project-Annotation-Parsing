# 加入当前目录的绝对路径
PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

# --bf16 开启 bf 16 混合精度
# -- tf32 关闭
# gradient_checkpointing=False 内存不够, 加入梯度检查点
torchrun --nproc_per_node=2 train.py \
    --model_name_or_path /workspace/Llama-2-7b-chat-hf \
    --data_path ./chatdoctor5k.json \
    --bf16 True \
    --output_dir /workspace/output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --deepspeed ds_zero3.json \
    --gradient_checkpointing True \
    --max_grad_norm 0.5 \
    2>&1 | tee -a ds_full_log.out
    # --tf32 True