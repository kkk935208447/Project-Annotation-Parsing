PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"


# --resume_from_checkpoint dir  表示trainer从dir 恢复 ckpt
# 注释掉: 与wandb 不能共存
# 2>&1 | tee -a examples/sft/qlora_ds_zero3_log.out
# --use_flash_attn True \
accelerate launch --config_file "examples/sft/configs/deepspeed_config_z3_qlora.yaml"  examples/sft/train.py \
    --seed 100 \
    --model_name_or_path "/workspace/Llama-2-7b-chat-hf" \
    --dataset_name "smangrul/ultrachat-10k-chatml" \
    --chat_template_format "chatml" \
    --add_special_tokens False \
    --append_concat_token False \
    --splits "train,test" \
    --max_seq_len 2048 \
    --num_train_epochs 2 \
    --logging_steps 1 \
    --log_level "info" \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 3 \
    --bf16 True \
    --packing True \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --max_grad_norm 1.0 \
    --output_dir "/workspace/output/llama-sft-qlora-dsz3" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --use_reentrant True \
    --use_flash_attn True \
    --dataset_text_field "content" \
    --use_peft_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target_modules "all-linear" \
    --use_4bit_quantization True \
    --use_nested_quant True \
    --bnb_4bit_compute_dtype "bfloat16" \
    --bnb_4bit_quant_storage_dtype "bfloat16" \
    2>&1 | tee -a examples/sft/qlora_ds_zero3_log.out

    # 上传至 hub 的参数
    # --push_to_hub \
    # --hub_private_repo True \
    # --hub_strategy "every_save" \