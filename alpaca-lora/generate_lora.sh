python generate.py \
    --base_model /workspace/Llama-2-7b-chat-hf \
    --load_in_8bit \
    --lora_weights ~/lora-alpaca/checkpoint-2 \
    2>&1 | tee -a ./log_gen.out