python finetune.py \
    --base_model /workspace/Llama-2-7b-chat-hf \
    --output_dir ~/lora-alpaca \
    2>&1 | tee -a ./log.out