# 加入当前目录的绝对路径
PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_lora.py \
  --base_model /workspace/Llama-2-7b-chat-hf \
  --data_path ./chatdoctor5k.json \
  --output_dir /workspace/output \
  --batch_size 32 \
  --micro_batch_size 1 \
  --num_epochs 1 \
  --learning_rate 3e-5 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora \
  2>&1 | tee -a lora_log.out
  
