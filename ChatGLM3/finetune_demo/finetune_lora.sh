PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

torchrun --nproc_per_node=1 finetune_demo/finetune_hf.py \
    /workspace/AdvertiseGen_fix \
    /workspace/chatglm3-6b \
    finetune_demo/configs/lora.yaml \
    2>&1 | tee -a finetune_demo/log_lora.out