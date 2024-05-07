PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"


# export CUDA_LAUNCH_BLOCKING=1
deepspeed train.py \
    --train_args_file train_args/llama2-7b-sft.yaml \
    2>&1 | tee -a longqlora_log.out


# accelerate launch train.py \
#     --train_args_file train_args/llama2-7b-sft.yaml \
#     2>&1 | tee -a longqlora_log.out
