PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"


accelerate launch --num_processes 1 --config_file train_args/deepspeed/deepspeed_plugin_z23_longqlora.yaml \
    train_zero23.py \
    --train_args_file train_args/llama2-7b-sft_zero23.yaml \
    2>&1 | tee -a longqlora_s123_log.out






# deepspeed train_zero23.py \
#     --train_args_file train_args/llama2-7b-sft_zero23.yaml \
#     2>&1 | tee -a longqlora_s123_log.out












