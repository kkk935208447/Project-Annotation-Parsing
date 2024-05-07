PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

PRE_SEQ_LEN=128

# --ptuning_checkpoint 加载断点续传的ptuning模型, 但需要注意的是: 这种方式不会加载optim 的状态, 学习率调度器状态等
# --resume_from_checkpoint 加载断点续传模型, 这种方式会加载trainer的State, 会加载optim 的状态, 学习率调度器状态等, 选择那种方式加载模型需要根据实际情况选择
# quantization_bit 4
CUDA_VISIBLE_DEVICES=0 python3 ptuning/web_demo.py \
    --model_name_or_path /workspace/chatglm2-6b \
    --ptuning_checkpoint /workspace/output/adgen-chatglm2-6b-pt-128-1e-3/checkpoint-1200 \
    --pre_seq_len $PRE_SEQ_LEN

