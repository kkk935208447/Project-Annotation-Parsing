# batch_dir=data/gpt3_generations/

# python self_instruct/identify_clf_or_not.py \
#     --batch_dir ${batch_dir} \
#     --engine "davinci" \
#     --request_batch_size 5



# 加入当前目录的绝对路径
PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

batch_dir=data/gpt3_generations_ceshi/
python self_instruct/identify_clf_or_not.py \
    --batch_dir ${batch_dir} \
    --engine "gpt-3.5-turbo-instruct" \
    --request_batch_size 5 \
    --api_key sk-xxxxxxxxxxxxxxxxx