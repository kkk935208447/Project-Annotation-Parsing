# batch_dir=data/gpt3_generations/

# python self_instruct/bootstrap_instructions.py \
#     --batch_dir ${batch_dir} \
#     --num_instructions_to_generate 50000 \
#     --seed_tasks_path data/seed_tasks.jsonl \
#     --engine "davinci"


# 加入当前目录的绝对路径
PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

batch_dir=data/gpt3_generations_ceshi/
python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 100 \
    --seed_tasks_path data/seed_tasks.jsonl \
    --engine "gpt-3.5-turbo-instruct" \
    --api_key sk-xxxxxxxxxxxxxxxxx
