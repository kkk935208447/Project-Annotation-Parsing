# batch_dir=data/gpt3_generations/

# python self_instruct/generate_instances.py \
#     --batch_dir ${batch_dir} \
#     --input_file machine_generated_instructions.jsonl \
#     --output_file machine_generated_instances.jsonl \
#     --max_instances_to_gen 5 \
#     --engine "davinci" \
#     --request_batch_size 5

# 加入当前目录的绝对路径
PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"

batch_dir=data/gpt3_generations_ceshi/


# --max_instances_to_gen 5 \
python self_instruct/generate_instances.py \
    --batch_dir ${batch_dir} \
    --input_file machine_generated_instructions.jsonl \
    --output_file machine_generated_instances.jsonl \
    --engine "gpt-3.5-turbo-instruct" \
    --request_batch_size 5 \
    --api_key sk-xxxxxxxxxxxxxxxxx