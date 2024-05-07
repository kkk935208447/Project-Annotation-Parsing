# batch_dir=data/gpt3_generations/

# python self_instruct/prepare_for_finetuning.py \
#     --instance_files ${batch_dir}/machine_generated_instances.jsonl \
#     --classification_type_files ${batch_dir}/is_clf_or_not_davinci_template_1.jsonl \
#     --output_dir ${batch_dir}/finetuning_data \
#     --include_seed_tasks \
#     --seed_tasks_path data/seed_tasks.jsonl

# 加入当前目录的绝对路径
PYTHONPATH=$PWD
export PYTHONPATH
echo "当前bash执行目录: $PWD, 已经将PYTHONPATH设置为: $PYTHONPATH"


batch_dir=data/gpt3_generations_ceshi
engine=gpt-3.5-turbo-instruct
python self_instruct/prepare_for_finetuning.py \
    --instance_files ${batch_dir}/machine_generated_instances.jsonl \
    --classification_type_files ${batch_dir}/is_clf_or_not_${engine}_template_1.jsonl \
    --output_dir ${batch_dir}/finetuning_data \
    --include_seed_tasks \
    --seed_tasks_path data/seed_tasks.jsonl