import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.instance_gen_template import output_first_template_for_clf, input_first_template_for_gen


template = "template_1"

random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_instructions.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_instances.jsonl",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--max_instances_to_generate",
        type=int,
        default=5,
        help="The max number of instances to generate for each instruction.",
    )
    parser.add_argument(
        "--generation_tasks_only",
        action="store_true",
        help="If specified, only do for generation tasks.",
    )
    parser.add_argument(
        "--classification_tasks_only",
        action="store_true",
        help="If specified, only do for classification tasks.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-3.5-turbo-instruct",
        help="The engine to use."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send in a batch."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("***** Parmeters *****".rjust(62))
    arguments = vars(args)
    for key, value in arguments.items():
        print(f"{key}".rjust(50)+ f": {value}")
    print("***** Parmeters *****".rjust(62))    

    with open(os.path.join(args.batch_dir, args.input_file)) as fin:
        lines = fin.readlines()  # 3. 打开指定的输入文件,并将其内容读取到 lines 列表中,每行作为一个元素。
        if args.num_instructions is not None:  # 4. 如果命令行参数 num_instructions 不为 None,则只保留前 num_instructions 行。
            lines = lines[:args.num_instructions]
        tasks = []
        for line in lines:
            data = json.loads(line)  # 5. 使用 json.loads 函数将每行 JSON 字符串解析为 Python 字典对象。
            # 6. 如果字典中存在 "metadata" 键,则将其值复制到 "instruction_metadata" 键,并删除原有的 "metadata" 键。
            if "metadata" in data:
                data["instruction_metadata"] = data["metadata"]
                del data["metadata"]
            tasks.append(data) # 7. 将解析后的字典对象添加到 tasks 列表中。

    task_clf_types = {}
    # 8. 从另一个文件中加载指令及其对应的分类任务标识。
    # 9. 将指令作为键,分类任务标识作为值存储在 task_clf_types 字典中。
    with open(os.path.join(args.batch_dir, f"is_clf_or_not_{args.engine}_{template}.jsonl")) as fin:
        for line in fin:
            data = json.loads(line)
            task_clf_types[data["instruction"]] = data["is_classification"].strip() in ["Yes", "yes", "YES"]

    # 10. 如果命令行参数 classification_tasks_only 为 True,则只保留分类任务的指令。
    if args.classification_tasks_only:
        tasks = [task for task in tasks if task_clf_types[task["instruction"]]]
    
    # 11. 如果命令行参数 generation_tasks_only 为 True,则只保留非分类任务的指令。
    if args.generation_tasks_only:
        tasks = [task for task in tasks if not task_clf_types[task["instruction"]]]
    # 12. 如果指定的输出文件存在,则从中加载已有的请求数据。
    # 13. 使用 tqdm.tqdm 显示加载进度条。
    # 14. 将每个请求数据存储在 existing_requests 字典中,以指令作为键。
    # 15. 打印加载的已有请求数量。
    output_path = os.path.join(args.batch_dir, args.output_file)
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")
    # 16. 创建一个进度条对象 progress_bar,用于跟踪任务处理的进度。
    progress_bar = tqdm.tqdm(total=len(tasks))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(tasks), args.request_batch_size):
            # 17. 将 tasks 列表分批处理,每批包含 args.request_batch_size 个任务
            batch = tasks[batch_idx: batch_idx + args.request_batch_size]
            if all(d["instruction"] in existing_requests for d in batch):
                # 18. 如果当前批次中所有任务的指令都在已有请求中,则直接从现有数据中提取相关信息,并写入输出文件。
                # 19. 使用 OrderedDict 保持字典键的顺序,以确保输出文件中字段的顺序一致。
                # 20. json.dumps 函数将字典对象转换为 JSON 字符串,ensure_ascii=False 用于正确处理非 ASCII 字符。
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "raw_instances", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                prompts = []
                # 21. 对于当前批次中的每个任务,根据其是否为分类任务,使用不同的模板构建提示输入。
                # 22. 将构建的提示添加到 prompts 列表中。
                for task in batch:
                    if task_clf_types[task["instruction"]]:
                        prompt = output_first_template_for_clf + " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                    else:
                        prompt = input_first_template_for_gen + " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                
                # 23. 调用 make_gpt3_requests 函数向 GPT-3 API 发送请求,生成任务实例。
                # 24. 传入了多个参数,如语言模型引擎、提示输入、温度、惩罚项、停止序列等,以控制生成的质量和长度。
                # 25. 对于分类任务,由于模板较长,max_tokens 参数设置为 300,否则设置为 350。
                results = make_gpt3_requests(
                    engine=args.engine,
                    prompts=prompts,
                    # because the clf template is longer, we need to decrease the max_tokens
                    max_tokens=300 if any(task_clf_types[task["instruction"]] for task in batch) else 350,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=1.5,
                    stop_sequences=[f"Example {args.max_instances_to_generate + 1}", "Task:"],
                    logprobs=1,
                    n=1,
                    best_of=1,
                    api_key=args.api_key,
                    organization=args.organization)
                # 26. 对于每个任务,将 GPT-3 生成的实例和元数据添加到任务字典中。
                # 27. 如果 GPT-3 响应为空,则将 "raw_instances" 设置为空字符串。
                # 28. 使用 OrderedDict 保持字典键的顺序,并将字典对象转换为 JSON 字符串写入输出文件。
                for i in range(len(batch)):
                    data = batch[i]
                    data["instance_metadata"] = results[i]
                    if results[i]["response"] is not None:
                        data["raw_instances"] = results[i]["response"]["choices"][0]["text"]
                    else:
                        data["raw_instances"] = ""
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "raw_instances", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            # 29. 更新进度条,反映当前批次的任务已经处理完毕。
            progress_bar.update(len(batch))
