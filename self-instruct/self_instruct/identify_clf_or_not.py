import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.clf_task_template import template_1


random.seed(42)


templates = {
    "template_1": template_1
}

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--template", type=str, default="template_1", help="Which template to use.")
    parser.add_argument(
        "--batch_dir",             # 生成的新指令保存地址
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default="template_1", 
        help="Which template to use. Currently only `template_1` is supported.",
    )
    parser.add_argument(
        "--engine",              # 不能使用 chat 的model engine
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
        help="The API key to use. If not specified, the key will be read from the environment variable `OPENAI_API_KEY`."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()



# 这段代码的主要功能是:
# 1. 读取一个包含机器生成的指令的输入文件。
# 2. 加载已有的分类标签数据,如果存在的话。
# 3. 对于新的指令,使用 OpenAI API 预测其是否为分类任务。
# 4. 将所有指令及其分类标签写入输出文件。
#
# 它使用了以下主要技术:
# 1. 命令行参数解析
# 2. 文件读写操作
# 3. 字典和列表操作
# 4. JSON 数据处理
# 5. 进度条显示
# 6. 调用外部函数 make_gpt3_requests 进行分类预测
#
# 这些技术的使用旨在实现对输入指令的批处理和分类,并将结果持久化存储,以供后续使用。该代码可用于自然语言处理任务中,通过向语言模型发送大量指令并预测其是否为分类任务,构建分类数据集。
"""
{"instruction": "Find the largest number in this list.", "is_classification": " Yes"} 
{"instruction": "What is the first name of your favorite actor?", "is_classification": " No"} 
{"instruction": "Give me the number of distinct elements in this set.", "is_classification": " Yes"} 
{"instruction": "Give me the top 5 countries that are exporting tea.", "is_classification": " Yes"}
"""
if __name__ == '__main__':
    args = parse_args()

    print("***** Parmeters *****".rjust(62))
    arguments = vars(args)
    for key, value in arguments.items():
        print(f"{key}".rjust(50)+ f": {value}")
    print("***** Parmeters *****".rjust(62))    

    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")) as fin:
        lines = fin.readlines()  # 3. 打开指定的输入文件 "machine_generated_instructions.jsonl",并将其内容读取到 lines 列表中,每行作为一个元素。
        # 4. 如果命令行参数 num_instructions 不为 None,则只保留前 num_instructions 行。
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]
    # 5. 构建输出文件的路径,包括批处理目录、引擎名称和模板名称等信息。
    output_path = os.path.join(args.batch_dir, f"is_clf_or_not_{args.engine}_{args.template}.jsonl")
    existing_requests = {}
    # 6. 检查输出文件是否已经存在,如果存在,则加载其中的数据到 existing_requests 字典中。
    # 7. 使用 tqdm.tqdm 显示加载进度条。
    # 8. 对于每行数据,尝试使用 json.loads() 函数将其解析为字典对象,并将其存储在 existing_requests 字典中,以指令作为键。
    # 9. 打印已加载的现有请求数量。
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")
    # 10. 创建一个进度条对象 progress_bar,用于跟踪任务处理的进度。
    progress_bar = tqdm.tqdm(total=len(lines))
    with open(output_path, "w") as fout:
        # 11. 打开输出文件,准备写入结果。
        # 12. 将输入文件的每个行数据解析为字典对象,构建一个批次 batch。
        # 13. 每个批次包含 args.request_batch_size 个任务。
        for batch_idx in range(0, len(lines), args.request_batch_size):
            batch = [json.loads(line) for line in lines[batch_idx: batch_idx + args.request_batch_size]]
            # 14. 检查当前批次中的所有任务是否都已经存在于 existing_requests 字典中。
            # 15. 如果全部存在,则直接将对应的数据(指令和分类标签)写入输出文件。
            # 16. 使用 OrderedDict 保持字典键的顺序,以确保输出文件中字段的顺序一致。
            # 17. 使用 json.dumps() 函数将字典对象转换为 JSON 字符串,并换行写入。
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                # prefix = compose_prompt_prefix(human_written_tasks, batch[0]["instruction"], 8, 2)
                # 18. 如果当前批次中存在新的任务,则构建提示输入。
                # 19. 使用指定的模板 (templates[args.template]) 作为前缀,并拼接任务指令和分类问题。
                prefix = templates[args.template]
                prompts = [prefix + " " + d["instruction"].strip() + "\n" + "Is it classification?" for d in batch]

                # 20. 调用 make_gpt3_requests 函数向 OpenAI API 发送请求,以获取每个任务的分类标签。
                # 21. 传入了多个参数,如语言模型引擎、提示输入、最大生成长度、采样参数等。
                results = make_gpt3_requests(
                    engine=args.engine,
                    prompts=prompts,
                    max_tokens=3,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop_sequences=["\n", "Task"],
                    logprobs=1,
                    n=1,
                    best_of=1,
                    api_key=args.api_key,
                    organization=args.organization)
                # 22. 遍历当前批次的任务,将从 OpenAI API 获取的分类标签添加到任务数据中。
                # 23. 如果 API 响应为空,则将分类标签设置为空字符串。
                # 24. 使用 OrderedDict 保持字典键的顺序,并将字典对象转换为 JSON 字符串写入输出文件。
                for i in range(len(batch)):
                    data = batch[i]
                    if results[i]["response"] is not None:
                        data["is_classification"] = results[i]["response"]["choices"][0]["text"]
                    else:
                        data["is_classification"] = ""
                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"]
                    }
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            # 25. 更新进度条,反映当前批次的任务已经处理完毕。
            progress_bar.update(len(batch))
