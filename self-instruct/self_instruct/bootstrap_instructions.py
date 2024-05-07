import os
import json
import random
import re
# 5. 导入 string 模块,提供一些常用的字符串常量和操作函数。
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
# 10. 从 multiprocessing 模块导入 Pool 类,用于实现多进程并行计算。
from multiprocessing import Pool
# 11. 从 functools 模块导入 partial 函数,用于创建偏函数(Partial Function),可以固定部分参数以简化函数调用。
from functools import partial
# 12. 从 rouge_score 模块导入 rouge_scorer 类,用于计算文本摘要的 ROUGE 评分,ROUGE 是一种常用的文本摘要评估指标。
from rouge_score import rouge_scorer
# 13. 从 gpt3_api 模块导入 make_requests 函数,重命名为 make_gpt3_requests,用于与 GPT-3 API 进行交互。
from gpt3_api import make_requests as make_gpt3_requests


random.seed(42)

# 这个函数的作用是将多个提示指令编码为单个字符串,以便后续的任务生成或分类。
# 它使用了一些字符串处理技术,如正则表达式替换、字符串格式化等,来规范化指令的格式并生成最终的提示字符串。
# 这个函数可以用于自然语言处理任务,例如文本生成、分类等,通过编码提示指令来指导模型生成所需的输出。
def encode_prompt(prompt_instructions, classification=False):
    # 这是一个函数,用于将多个提示指令编码为单个字符串。
    """Encode multiple prompt instructions into a single string."""
    # 1. 如果 classification 为 True,表示这是一个分类任务。
    if classification:
         # 2. 设置提示为生成一系列分类任务,并尽可能指定可能的输出标签。
        prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
    else:
         # 3. 如果 classification 为 False,表示这是一个其他类型的任务。
         # 4. 设置提示为生成一系列任务。
        prompt = "Come up with a series of tasks:\n"
    # 5. 遍历每个提示指令 instruction。
    # 6. idx 是指令的索引,instruction 是指令的内容。
    for idx, instruction in enumerate(prompt_instructions):
        # 7. 使用正则表达式替换多个连续空白字符为单个空格,并去除指令前后的空白字符和结尾的冒号。
        # 8. 这是为了规范化指令的格式。
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        # 9. 将规范化后的指令添加到提示字符串中,并在前面加上序号。
        prompt += f"{idx+1}. {instruction}\n"
    # 10. 在提示字符串的最后添加一个新的序号,用于生成下一个任务或标签。
    prompt += f"{len(prompt_instructions) + 1}."
    # 11. 返回编码后的提示字符串。
    return prompt


# 这个函数的作用是从给定的机器指令列表中随机采样 n 个指令。
# 它使用了 Python 标准库中的 random.sample 函数,可以从一个列表中随机采样指定数量的元素。
# 该函数可以用于数据增强、模型评估等场景,通过随机采样机器指令来生成不同的输入数据。
def sample_machine_instructions(machine_instructions, similarities, n):
    # 这是一个函数,用于从机器指令列表中随机采样 n 个指令。
    """Sample n machine instructions from a list of machine instructions."""
    # 1. 使用 random.sample 函数从 machine_instructions 列表中随机采样 n 个元素。
    # 2. min(n, len(machine_instructions)) 确保采样数量不超过列表的长度。
    # 3. 返回采样得到的机器指令列表。
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):
    # 这是一个函数,用于在字符串 s 中查找单词 w。

    # 1. 使用正则表达式 r'\b({0})\b' 创建一个模式,其中 {0} 将被替换为单词 w。
    # 2. \b 是一个单词边界匹配符,用于匹配单词的开始和结束位置。
    # 3. flags=re.IGNORECASE 表示在匹配时忽略大小写。
    # 4. re.compile 函数编译正则表达式模式,用于后续的匹配操作。
    # 5. search 方法在字符串 s 中搜索匹配的子字符串,如果找到则返回一个匹配对象,否则返回 None。
    # 6. 返回匹配对象或 None。
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


# 这个函数的作用是对 GPT-3 模型的响应进行后处理,以获取更加合适的指令列表。
# 它使用了多种过滤技术,包括正则表达式、字符串操作、关键词过滤等,来过滤掉不合适的指令。
# 这些过滤技术针对的问题包括:
# 1. 指令长度不合适(太短或太长)
# 2. 包含不适合语言模型的关键词(如图像、文件等)
# 3. 以特定模式开头的指令(如 "Write a program")
# 4. 以标点符号或非英文字符开头的指令
# 通过这些过滤,函数可以获得更加合理、清晰的指令列表,以便后续的任务处理。
# 这种后处理对于提高模型输出的质量和可用性非常重要,尤其是在需要生成自然语言的任务中。
def post_process_gpt3_response(response):
    # 这是一个函数,用于对 GPT-3 模型的响应进行后处理。

    # 1. 如果响应为空或者完成原因是序列长度超出限制,则返回空列表。
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []
    
    # 2. 使用正则表达式将响应文本按照 "\n数字. " 的模式进行分割,得到原始指令列表。
    # 3. 这种模式匹配每个指令前面的序号和换行符。
    raw_instructions = re.split(r"\n\d+\s?\. ", response["choices"][0]["text"])
    # 4. 创建一个空列表,用于存储处理后的指令。
    instructions = []
    for inst in raw_instructions:
        # 5. 使用正则表达式将多个连续空白字符替换为单个空格,并去除前后的空白字符。
        inst = re.sub(r"\s+", " ", inst).strip()
        # 6. 再次去除前后的空白字符,并将指令的第一个字符转换为大写。
        inst = inst.strip().capitalize()
        if inst == "":
            # 7. 如果指令为空字符串,则跳过该指令。
            continue
        # filter out too short or too long instructions
        # 8. 过滤掉太短或太长的指令。
        # 9. 太短的指令可能缺乏足够的信息,而太长的指令可能包含无关的内容。
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        # 10. 过滤掉包含不适合于语言模型的关键词的指令,如 "image"、"graph"、"picture"、"file"、"map"、"draw"、"plot"、"go to"。
        # 11. 这些关键词通常与生成图像或文件相关,而不是生成文本。
        if any(find_word_in_string(word, inst) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result. 
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        # 12. 过滤掉以 "Write a program" 开头的指令。
        # 13. 这是因为模型倾向于将这种指令添加到一些现有指令中,导致出现大量这种指令。
        # 14. 对于这种指令,不清楚模型是需要编写程序还是直接输出结果。
        # 15. 注意,这里的过滤并不是针对所有编程相关的指令。
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        # 16. 过滤掉以标点符号开头的指令。
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        # 17. 过滤掉以非英文字符开头的指令。
        if not inst[0].isascii():
            continue
        # 18. 将通过上述过滤的指令添加到 instructions 列表中。
        instructions.append(inst)
    # 19. 返回处理后的指令列表。
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="data/gpt3_generations/",    # 生成的新指令保存地址
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",  # 175种子指令路径
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",   # 利用llm生成的指令数
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",        # 是否只使用分类的seed task
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-3.5-turbo-instruct", # 不能使用 chat 的model engine
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",    # 随机抽样种子指令和机器生成指令的总体数量
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
    )
    parser.add_argument(
        "--api_key",               # openai api key
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",           # openai organization
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("***** Parmeters *****".rjust(62))
    arguments = vars(args)
    for key, value in arguments.items():
        print(f"{key}".rjust(50)+ f": {value}")
    print("***** Parmeters *****".rjust(62))

    # 3. 从指定的文件路径 args.seed_tasks_path 中读取种子任务数据。
    # 4. 使用列表推导式将每行数据 l 通过 json.loads 从 JSON 字符串转换为 Python 字典对象,并存储在 seed_tasks 列表中
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    # 5. 如果命令行参数 use_clf_seed_tasks_only 为 True,则只保留那些 is_classification 字段为 True 的种子任务。
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    # 6. 从 seed_tasks 列表中提取每个任务的 "instruction" 字段,并存储在 seed_instructions 列表中。
    seed_instructions = [t["instruction"] for t in seed_tasks]
    # 7. 打印加载的人工编写的种子指令数量。
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")
    # 8. 使用 os.makedirs 函数创建指定的批次目录 args.batch_dir,如果目录已存在则不会报错。
    os.makedirs(args.batch_dir, exist_ok=True)
    # 9. 初始化请求索引 request_idx 为 0。
    request_idx = 0
    # load the LM-generated instructions
    # 10. 初始化一个空列表 machine_instructions,用于存储由语言模型生成的指令。
    machine_instructions = []
    # 11. 如果存在 "machine_generated_instructions.jsonl" 文件,则从中加载之前生成的机器指令。
    # 12. 使用 json.loads 将每行数据从 JSON 字符串转换为 Python 字典对象。
    # 13. 将每个指令 instruction_info["instruction"] 添加到 machine_instructions 列表中。
    # 14. 更新 request_idx 为最后一个指令的 request_idx 加 1。
    # 15. 打印加载的机器生成指令数量。
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities = {}
    # 17. 创建一个 RougeScorer 对象,用于计算 ROUGE-L 分数。
    # 18. ROUGE-L 是一种自动评估文本生成质量的指标,常用于评估机器翻译和文本摘要等任务。
    # 19. use_stemmer=False 表示不使用词干提取,直接使用原始词形。
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    # now let's generate new instructions!
    # 20. 创建一个进度条对象 progress_bar,用于跟踪新指令生成的进度。
    # 21. tqdm 是一个第三方库,可以在终端显示进度条和剩余时间等信息。
    # 22. total=args.num_instructions_to_generate 表示总共需要生成的指令数量。
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    # 23. 如果已经加载了一些机器生成的指令,则将进度条更新到这些指令的数量。
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            batch_inputs = []
            for _ in range(args.request_batch_size):
                # sample machine instructions from the pool
                # 24. 从已有的机器指令池中随机采样 2 个指令,作为提示指令的一部分。
                # 25. sample_machine_instructions 函数的实现在其他地方定义。
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, similarities=None,n=2)
                # sample human instructions from the pool
                # 26. 从人工编写的种子指令中随机采样一些指令,并添加到提示指令列表中。
                # 27. 采样的数量是 args.num_prompt_instructions 减去已采样的机器指令数量。
                prompt_instructions += random.sample(seed_instructions, args.num_prompt_instructions - len(prompt_instructions))
                # 28. 对提示指令列表进行随机打乱,以增加多样性。
                random.shuffle(prompt_instructions)
                # 29. 调用 encode_prompt 函数将提示指令编码为一个字符串,作为语言模型的输入。
                # 30. classification 参数指示是否仅使用分类任务的种子指令。
                prompt = encode_prompt(prompt_instructions, classification=args.use_clf_seed_tasks_only)
                # 31. 将编码后的提示添加到批次输入列表 batch_inputs 中。
                batch_inputs.append(prompt)

            # 32. 调用 make_gpt3_requests 函数向 GPT-3 API 发送请求,生成新的指令。
            # 33. 传入了多个参数,如语言模型引擎、温度、top-p采样参数、惩罚项等,以控制生成的多样性和质量。
            # 34. 还指定了停止序列和输出数量等参数。
            results = make_gpt3_requests(
                engine=args.engine,
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                logprobs=1,
                n=1,
                best_of=1,
                api_key=args.api_key,
                organization=args.organization,
            )
            instructions = []
            all_metadata = []
            for result in results:
                # 35. 对每个 GPT-3 响应调用 post_process_gpt3_response 函数进行后处理,得到新的指令列表。
                new_instructions = post_process_gpt3_response(result["response"])
                # 36. 将新指令添加到 instructions 列表中。
                instructions += new_instructions
                # 37. 将响应的元数据重复添加到 all_metadata 列表中,与新指令一一对应。
                all_metadata += [result] * len(new_instructions)


            # 1. 使用 zip 函数将 instructions 列表和 all_metadata 列表打包成一个元组列表。
            # 2. 每个元组包含一个指令 inst 和对应的元数据 metadata。
            # 3. 这种方式可以方便地同时遍历两个列表,并将对应的元素关联起来。
            for inst, metadata in zip(instructions, all_metadata):
                # 4. 创建一个包含 4 个进程的进程池 p。
                # 5. 使用 multiprocessing.Pool 可以并行执行多个任务,提高计算效率。
                # 6. 将 ROUGE-L 分数计算任务分配给这 4 个进程,充分利用 CPU 资源。
                with Pool(4) as p:
                    # 7. 使用 p.map 函数将 scorer.score 函数并行应用于所有已有指令。
                    # 8. scorer.score 函数用于计算新指令 inst 与某个指令之间的 ROUGE-L 分数。
                    # 9. partial 函数用于部分应用参数,将 inst 作为第一个参数固定,剩下的参数从 seed_instructions + machine_instructions 中取。
                    # 10. 结果是一个包含所有 ROUGE-L 分数的列表 rouge_scores。
                    rouge_scores = p.map(partial(scorer.score, inst), seed_instructions + machine_instructions)

                # 11. 从计算结果中提取 ROUGE-L F-measure 分数,存储在 rouge_scores 列表中。
                # 12. scorer.score 函数返回一个字典,其中 "rougeL" 键对应的值是一个包含多个指标的对象,使用 .fmeasure 属性可以获取 F-measure 分数。
                rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
                # rouge_scores = [scorer.score(inst, e_inst)["rougeL"].fmeasure for e_inst in human_instructions + machine_instructions]
                
                # 15. 如果新指令 inst 与任何已有指令的最大 ROUGE-L 分数超过 0.7,则跳过该指令。
                # 16. 这是为了避免生成与已有指令过于相似的重复指令,确保新指令的多样性和独特性。
                if max(rouge_scores) > 0.7:
                    continue

                # 17. np.argsort返回按升序排列的 rouge_scores 的索引数组。[-10:] 表示取相似度最高的10个索引, [::-1] 再把它们倒序, 即最相似的10个索引降序排列。
                # 20. 使用字典推导式构建一个字典 most_similar_instructions,其中键为最相似的 10 个指令,值为对应的 ROUGE-L 分数。
                all_instructions = seed_instructions + machine_instructions
                most_similar_instructions = {
                        all_instructions[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                    }
                # 21. 将新指令 inst 添加到 machine_instructions 列表中,用于后续的指令生成。
                machine_instructions.append(inst)
                # 22. 将新指令及其相关信息写入 "machine_generated_instructions.jsonl" 文件。
                # 23. 包括指令内容、最相似的 10 个指令及分数、平均相似度分数、响应元数据和请求索引。
                # 24. json.dumps 函数将字典对象转换为 JSON 字符串,以便写入文件。
                # 25. float(np.mean(rouge_scores)) 计算所有 ROUGE-L 分数的平均值,作为平均相似度分数。
                fout.write(json.dumps({
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    "metadata": metadata,
                    "request_idx": request_idx
                }) + "\n")
                # 26. 更新进度条,表示生成了一个新指令。
                progress_bar.update(1)
            # 27. 增加请求索引 request_idx,用于下一批次的请求。
            request_idx += 1
