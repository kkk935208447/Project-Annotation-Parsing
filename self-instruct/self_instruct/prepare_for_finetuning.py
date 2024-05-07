import os
import json
import argparse
import glob
import re
import random
import tqdm
import pandas as pd


random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_files",
        nargs="+",
        default=["data/batch_221203/machine_generated_instances.jsonl"],
        type=str,
        help="The input files that contain the machine generated instances."
    )
    parser.add_argument(
        "--classification_type_files",
        nargs="+",
        default=["data/batch_221203/is_clf_or_not_davinci_template_1.jsonl"],
    )
    parser.add_argument(
        "--output_dir",
        default="data/gpt3_generations/batch_221203/finetuning/",
        type=str,
        help="The output dir to save the cleaned version of the generated instances, so that it can be used for GPT3 finetuning."
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="The number of instructions to load."
    )
    parser.add_argument(
        "--include_seed_tasks",
        action="store_true",
        help="Whether to include the seed human-written instances in the finetuning data."
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the seed data.",
    )
    return parser.parse_args()

# 这个函数的主要作用是将 Alpaca 数据集中的 instruction、input 和 output 按照一定的模板进行编码,生成训练所需的数据格式。它使用了以下几种核心技术:
# 模板设计: 该函数定义了两组编码模板,一组包含 input 字段,另一组不包含。通过动态填充这些模板,可以生成多种格式的 prompt 和 completion 数据,增加训练数据的多样性。
# 随机选择模板: 该函数支持随机选择模板的功能,当 random_template 参数为 True 时,会从两组模板中随机选择一个进行编码。这种随机性可以进一步增强训练数据的多样性,提高模型的泛化能力。
# 数据打包: 最终,该函数将 prompt、completion 以及其他相关信息打包成一个字典并返回,方便后续的数据处理和训练。
# 总的来说,这个函数是 Alpaca 数据集预处理的一个关键步骤,它利用模板设计和随机选择等技术,生成了格式化的训练数据,为后续的模型训练做好了准备。
def encode_instance(instruction, input, output, random_template=True):
    # 1. 这个函数的作用是将给定的 instruction、input 和 output 按照一定的模板进行编码,生成训练所需的数据格式。
    # 2. 函数接受四个参数:
    #    - instruction: 任务描述或指令
    #    - input: 任务输入
    #    - output: 任务输出
    #    - random_template: 是否随机选择模板,默认为 True
    encoding_templates_w_input = [
        # 3. 这是一个包含多个编码模板的列表,这些模板都包含 input 字段。
        # 4. 每个模板都是一个包含 prompt 和 completion 两个部分的字符串,使用 python 的 format 函数进行动态填充。
        ("{instruction}\nInput: {input}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\nInput: {input}\n\nOutput:", " {output}<|endoftext|>"),
        ("Task: {instruction}\nInput: {input}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\n{input}\n\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\n{input}\n\n", "{output}<|endoftext|>"),
        ("{instruction}\n{input}\n\n", "{output}<|endoftext|>"),
        ("Task: {instruction}\n\n{input}\n\n", "{output}<|endoftext|>"),
    ]
    encoding_templates_wo_input = [
        # 5. 这是另一个包含多个编码模板的列表,这些模板不包含 input 字段。
        ("{instruction} Output:", " {output}<|endoftext|>"),
        ("{instruction}\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n\nOutput:", " {output}<|endoftext|>"),
        ("{instruction}\n", "{output}<|endoftext|>"),
        ("{instruction}\n\n", "{output}<|endoftext|>"),
        ("Task: {instruction}\n\n", "{output}<|endoftext|>"),
    ]
    if random_template:
        # 6. 如果 random_template 为 True,则随机选择一个模板进行编码
        if input.strip() != "":
            # 7. 如果 input 不为空,则从 encoding_templates_w_input 列表中随机选择一个模板
            prompt_template, completion_template = random.choice(encoding_templates_w_input)
            prompt = prompt_template.format(instruction=instruction.strip(), input=input.strip())
            completion = completion_template.format(output=output.strip())
        else:
            # 8. 如果 input 为空,则从 encoding_templates_wo_input 列表中随机选择一个模板
            prompt_template, completion_template = random.choice(encoding_templates_wo_input)
            prompt = prompt_template.format(instruction=instruction.strip())
            completion = completion_template.format(output=output.strip())
    else:
        # 9. 如果 random_template 为 False,则使用固定的模板进行编码
        prompt = instruction.strip() + "\n\n" + input.strip() + "\n\n"
        completion = output.strip() + "<|endoftext|>"

    # 10. 最终将 prompt 和 completion 以及其他相关信息打包成一个字典并返回
    data = {
        "prompt": prompt,
        "completion": completion,
        "instruction": instruction.strip(),
        "input": input.strip(),
        "output": output.strip(),
    }
    return data

# 这个函数使用了以下几种核心技术:
# 正则表达式: 该函数大量使用正则表达式来匹配和提取输入输出信息。通过灵活使用正则表达式,可以处理各种复杂的文本格式。
# 字符串拆分和清理: 该函数将 response_text 字符串拆分成输入和输出部分,并去除首尾空白字符。同时还处理可能出现的多个输入/输出对的情况。
# 容错性处理: 该函数能够处理不同格式的 response_text,即使没有明确的 "Output:" 标识,也能从整个字符串中提取出输出部分。
# 总的来说,这个函数是 Alpaca 数据集预处理的一个关键步骤,它利用正则表达式和字符串操作等技术,从复杂的响应文本中准确地提取出任务的输入和输出部分,为后续的模型训练做好了准备。
def parse_input_output(response_text):
    # 1. 这个函数的目的是从给定的 response_text 中提取出任务的输入和输出部分。
    # 2. 输入参数 response_text 是一个字符串,表示任务的完整响应文本。

    # 3. 检查 response_text 中是否包含 "Output 序号:" 的模式。
    # 4. 这种模式通常用于表示任务的输出部分。
    if re.findall(r"Output\s*\d*\s*:", response_text):
        
        # 5. 使用正则表达式将 response_text 拆分成输入和输出两部分。
        # 6. inst_input 变量存储任务的输入部分,并去除首尾空白字符。
        inst_input = re.split(r"Output\s*\d*\s*:", response_text)[0].strip()
        # 7. inst_output 变量存储任务的输出部分,并去除首尾空白字符。
        inst_output = re.split(r"Output\s*\d*\s*:", response_text)[1].strip()
    # 8. 如果 response_text 中没有包含 "Output 序号:" 的模式,
    else:
        # 9. 则假定整个 response_text 就是任务的输出部分,将输入部分设为空字符串。
        inst_input = ""
        # 10. 将 response_text 整体作为任务的输出部分,并去除首尾空白字符。
        inst_output = response_text.strip()
    # to avoid the case multiple input/output pairs are generated
    # 11. 为了处理可能出现的多个输入/输出对的情况,进行进一步的清理
    if re.findall(r"Input\s*\d*\s*:", inst_output):
        # 12. 如果 inst_output 中包含 "Input 序号:" 的模式,
        # 13. 则认为这是多个输入/输出对的情况,只取第一个输出部分。
        inst_output = re.split(r"Input\s*\d*\s*:", inst_output)[0].strip()
    # remove the prefix "Input:" from the string
    # 14. 最后,去除 inst_input 中可能存在的 "Input:" 前缀
    inst_input = re.sub(r"^Input\s*\d*\s*:", "", inst_input).strip()
    # 15. 返回提取的任务输入和输出部分
    return inst_input, inst_output


def filter_duplicate_instances(instances):
    # 1. 这个函数的目的是过滤掉 Alpaca 数据集中的重复实例。
    # 2. 输入参数 instances 是一个列表,每个元素都是一个包含 instruction、input 和 output 三个部分的元组。

    # if the instances have same non-empty input, but different output, we will not use such instances
    # 3. 初始化一个布尔变量 same_input_diff_output,用于标记是否存在输入相同但输出不同的实例。
    same_input_diff_output = False
    for i in range(1, len(instances)):  # 4. 遍历 instances 列表中的所有实例,从第二个实例开始。
        # 5. 对于每个实例 i,都与之前的所有实例 j 进行比较。
        for j in range(0, i):
            if instances[i][1] == "":  # 6. 如果当前实例 i 的输入为空字符串,则跳过该实例的比较。
                continue
            # 7. 如果当前实例 i 的输入与之前实例 j 的输入相同,但输出不同,
            if instances[i][1] == instances[j][1] and instances[i][2] != instances[j][2]:
                same_input_diff_output = True   # 8. 则设置 same_input_diff_output 为 True,表示找到了这种情况。
                break   # 9. 一旦找到,立即退出内层循环。
    if same_input_diff_output:  # 10. 如果存在输入相同但输出不同的实例,
        return []  # 11. 则直接返回一个空列表,不保留任何实例。

    # remove duplicate instances
    # 13. 使用 set() 函数去除列表中的重复实例,然后再转换回列表。
    instances = list(set(instances))
    return instances

def filter_invalid_instances(instances):
    # 1. 这个函数的目的是过滤掉 Alpaca 数据集中的无效实例。
    # 2. 输入参数 instances 是一个列表,每个元素都是一个包含 instruction、input 和 output 三个部分的元组。
    filtered_instances = []  # 3. 初始化一个空列表 filtered_instances,用于存储过滤后的有效实例。
    for instance in instances:   # 4. 遍历输入的 instances 列表中的每个实例。
        # if input and output are the same, we will not use such instances
        if instance[1] == instance[2]:  # 5. 如果实例的输入和输出部分完全相同,则跳过该实例,不加入到 filtered_instances 中。
            continue
        # if output is empty, we will not use such instances
        if instance[2] == "":  # 6. 如果实例的输出部分为空字符串,则跳过该实例,不加入到 filtered_instances 中。
            continue
        # if input or output ends with a colon, these are usually imcomplete generation. We will not use such instances
        # 7. 如果实例的输入或输出部分以冒号结尾,通常表示生成不完整,则跳过该实例,不加入到 filtered_instances 中。
        if instance[1].strip().endswith(":") or instance[2].strip().endswith(":"):
            continue
        filtered_instances.append(instance)  # 8. 如果实例通过了上述检查,则将其加入到 filtered_instances 列表中。
    return filtered_instances

def parse_instances_for_generation_task(raw_text, instruction, response_metadata):
    # 10. 这个函数的目的是从给定的 raw_text 中提取出用于生成任务的实例。
    # 11. 输入参数包括:
    #     - raw_text: 原始的响应文本
    #     - instruction: 任务的指令
    #     - response_metadata: 响应的元数据,包含生成完成的原因等信息。
    instances = []   # 12. 初始化一个空列表 instances,用于存储提取出的实例。
    raw_text = raw_text.strip()   # 13. 对输入的 raw_text 进行首尾空白字符的去除。
     # 14. 检查 raw_text 中是否包含 "Example 序号." 的模式,这是一种常见的实例分隔方式。
    if re.findall("Example\s?\d*\.?", raw_text):
        # 15. 使用正则表达式将 raw_text 按照实例分隔符拆分成多个实例文本。
        instance_texts = re.split(r"Example\s?\d*\.?", raw_text)
        # 16. 对拆分出的每个实例文本,去除首尾空白字符,并排除空字符串。
        instance_texts = [it.strip() for it in instance_texts if it.strip() != ""]
        # 17. 遍历每个实例文本,调用 parse_input_output 函数提取输入和输出部分,
        for instance_text in instance_texts:
             # 18. 然后将 instruction、input 和 output 组成一个元组,添加到 instances 列表中。
            inst_input, inst_output = parse_input_output(instance_text)
            instances.append((instruction.strip(), inst_input.strip(), inst_output.strip()))
    # 19. 如果 raw_text 中包含 "Output 序号:" 的模式,则假定只有一个输入输出对。
    elif re.findall(r"Output\s*\d*\s*:", raw_text):
        # we assume only one input/output pair in this case
        # 20. 调用 parse_input_output 函数提取输入和输出部分,
        inst_input, inst_output = parse_input_output(raw_text)
        instances.append((instruction.strip(), inst_input.strip(), inst_output.strip()))  # 21. 然后将其添加到 instances 列表中。
    else:
        return []  # 22. 如果 raw_text 中没有找到任何实例分隔模式,则返回一个空列表。
    
    # if the generation stops because of length, we remove the last instance
    # 23. 如果响应被截断(因为达到了最大长度限制),则删除最后一个实例
    if response_metadata["response"]["choices"][0]["finish_reason"] == "length":
        instances = instances[:-1]  # 24. 当响应被截断时,最后一个实例可能是不完整的,因此将其从 instances 列表中删除。
    # 25. 调用 filter_invalid_instances 函数,过滤掉无效的实例。
    instances = filter_invalid_instances(instances)
    # 26. 调用 filter_duplicate_instances 函数,过滤掉重复的实例。
    instances = filter_duplicate_instances(instances)
    return instances

def parse_instances_for_classification_task(raw_text, instruction, response_metadata):
    # 28. 这个函数的目的是从给定的 raw_text 中提取出用于分类任务的实例。
   
    instances = []   # 30. 初始化一个空列表 instances,用于存储提取出的实例。
    if not "Class label:" in raw_text:  # 31. 如果 raw_text 中没有包含 "Class label:" 字符串,则返回一个空列表。
        return []
    instance_texts = raw_text.split("Class label:")[1:]  # 32. 使用 "Class label:" 作为分隔符,将 raw_text 拆分成多个实例文本。
    for instance_text in instance_texts:   # 33. 遍历每个实例文本,
        instance_text = instance_text.strip()  # 34. 去除首尾空白字符。
        fields = instance_text.split("\n", 1)  # 35. 使用换行符 "\n" 将实例文本拆分成两个字段,第一个字段为类别标签,第二个字段为输入文本。
        if len(fields) == 2:  # 36. 如果成功拆分成两个字段,
            # the first field split by \n is the class label
            class_label = fields[0].strip()  # 37. 则将第一个字段作为类别标签,去除首尾空白字符。
            # the rest is the input
            input_text = fields[1].strip()  # 38. 将第二个字段作为输入文本,去除首尾空白字符。
        elif len(fields) == 1:  # 39. 如果只拆分成一个字段,则认为整个字段就是类别标签,输入文本为空字符串。
            # the first field split by \n is the input
            class_label = fields[0].strip()
            input_text = ""
        else:
            # 40. 如果拆分字段的数量不对,则抛出异常。
            raise ValueError("Invalid instance text: {}".format(instance_text))
        # 41. 将 instruction、input_text 和 class_label 组成一个元组,添加到 instances 列表中。
        instances.append((instruction.strip(), input_text.strip(), class_label.strip()))

    # if the generation stops because of length, we remove the last instance
    # 42. 如果响应被截断(因为达到了最大长度限制),则删除最后一个实例
    if response_metadata["response"]["choices"][0]["finish_reason"] == "length":
        instances = instances[:-1]
        # 43. 当响应被截断时,最后一个实例可能是不完整的,因此将其从 instances 列表中删除。
    instances = filter_invalid_instances(instances)  # 44. 调用 filter_invalid_instances 函数,过滤掉无效的实例。
    instances = filter_duplicate_instances(instances)  # 45. 调用 filter_duplicate_instances 函数,过滤掉重复的实例。
    return instances  # 46. 返回过滤后的有效实例列表。


if __name__ == "__main__":
    args = parse_args()

    print("***** Parmeters *****".rjust(62))
    arguments = vars(args)
    for key, value in arguments.items():
        print(f"{key}".rjust(50)+ f": {value}")
    print("***** Parmeters *****".rjust(62))    

    training_instances = []   # 3. 初始化一个空列表 training_instances,用于存储最终准备好的训练实例。
    
    generated_tasks = []  # 4. 初始化一个空列表 generated_tasks,用于存储原始的生成任务。
    for instance_file in args.instance_files:  # 5. 遍历命令行参数中指定的所有实例文件。
        with open(instance_file) as fin:
            for line in fin:
                # 7. 将每一行的 JSON 数据解码成 Python 字典,并添加到 generated_tasks 列表中。
                generated_tasks.append(json.loads(line))
    print(f"Loaded {len(generated_tasks)} raw generated tasks")   # 8. 打印已加载的原始生成任务数量。

    task_clf_types = {}   # 9. 初始化一个字典 task_clf_types,用于存储每个任务是否为分类任务的标记。
    for file in args.classification_type_files:
         # 10. 遍历命令行参数中指定的所有分类任务类型文件。
        with open(file) as fin:
            for line in fin: # 11. 打开每个分类任务类型文件并逐行读取。
                data = json.loads(line)
                # 12. 将每一行的 JSON 数据解码成 Python 字典,并将任务指令和是否为分类任务的标记存储到 task_clf_types 字典中。
                task_clf_types[data["instruction"]] = data["is_classification"].strip() in ["Yes", "yes", "YES"]

    # 13. 遍历所有原始的生成任务。
    # 14. 使用 tqdm 库显示进度条,以便监控处理过程。
    for task in tqdm.tqdm(generated_tasks):
        # get instruction
        instruction = task["instruction"]  # 15. 从当前任务字典中获取任务指令。
        task["is_classification"] = task_clf_types[instruction] # 16. 根据之前构建的 task_clf_types 字典,设置当前任务是否为分类任务的标记。

        # get the instances
        if task["is_classification"]:  # 17. 如果当前任务是分类任务,
            # 18. 则调用 parse_instances_for_classification_task 函数提取分类任务的实例。
            task_instances = parse_instances_for_classification_task(task["raw_instances"], instruction, task["instance_metadata"])
        else: # 19. 如果当前任务不是分类任务,
            # 20. 则调用 parse_instances_for_generation_task 函数提取生成任务的实例。
            task_instances = parse_instances_for_generation_task(task["raw_instances"], instruction, task["instance_metadata"])

        # we only allow max 5 instances per task
        # 21. 为了限制每个任务的实例数量,随机选择最多 5 个实例。
        task_instances = random.sample(task_instances, min(len(task_instances), 5))
        
        if not task_instances: # 22. 如果当前任务没有有效的实例,则跳过该任务
            continue
        # 23. 将当前任务的有效实例添加到 training_instances 列表中。
        training_instances += task_instances

    os.makedirs(args.output_dir, exist_ok=True) # 24. 创建命令行参数指定的输出目录,如果目录已存在则不会报错。
    # 25. 打开一个名为 "all_generated_instances.jsonl" 的输出文件。
    with open(os.path.join(args.output_dir, "all_generated_instances.jsonl"), "w") as fout:
        for instance in training_instances:  # 26. 遍历 training_instances 列表中的所有实例,
            # 27. 将每个实例的指令、输入和输出以 JSON 格式写入输出文件。
            fout.write(json.dumps({
                "instruction": instance[0],
                "input": instance[1],
                "output": instance[2],
            }) + "\n")
    print(f"Saved {len(training_instances)} instances")   # 28. 打印已保存的实例数量。
    # 29. 从 training_instances 列表中提取出所有唯一的任务指令,并存储在 unique_instructions 集合中。
    unique_instructions = set([it[0] for it in training_instances])  
    print(f"Unique instructions: {len(unique_instructions)}")  # 30. 打印唯一任务指令的数量。
    # 31. 从 unique_instructions 中筛选出分类任务的指令,存储在 clf_instructions 列表中。
    clf_instructions = [instruction for instruction in unique_instructions if task_clf_types[instruction]]
    print(f"Classification instructions: {len(clf_instructions)}")  # 32. 打印分类任务指令的数量。
    # 33. 从 unique_instructions 中筛选出非分类任务的指令,存储在 non_clf_instructions 列表中。
    non_clf_instructions = [instruction for instruction in unique_instructions if not task_clf_types[instruction]]
    print(f"Non-classification instructions: {len(non_clf_instructions)}")  # 34. 打印非分类任务指令的数量。

    if args.num_instructions is not None:   # 35. 如果命令行参数中指定了 num_instructions,
        print(f"Sampling {args.num_instructions} instructions")  # 36. 则打印采样的任务指令数量。
        # 37. 从 unique_instructions 中随机采样 args.num_instructions 个指令,存储在 sampled_instructions 列表中。
        sampled_instructions = random.sample(unique_instructions, args.num_instructions)
        # 38. 从 training_instances 中筛选出仅包含 sampled_instructions 中的任务指令的实例,更新 training_instances 列表。
        training_instances = [it for it in training_instances if it[0] in sampled_instructions]
        print(f"Only using {len(training_instances)} instances for these sampled instructions.")  # 39. 打印最终用于训练的实例数量。
        with open(os.path.join(args.output_dir, f"sampled_generated_instances_{args.num_instructions}.jsonl"), "w") as fout:
            # 40. 创建一个新的输出文件 "sampled_generated_instances_<num_instructions>.jsonl",
            for instance in training_instances:
                # 41. 并将筛选后的训练实例以 JSON 格式写入该文件。
                fout.write(json.dumps({
                    "instruction": instance[0],
                    "input": instance[1],
                    "output": instance[2],
                }) + "\n")

    if args.include_seed_tasks:  # 42. 如果命令行参数中指定了 include_seed_tasks,
        seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]  # 43. 则从命令行参数中指定的种子任务文件中读取种子任务,并解码成 Python 字典。
        for task in seed_tasks:  # 44. 遍历所有种子任务,
            for instance in task["instances"]:
                # 45. 将每个种子任务的实例添加到 training_instances 列表中。
                training_instances.append((task["instruction"], instance["input"], instance["output"]))
        print(f"Included {len(seed_tasks)} seed tasks")  # 46. 打印已包含的种子任务数量。

    # get the prompt and completion for training gpt3
    # 47. 初始化一个空列表 gpt3_instances,用于存储最终准备好的 GPT-3 微调数据。
    gpt3_instances = []
    for instance in training_instances:  # 48. 遍历 training_instances 列表中的所有实例,
        # get input and do preprocessing
        inst_input = instance[1]   # 49. 获取当前实例的输入部分。
        # for some tasks, we check whether the input contains colon, and if so, we remove the part before the colon
        if random.random() < 0.5:  # 50. 以 50% 的概率执行以下预处理操作:
            colon_words = re.findall(r"(\w+):", inst_input)   # 51. 使用正则表达式匹配输入中的冒号前的单词,存储在 colon_words 列表中。
            # if only one colon is found, we assume the instance only have one input and we remove the field name before the colon
            if len(set(colon_words)) == 1:   # 52. 如果 colon_words 列表中只有一个唯一的单词,
                inst_input = inst_input.split(":", 1)[1].strip()  # 53. 则认为输入中只有一个字段,将冒号前的部分删除。
            else:  # 54. 如果 colon_words 列表中有多个不同的单词,则保留原始输入。
                inst_input = inst_input.strip()
            # we also replace two consecutive new lines with one new line half of the time
            # 55. 另外,以 50% 的概率将输入中连续的两个换行符替换为一个换行符。
            inst_input = inst_input.replace("\n\n", "\n")
        # 56. 使用 encode_instance 函数将当前实例的指令、输入和输出编码成 GPT-3 微调所需的格式,加入到 gpt3_instances 列表中。
        gpt3_instances.append(encode_instance(instance[0], inst_input, instance[2]))

    # remove duplicates
    filtered_instances = []  # 57. 初始化一个空列表 filtered_instances,用于存储去重后的 GPT-3 微调实例。
    prompt_completion_set = set()   # 58. 初始化一个集合 prompt_completion_set,用于存储已处理过的提示-完成对。
    for instance in gpt3_instances:   # 59. 遍历 gpt3_instances 列表中的所有实例,
        # 60. 将当前实例的提示和完成部分组成一个元组。
        instance_pair = (instance["prompt"], instance["completion"])
        # 61. 如果该提示-完成对不在 prompt_completion_set 集合中,
        if instance_pair not in prompt_completion_set:
            # 62. 则将其添加到集合中,并将当前实例添加到 filtered_instances 列表中。
            prompt_completion_set.add((instance["prompt"], instance["completion"]))
            filtered_instances.append(instance)
    # 63. 更新 gpt3_instances 列表为去重后的实例列表。
    gpt3_instances = filtered_instances

    # shuffle
    random.shuffle(gpt3_instances) # 64. 对 gpt3_instances 列表进行随机排序。
    # 65. 创建一个新的输出文件 "gpt3_finetuning_data_<num_instances>.jsonl",
    with open(os.path.join(args.output_dir, f"gpt3_finetuning_data_{len(gpt3_instances)}.jsonl"), "w") as fout:
        for instance in gpt3_instances:
            # 66. 并将 gpt3_instances 列表中的所有实例以 JSON 格式写入该文件。
            fout.write(json.dumps({
                "prompt": instance["prompt"],
                "completion": instance["completion"],
            }) + "\n")