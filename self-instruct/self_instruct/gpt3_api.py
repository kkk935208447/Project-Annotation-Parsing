import json
import tqdm
import os
import random
import openai
from datetime import datetime
import argparse
import time
    
# 这个函数的主要目的是向 OpenAI API 发送请求,生成指定长度的文本。
# 它使用了以下技术:
# 1. 函数定义和参数传递
# 2. 异常处理和重试机制
# 3. OpenAI SDK 集成
# 4. 字典和列表操作
# 5. 字符串操作
# 6. 时间处理
def make_requests(
        engine, prompts, max_tokens, temperature, 
        top_p, 
        frequency_penalty, presence_penalty, 
        stop_sequences, logprobs, n, best_of, 
        retries=3, api_key=None, organization=None
    ):
    # 1. 定义一个函数 make_requests,用于向 OpenAI API 发送请求并获取响应。
    # 2. 该函数接受多个参数,包括:
    #    - engine: 指定要使用的语言模型引擎,如 "text-davinci-003"。
    #    - prompts: 输入的提示文本,可以是单个字符串或字符串列表。
    #    - max_tokens: 生成文本的最大长度。
    #    - temperature: 采样温度,控制生成文本的随机性。
    #    - top_p: 进行 Top-p 采样时的阈值。
    #    - frequency_penalty: 惩罚项,用于减少生成文本中的重复。
    #    - presence_penalty: 惩罚项,用于增加生成文本的多样性。
    #    - stop_sequences: 停止生成的序列列表。
    #    - logprobs: 指定是否返回每个 Token 的对数概率。
    #    - n: 生成多少个完成结果。
    #    - best_of: 从 n 个结果中选择最佳的一个。
    #    - retries: 最大重试次数。
    #    - api_key: OpenAI API 密钥。
    #    - organization: OpenAI 组织 ID。

    # 3. 初始化 response 变量为 None。
    # 4. 将目标长度 target_length 设置为 max_tokens。
    response = None
    target_length = max_tokens
    # 5. 如果提供了 api_key,则将其设置为 OpenAI SDK 的 API 密钥。
    if api_key is not None:
        openai.api_key = api_key
    # 6. 如果提供了 organization,则将其设置为 OpenAI SDK 的组织 ID。
    if organization is not None:
        openai.organization = organization
    # 7. 初始化重试计数器 retry_cnt 为 0。
    # 8. 设置初始回退时间 backoff_time 为 30 秒。
    retry_cnt = 0
    backoff_time = 30
    while retry_cnt <= retries:
        try:
            # 9. 调用 OpenAI SDK 的 Completion.create 方法向 API 发送请求。
            # 10. 传入所有必要的参数,如引擎、提示、生成长度、采样参数等。
            # 11. 如果请求成功,则跳出循环。
            response = openai.Completion.create(
                engine=engine,
                prompt=prompts,
                max_tokens=target_length,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_sequences,
                logprobs=logprobs,
                n=n,
                best_of=best_of,
            )
            break
        except openai.error.OpenAIError as e:
            # 12. 如果出现 OpenAI API 错误,则打印错误信息。
            print(f"OpenAIError: {e}.")
            # 13. 如果错误信息中包含 "Please reduce your prompt",说明提示过长。
            # 14. 将目标长度 target_length 缩小至原来的 80%,并打印相关信息。
            if "Please reduce your prompt" in str(e):
                target_length = int(target_length * 0.8)
                print(f"Reducing target length to {target_length}, retrying...")
            else:
                # 15. 对于其他错误,打印重试信息,并等待一段时间后重试。
                # 16. 每次重试时,回退时间会增加 1.5 倍。
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
            # 17. 重试计数器加 1。
            retry_cnt += 1
    
    if isinstance(prompts, list):
        results = []
        # 18. 如果输入的 prompts 是一个列表,则对每个提示进行处理。
        # 19. 从 API 响应中提取对应的结果,并将提示、结果和创建时间存储在字典中。
        # 20. 将所有字典添加到 results 列表中,最后返回该列表。
        for j, prompt in enumerate(prompts):
            data = {
                "prompt": prompt,
                "response": {"choices": response["choices"][j * n: (j + 1) * n]} if response else None,
                "created_at": str(datetime.now()),
            }
            results.append(data)
        return results
    else:
        # 21. 如果输入的 prompts 是单个字符串,则将提示、结果和创建时间存储在字典中,并将该字典作为列表的单个元素返回。
        data = {
            "prompt": prompts,
            "response": response,
            "created_at": str(datetime.now()),
        }
        return [data]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The input file that contains the prompts to GPT3.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file to save the responses from GPT3.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="The openai GPT3 engine to use.",
    )
    parser.add_argument(
        "--max_tokens",
        default=500,
        type=int,
        help="The max_tokens parameter of GPT3.",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="The temprature of GPT3.",
    )
    parser.add_argument(
        "--top_p",
        default=0.5,
        type=float,
        help="The `top_p` parameter of GPT3.",
    )
    parser.add_argument(
        "--frequency_penalty",
        default=0,
        type=float,
        help="The `frequency_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--presence_penalty",
        default=0,
        type=float,
        help="The `presence_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--stop_sequences",
        default=["\n\n"],
        nargs="+",
        help="The `stop_sequences` parameter of GPT3.",
    )
    parser.add_argument(
        "--logprobs",
        default=5,
        type=int,
        help="The `logprobs` parameter of GPT3"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="The `n` parameter of GPT3. The number of responses to generate."
    )
    parser.add_argument(
        "--best_of",
        type=int,
        help="The `best_of` parameter of GPT3. The beam size on the GPT3 server."
    )
    parser.add_argument(
        "--use_existing_responses",
        action="store_true",
        help="Whether to use existing responses from the output file if it exists."
    )
    parser.add_argument(
        "--request_batch_size",
        default=20,
        type=int,
        help="The number of requests to send to GPT3 at a time."
    )
    return parser.parse_args()


# 这段代码的主要功能是:
# 1. 读取输入文件中的提示,并尝试从已有的响应数据中找到匹配的结果。
# 2. 对于新的提示,调用 make_requests 函数向 OpenAI API 发送请求,获取响应数据。
# 3. 将所有响应数据写入指定的输出文件中。
#
# 它使用了以下主要技术:
# 1. 命令行参数解析
# 2. 文件读写操作
# 3. 字典和列表操作
# 4. JSON 数据处理
# 5. 进度条显示
# 6. 调用外部函数 make_requests 生成响应
#
# 这些技术的使用旨在实现对输入提示的批处理,并将响应数据持久化存储,以供后续使用。该代码可用于自然语言处理任务中,通过向语言模型发送大量提示并收集响应,构建丰富的数据集。
if __name__ == "__main__":
    random.seed(123)
    args = parse_args()

    print("***** Parmeters *****".rjust(62))
    arguments = vars(args)
    for key, value in arguments.items():
        print(f"{key}".rjust(50)+ f": {value}")
    print("***** Parmeters *****".rjust(62))

    # 4. 创建输出文件的目录,如果目录不存在则自动创建。
    # 5. os.path.dirname() 函数用于获取文件路径的目录部分。
    # 6. exist_ok=True 参数表示如果目录已经存在则不会抛出异常。
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # read existing file if it exists
    # 7. 检查输出文件是否已经存在,并且 args.use_existing_responses 参数为 True。
    # 8. 如果满足条件,则打开输出文件并读取其中的数据。
    # 9. 使用 json.loads() 函数将每行 JSON 字符串解析为 Python 字典对象。
    # 10. 将每个提示及其对应的响应数据存储在 existing_responses 字典中。
    existing_responses = {}
    if os.path.exists(args.output_file) and args.use_existing_responses:
        with open(args.output_file, "r") as fin:
            for line in fin:
                data = json.loads(line)
                existing_responses[data["prompt"]] = data

    # do new prompts
    # 11. 打开输入文件并读取其中的所有提示。
    # 12. 如果输入文件是 .jsonl 格式,则从每行 JSON 字符串中提取 "prompt" 字段的值。
    # 13. 如果输入文件是其他格式,则对每行文本进行字符串处理,去除两端空白并将 "\\n" 替换为换行符 "\n"。
    # 14. 将所有提示存储在 all_prompts 列表中。
    with open(args.input_file, "r") as fin:
        if args.input_file.endswith(".jsonl"):
            all_prompts = [json.loads(line)["prompt"] for line in fin]
        else:
            all_prompt = [line.strip().replace("\\n", "\n") for line in fin]

    with open(args.output_file, "w") as fout:
        for i in tqdm.tqdm(range(0, len(all_prompts), args.request_batch_size)):
            # 17. 每次处理 args.request_batch_size 个提示,构建 batch_prompts 列表。
            batch_prompts = all_prompts[i: i + args.request_batch_size]
            # 18. 检查当前批次中的所有提示是否都已经存在于 existing_responses 字典中。
            # 19. 如果全部存在,则直接将对应的响应数据写入输出文件。
            # 20. 使用 json.dumps() 函数将字典对象转换为 JSON 字符串,并换行写入。
            if all(p in existing_responses for p in batch_prompts):
                for p in batch_prompts:
                    fout.write(json.dumps(existing_responses[p]) + "\n")
            else:
                # 21. 如果当前批次中存在新的提示,则调用 make_requests 函数向 OpenAI API 发送请求,生成对应的响应。
                # 22. 传入多个参数,如引擎、提示、生成长度、采样参数等,这些参数的具体含义在前面的注释中已经解释过。
                results = make_requests(
                    engine=args.engine,
                    prompts=batch_prompts,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    stop_sequences=args.stop_sequences,
                    logprobs=args.logprobs,
                    n=args.n,
                    best_of=args.best_of,
                )
                # 23. 遍历 make_requests 函数返回的结果列表,将每个响应数据写入输出文件。
                # 24. 同样使用 json.dumps() 函数将字典对象转换为 JSON 字符串,并换行写入。
                for data in results:
                    fout.write(json.dumps(data) + "\n")