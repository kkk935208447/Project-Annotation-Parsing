# 导入必要的库
import openai
import pandas as pd

# 设置OpenAI API密钥
openai.api_key = "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"

# 定义一个函数，用于处理问题和CSV文件
def csv_prompter(question,csv_name):
    # 创建一个包含问题的提示文本
    fulltext = "A question is provided below. Given the question, extract " + \
               "keywords from the text. Focus on extracting the keywords that we can use " + \
               "to best lookup answers to the question. \n" + \
               "---------------------\n" + \
               "{}\n".format(question) + \
               "---------------------\n" + \
               "Provide keywords in the following comma-separated format.\nKeywords: "
    # 创建消息列表，包含系统消息和用户提示
    messages = [
        {"role": "system", "content": ""},
    ]
    messages.append(
        {"role": "user", "content": f"{fulltext}"}
    )
    # 使用OpenAI的ChatCompletion模型生成关键词
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    keyword_list = rsp.get("choices")[0]["message"]["content"]
    keyword_list = keyword_list.split(", ")

    print(keyword_list)

    # 读取CSV文件
    df = pd.read_csv(csv_name)
    divided_text = []
    # 将CSV数据分成多个文本块
    csvdata = df.to_dict('records')
    step_length = 15
    for csv_item in range(0,len(csvdata),step_length):
        csv_text = str(csvdata[csv_item:csv_item+step_length]).replace("}, {", "\n\n").replace("\"", "")#.replace("[", "").replace("]", "")
        divided_text.append(csv_text)

    answer_llm = ""
    # 初始化文本块的得分列表
    score_textlist = [0] * len(divided_text)
    # 计算每个文本块与关键词的匹配得分
    for i, chunk in enumerate(divided_text):
        for t, keyw in enumerate(keyword_list):
            if keyw.lower() in chunk.lower():
                score_textlist[i] = score_textlist[i] + 1

    answer_list = []
    # 根据得分对文本块进行排序
    divided_text = [item for _, item in sorted(zip(score_textlist, divided_text), reverse=True)]

    for i, chunk in enumerate(divided_text):

        if i>5:
            continue
        # 创建用于提问的文本块
        fulltext = "{}".format(chunk) + \
                   "\n---------------------\n" + \
                   "Based on the Table above and not prior knowledge, " + \
                   "Select the Table Entries that will help to answer the question: {}\n Output in the format of \" Disease: <>; Symptom: <>; Medical Test: <>; Medications: <>;\". If there is no useful form entries, output: 'No Entry'".format(question)

        print(fulltext)
        # 使用OpenAI的ChatCompletion模型生成答案
        messages = [
            {"role": "system", "content": ""},
        ]
        messages.append(
            {"role": "user", "content": f"{fulltext}"}
        )
        rsp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        answer_llm = rsp.get("choices")[0]["message"]["content"]

        print("\nAnswer: " + answer_llm)
        print()
        if not "No Entry" in answer_llm:
            answer_list.append(answer_llm)


    # 创建用于提问的文本块
    fulltext = "The original question is as follows: {}\n".format(question) + \
               "Based on this Table:\n" + \
               "------------\n" + \
               "{}\n".format(str("\n\n".join(answer_list))) + \
               "------------\n" + \
               "Answer: "
    print(fulltext)
    # 使用OpenAI的ChatCompletion模型生成最终答案
    messages = [
        {"role": "system", "content": ""},
    ]
    messages.append(
        {"role": "user", "content": f"{fulltext}"}
    )
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    answer_llm = rsp.get("choices")[0]["message"]["content"]

    print("\nFinal Answer: " + answer_llm)
    print()

    return answer_llm

question = "If I have frontal headache, fever, and painful sinuses, what disease should I have, and what medical test should I take?"
csv_name = r"C:\Users\dzysunshine\Desktop\相关资料\ChatDoctor\Autonomous_ChatGPT_API\disease_database_mini.csv"
FinalAnswer=csv_prompter(question,csv_name)
print(FinalAnswer)
