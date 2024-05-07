# 导入必要的库
import torch
from llama_index import WikipediaReader

# 定义一个函数，用于将文本分成多个部分，每部分包含指定数量的单词
def divide_string(wiki_page, word_limit=50):
    divided_text = []
    for each_page in wiki_page:
      words = each_page[0].text.split()
      
      for i in range(0, len(words), word_limit):
          chunk = ' '.join(words[i:i+word_limit])
          divided_text.append(chunk)
    
    return divided_text
    
    
# 定义一个函数，用于处理问题并生成答案
def wiki_prompter(generator,tokenizer,question):
    
    # 创建一个包含问题的提示文本
    fulltext = "A question is provided below. Given the question, extract " +\
    "keywords from the text. Focus on extracting the keywords that we can use " +\
    "to best lookup answers to the question. \n" +\
    "---------------------\n" +\
    "{}\n".format(question) +\
    "---------------------\n" +\
    "Provide keywords in the following comma-separated format.\nKeywords: "
    
    # 使用Tokenizer将文本转换为模型输入
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()


    with torch.no_grad():
        # 使用Generator生成答案
        generated_ids = generator(
            gen_in,
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.1,  # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
            temperature=0.5,  # default: 1.0
            top_k=50,  # default: 50
            top_p=1.0,  # default: 1.0
            early_stopping=True,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  # for some reason, batch_decode returns an array of one element?
        text_without_prompt = generated_text[len(fulltext):]
    response = text_without_prompt
    response = response.split("===")[0]
    response.strip()
    print(response)
    keywords = response.split(", ")
    print(keywords)
    
    wiki_docs=[]
    for keyw in keywords:
        try:
            # 使用WikipediaReader加载关键词相关的Wikipedia页面
            wiki_one = WikipediaReader().load_data(pages=[keyw], auto_suggest=False)
            wiki_docs.append(wiki_one)
        except:
            print("No wiki: "+keyw)
            
    # 将Wikipedia页面分成多个文本块
    divided_text = divide_string(wiki_docs, 250)    
    
    answer_llama=""
    # 初始化文本块的得分列表
    score_textlist = [0] * len(divided_text)

    # 计算每个文本块与关键词的匹配得分
    for i, chunk in enumerate(divided_text):
        for t, keyw in enumerate(keywords):
            if keyw.lower() in chunk.lower():
                score_textlist[i]=score_textlist[i]+1
                 
    answer_list=[]        
    divided_text = [item for _, item in sorted(zip(score_textlist, divided_text), reverse=True)]
    divided_text.append("_")
    for i, chunk in enumerate(divided_text):
        if i<4 and not i==int(len(divided_text)-1):
            fulltext = "Context information is below. \n" +\
            "---------------------\n" +\
            "{}".format(chunk) +\
            "\n---------------------\n" +\
            "Given the context information and not prior knowledge, " +\
            "answer the question: {}\n".format(question) +\
            "Response: "
        elif i==int(len(divided_text)-1) and len(answer_list)>1 :
            fulltext = "The original question is as follows: {}\n".format(question) +\
            "We have provided existing answers:\n" +\
            "------------\n" +\
            "{}\n".format(str("\n\n".join(answer_list))) +\
            "------------\n" +\
            "The best one answer: "
        else:
            continue
          
        print(fulltext)
        # 使用Tokenizer将文本转换为模型输入
        gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
    
    
        with torch.no_grad():
            generated_ids = generator(
                gen_in,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1,  # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=0.5,  # default: 1.0
                top_k=50,  # default: 50
                top_p=1.0,  # default: 1.0
                early_stopping=True,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  
            text_without_prompt = generated_text[len(fulltext):]
        
        
        answer_llama = text_without_prompt        
        print()
        print("\nAnswer: " + answer_llama)
        print()
        answer_list.append(answer_llama)
    
    return answer_llama


