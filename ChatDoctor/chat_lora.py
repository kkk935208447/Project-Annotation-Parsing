import os

from transformers import AutoModelForCausalLM, LLaMATokenizer
from peft import PeftModel
import torch

def load_model(model_name, LORA_WEIGHTS, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+model_name+"...")

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = LLaMATokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
    generator = model.generate
load_model("/data/sim_chatgpt/llama-7b-hf","/students/julyedu_522454/ChatDoctor/ChatDoctor/lora_models/")

First_chat = "ChatDoctor: I am ChatDoctor, what medical questions do you have?"
print(First_chat)
history = []
history.append(First_chat)

def go():
    invitation = "ChatDoctor: "
    human_invitation = "Patient: "

    # input
    msg = input(human_invitation)
    print("")

    history.append(human_invitation + msg)

    fulltext = "If you are a doctor, please answer the medical questions based on the patient's description. \n\n" + "\n\n".join(history) + "\n\n" + invitation

    generated_text = ""
    gen_in = tokenizer(fulltext, return_tensors="pt")
    gen_in = gen_in.to('cuda:0')
    with torch.no_grad():
        generated_ids = generator(
            **gen_in,
            max_new_tokens=200,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
            temperature=0.5, # default: 1.0
            top_k = 50, # default: 50
            top_p = 1.0, # default: 1.0
            early_stopping=True,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

        text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt

    response = response.split(human_invitation)[0]

    response.strip()

    print(invitation + response)

    print("")

    history.append(invitation + response)

while True:
    go()

