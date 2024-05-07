from transformers import AutoTokenizer, OPTForCausalLM
import torch

model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model.eval()
prompt = "I love you"

# 不同的pad长度, casul lm 计算的loss相同吗
# 1.
inputs = tokenizer(prompt, return_tensors="pt",padding="max_length",truncation=True,max_length=7)
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))

# Generate
res = model(**inputs,labels=inputs.input_ids)
print(res.logits.shape,res.loss,end="\n\n++++++++++++++++++++++++++")

# 2.
inputs = tokenizer(prompt, return_tensors="pt",padding="max_length",truncation=True,max_length=6)
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))

# Generate
res = model(**inputs,labels=inputs.input_ids)
print(res.logits.shape,res.loss)
print(res.logits.shape,res.loss,end="\n\n++++++++++++++++++++++++++")
#3.
inputs = tokenizer(prompt, return_tensors="pt",padding="max_length",truncation=True,max_length=5)
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))

# Generate
res = model(**inputs,labels=inputs.input_ids)
print(res.logits.shape,res.loss)
print("="*200)
# 手动算:
inputs = tokenizer(prompt, return_tensors="pt",padding="max_length",truncation=True,max_length=7)
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))

# Generate
res_logits = model(**inputs,labels=inputs.input_ids).logits
print(res_logits.shape)
res_logits = res_logits[:, :-1, :]
labels = inputs.input_ids[...,1:]
print(labels)
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=1)
loss = loss_fct(res_logits.view(-1, 50272), labels.view(-1))
print(loss)