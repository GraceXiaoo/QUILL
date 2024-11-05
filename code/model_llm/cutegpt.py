# -*- coding: utf-8 -*-
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import openpyxl
from tqdm import tqdm
overall_instruction = "你是复旦大学知识工场实验室训练出来的语言模型CuteGPT。给定任务描述，请给出对应请求的回答。\n"

model_name = "/data/dell/hqx/ckp/llama_13b_112_sft_v1"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
print('additional_special_token:', tokenizer.additional_special_tokens)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
device = torch.device("cuda")

def get_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
    input_ids = input_ids["input_ids"].to(device)

    with torch.no_grad():
        outputs=model.generate(
                input_ids=input_ids,
                top_p=0.8,
                top_k=50,
                repetition_penalty=1.1,
                max_new_tokens = 512,
                early_stopping = True,
                eos_token_id = tokenizer.convert_tokens_to_ids('<end>'),
                pad_token_id = tokenizer.eos_token_id,
                min_length = input_ids.shape[1] + 1
        )
    s = outputs[0][input_ids.shape[1]:]
    response=tokenizer.decode(s)
    res = response.replace('<s>', '').replace('<end>', '').replace('</s>', '')
    return res