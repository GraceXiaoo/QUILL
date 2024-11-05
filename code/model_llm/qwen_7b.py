import time
import os
import random 
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

#transformers 4.40.1
#transformers>=4.32.0,<4.38.0
#pip uninstall transformer-engine
#pip install transformers_stream_generator
# Note: The default behavior now has injection attack prevention off.
model_path="/data1/dcy/downloads/model/Qwen/Qwen1.5-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
model = model.eval()#评估模式
device=torch.device('cuda')

def get_response(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=4096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return res

def compute_ppl(text):
    input_ids=tokenizer.encode(text,return_tensors='pt').to(device)
    #print(input_ids)
    #print(input_ids.shape)
    #序列长度
    seq_len=input_ids.shape[1]
    #滑步窗口
    stride = 5
    nlls = [] 
    prev_end_loc = 0 
    max_length = model.config.seq_length
    for begin_loc in tqdm( range ( 0 , seq_len, stride)): 
        end_loc = min (begin_loc + max_length, seq_len) 
        trg_len = end_loc - prev_end_loc   # 可能与上一个循环的步幅不同
        input_ids = input_ids[:, begin_loc:end_loc].to(device) 
        target_ids = input_ids.clone() 
        #抛除上下文标记的对数似然损失
        target_ids[:, :-trg_len] = - 100 
        with torch.no_grad():
            outputs=model(input_ids,output_hidden_states=True,labels=target_ids)#获取概率
            print(output)
            neg_log_likelihood =outputs.loss
            print(neg_log_likelihood)
        #print(outputs)
        nlls.append(neg_log_likelihood) 
        prev_end_loc = end_loc 
        if end_loc == seq_len:
            break 
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
