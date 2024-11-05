from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import random 
from tqdm import tqdm
import pandas as pd
from transformers.generation import GenerationConfig
from tqdm import tqdm
import torch
#最新的transformers不太行 4.30.2
#
#加载模型
#
model_path="/data1/dcy/downloads/model/01-ai/Yi-6B-Chat"
#model_path="/data1/dcy/downloads/model/Qwen/Qwen1.5-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).cuda().eval()
device=torch.device('cuda:7')
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
    max_length = model.config.max_length
    for begin_loc in tqdm( range ( 0 , seq_len, stride)): 
        end_loc = min (begin_loc + max_length, seq_len) 
        trg_len = end_loc - prev_end_loc   # 可能与上一个循环的步幅不同
        input_ids = input_ids[:, begin_loc:end_loc].to(device) 
        target_ids = input_ids.clone() 
        #抛除上下文标记的对数似然损失
        target_ids[:, :-trg_len] = - 100 
        with torch.no_grad():
            # Prompt content: "hi"
            outputs=model(input_ids,output_hidden_states=True,labels=target_ids)#获取概率
            neg_log_likelihood =outputs.loss 
            print(neg_log_likelihood)
        #print(outputs)
        nlls.append(neg_log_likelihood) 
        prev_end_loc = end_loc 
        if end_loc == seq_len:
            break 
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
