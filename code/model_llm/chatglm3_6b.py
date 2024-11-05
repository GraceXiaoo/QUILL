from transformers import AutoTokenizer, AutoModel
import csv
import time
import os
import random 
from tqdm import tqdm
import pandas as pd
from transformers.generation import GenerationConfig
import openpyxl
from tqdm import tqdm
import torch
#最新的transformers不太行，需要安装4.30.2
#加载模型
model_path="/data/cache/huggingface/hub/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
model = model.eval()#评估模式
device=torch.device('cuda')
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
            neg_log_likelihood =outputs.loss 
        #print(outputs)
        nlls.append(neg_log_likelihood) 
        prev_end_loc = end_loc 
        if end_loc == seq_len:
            break 
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
def get_response(prompt):
    res, history = model.chat(tokenizer,prompt, history=[])
    return res
