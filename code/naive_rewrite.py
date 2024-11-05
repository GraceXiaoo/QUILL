# -*- coding: utf-8 -*-
from utils.utils import *
from app.app_compute import *
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os

def pipeline(model_info, data_info,prompting):
    llm = model_info['model']
    tokenizer = model_info['tokenizer']
    sampling_params = model_info['sampling_params']    
    query = data_info['挖空语料-插入点']
    golden_author = data_info['作者']
    english = data_info['语言类别']
    #大模型改写
    if english == '英文':
        try:
            print('English rewrite')
            with open(f'/QUILL/code/prompt/prompt_ch_naive_{prompting}_en.md', 'r') as file:
                prom_rewrite = file.read()
            prompt_rewrite = get_prompt(prom_rewrite,query)
            ans_res=get_response(prompt_rewrite, tokenizer, sampling_params, llm)
            print('LLM output:',ans_res)
        except:
            ans_res={'output':'nan','quote':'nan'}
            print('LLM output:',ans_res)
        print('Before Extract:',ans_res)
        ans_res = extract_quote(ans_res)
        print('After Extract:',ans_res)  
        if ans_res != 'nan':    
            ans_res = get_dict(query,ans_res,1)
        else:
            ans_res={'output':'nan','quote':'nan'}
        try:
            quote=ans_res["quote"]
            ans_rewrite=ans_res["output"]
        except:
            print(f'English Output Error')
            return 'nan', ans_res, 'nan','nan', 'nan', 'nan', 'nan'
    else:
        try:
            print('Chinese rewrite')
            with open(f'/QUILL/code/prompt/prompt_ch_naive_{prompting}.md', 'r') as file:
                prom_rewrite = file.read()
            prompt_rewrite = get_prompt(prom_rewrite,query)
            ans_res=get_response(prompt_rewrite, tokenizer, sampling_params, llm).replace('[Q]', f'')
            print('LLM output:',ans_res)
        except:
            ans_res={'输出文本':'nan','引言':'nan'}
            print('LLM output:',ans_res)
        print('Before Extract',ans_res)
        ans_res = extract_quote(ans_res)
        print('After Extract',ans_res)  
        if ans_res!= 'nan':        
            ans_res = get_dict(query,ans_res,en=0)
        else:
            ans_res={'输出文本':'nan','引言':'nan'}
        try:
            quote=ans_res['引言']
            ans_rewrite=ans_res['输出文本']
        except Exception as e:
            print(e)
            print('Chinese Output Error')
            return 'nan', ans_res, 'nan','nan', 'nan', 'nan', 'nan'
    return quote, ans_rewrite

def main(args):
    model_name = args.model_name
    tensor_parallel_size=args.tensor_parallel_size
    gpu_memory_utilization=args.gpu_memory_utilization
    sampling_params = SamplingParams(temperature=0, top_p=0.8, repetition_penalty=1.05, max_tokens=1000)
    if model_name =='qwen1.5-7b-chat':
        model_path='/Qwen/Qwen1.5-7B-Chat'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = LLM(model=model_path,tensor_parallel_size=tensor_parallel_size,gpu_memory_utilization=gpu_memory_utilization)
    elif model_name=='qwen1.5-14b-chat':
        model_path="/Qwen/Qwen1.5-14B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = LLM(model=model_path,tensor_parallel_size=tensor_parallel_size,gpu_memory_utilization=gpu_memory_utilization)
    elif model_name=='qwen1.5-110b-chat':
        model_path="/Qwen/Qwen1.5-110B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = LLM(model=model_path,tensor_parallel_size=tensor_parallel_size,gpu_memory_utilization=gpu_memory_utilization)
    elif model_name=='chatglm3-6b':
        model_path='/chatglm3-6b'
        tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=True,trust_remote_code=True)
        llm = LLM(model=model_path,tensor_parallel_size=tensor_parallel_size,gpu_memory_utilization=gpu_memory_utilization,trust_remote_code=True)
    elif model_name=='llama2-7b-chat-hf':
        model_path = '/models--meta-llama--Llama-2-7b-chat-hf/snapshots/01622a9d125d924bd828ab6c72c995d5eda92b8e'
        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        llm = LLM(model=model_path,tensor_parallel_size=tensor_parallel_size,gpu_memory_utilization=gpu_memory_utilization,trust_remote_code=True)
    elif model_name=='llama2-13b-chat-hf':
        model_path = '/models--meta-llama--Llama-2-13b-chat-hf/snapshots/0ba94ac9b9e1d5a0037780667e8b219adde1908c'
        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        llm = LLM(model=model_path,tensor_parallel_size=tensor_parallel_size,gpu_memory_utilization=gpu_memory_utilization,trust_remote_code=True)
    elif model_name=='llama2-70b-chat-hf':
        model_path='/Llama-2-70b-chat-hf'
        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        llm = LLM(model=model_path,tensor_parallel_size=tensor_parallel_size,gpu_memory_utilization=gpu_memory_utilization,trust_remote_code=True)
    elif model_name=='Mixture-7b-v0.2':
        model_path = '/Mixtral-4x7b-chat'
        tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=True,trust_remote_code=True)
        llm = LLM(model=model_path,tensor_parallel_size=tensor_parallel_size,gpu_memory_utilization=gpu_memory_utilization,trust_remote_code=True)
    elif model_name=='chatgpt-3.5-turbo':
        tokenizer=llm='chatgpt'
    elif model_name=='llama2-70b-chat':
        tokenizer=llm='llama2-70b-chat'
    elif model_name=='qwen1.5-72b-chat':
        tokenizer=llm='qwen1.5-72b-chat'
    else:
        print("Model ERROR")
    
    model_info={
        'tokenizer':tokenizer,
        'model':llm,
        'sampling_params':sampling_params
    }
    file_name=args.file_name
    prompting=args.prompt
    file_path = f'QUILL/data/dev/{file_name}.xlsx'
    df = pd.read_excel(file_path)

    rec_quotes=[]
    ans_rewrite_list=[]
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        quote, ans_rewrite = pipeline(model_info, row,prompting)
        rec_quotes.append(quote)
        ans_rewrite_list.append(ans_rewrite)
    df['rec_quotes']=rec_quotes
    df['ans_rewrite_list']=ans_rewrite_list
    file_path = f'QUILL/data/eval/{model_name}/naive_res_{file_name}_{model_name}_{prompting}.xlsx'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_excel(file_path, index=False)
    print("The new Excel file is saved!!!")


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="QUILL Pipeline")

    parser.add_argument('--model_name', type=str, required=True, help="LLM name")
    parser.add_argument('--file_name', type=str, required=True, help='Your file name')
    parser.add_argument('--tensor_parallel_size', type=int, required=True, help='vllm parameter1 ')
    parser.add_argument('--gpu_memory_utilization', type=float, required=True, help='vllm parameter2')
    parser.add_argument('--prompt', type=str, required=True, help='prompt')
    args = parser.parse_args()

    main(args)

    