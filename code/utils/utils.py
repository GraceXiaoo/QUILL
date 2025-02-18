import json
import time
import requests
from ..model_llm.chatgpt import get_response as get_response_gpt
import random
import dashscope
from http import HTTPStatus
import math

## Ablation
##Input the dict of Novelty and relevances
with open("QUILL/data/Search_dict/Search_quote_novelty.json", "r", encoding="utf-8") as f:
    data = json.load(f)
quote_novelty_dict = {item["quote"]: item["novelty"] for item in data}
quote_ppl_dict = {item["quote"]: item["PPL"] for item in data}

with open("QUILL/data/Search_dict/Search_quote_rel.json", "r", encoding="utf-8") as f:
    Search_quote_rel = json.load(f)

###FUNCTION
def above_text(sentence):
    quote_index = sentence.find('[Q]')
    if quote_index ==0:
        return ''
    elif quote_index ==-1:
        return sentence.strip()
    else:
        return sentence[:quote_index].strip()
    
def get_novelty(string):
    novelty = None  
    if string in quote_novelty_dict:
        novelty = quote_novelty_dict[string]
        if novelty > 20:
            novelty = 20
    return novelty


## Naive
def get_response(prompt,tokenizer,sampling_params,llm):
    if llm =='chatgpt':
        try:
            res=get_response_gpt(prompt)
            return res
        except Exception as e:
            return e
    messages = [
        {"role": "system", "content": "你是一名文学专家，擅长引经据典相关的任务。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    try:
        outputs = llm.generate([text],sampling_params=sampling_params)
    except Exception as e:
        print(f"Error occurred: {e}")    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

    return generated_text

def get_ali_api_response(prompt):
    time.sleep(5)
    dash_api_keys = ["[Your api key 1]", "[Your api key 2]", "[Your api key 3]"]
    index = random.randint(0, 2)
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.(Please output only one quote)'},
                {'role': 'user', 'content': prompt}]
    response = dashscope.Generation.call(
        api_key= dash_api_keys[index],
        model='llama3-70b-instruct',
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        return  response['output']['choices'][0]['message']['content']
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        return 'nan'
    
def get_dict(query,quote,en=1):
    passage = query.replace('[Q]',quote)
    if '\n' in quote:
        quote = quote.replace('\n', '')
    if en ==1:
        dict = {'quote':quote,'output':passage}
    else:
        dict = {'引言':quote,'输出文本':passage}
    return dict

def get_prompt(prompt,query):
    if '{query}' in prompt:
        prompt = prompt.replace('{query}', query)
    return prompt

def safe_exp(value):
    # 设置阈值来避免指数溢出
    if value > 700:
        return float('inf')  # 超过范围返回正无穷
    elif value < -700:
        return 0  # 太小返回0
    else:
        return math.exp(value)
