import json
import pandas as pd
from openai import OpenAI

api_key= '[Your api key]'
model = 'gpt-4o'
client = OpenAI(api_key=api_key)

with open('QUILL/code/prompt/prompt_4_get_rel.md','r',encoding='utf-8') as f:
    prompt = f.read()

def get_prompt(prompt,answer,query,quote):
    if '{answer}' in prompt:
        prompt = prompt.replace('{answer}',answer)
    if '{query}' in prompt:
        prompt = prompt.replace('{query}',query)
    if '{quote}' in prompt:
        prompt = prompt.replace('{quote}',quote)
    return prompt

def get_list(answer,query,quote_list):
    prompt_ = get_prompt(prompt,answer,query,quote_list)
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": prompt_}
    ]
    )
    answer = completion.choices[0].message.content
    return eval(answer)

if __name__ == '__main__':
    filepath = '/QUILL/data/eval/ablation/res_{file_name}_{ppl_fun}.xlsx'
    df = pd.read_excel(filepath,engine='openpyxl')
    golden_quote = df['引言']
    query = df['挖空语料-插入点']
    quote_list = df['rerank_all'].to_list()
    dict = [{'golden_quote':a,'query':p,'quote':q} for a,p,q in zip(golden_quote,query,quote_list)]
    rel_list =[]
    for i in dict:
        rel = get_list(i['golden_quote'],i['query'],i['quote'])
        rel_list.append(rel)
        print(rel)
    df['rel'] = rel_list
    with open('/QUILL/data/Search_dict/Search_quote_rel.json','w',encoding='utf-8') as f:
        json.dump(rel_list,f,ensure_ascii=False)