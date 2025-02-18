from utils.search_corpus import Search_fun
from utils.utils import *
from app.app_compute import *
import argparse
import pandas as pd
from tqdm import tqdm
import json
import math
# Open the Search_dict
with open('/QUILL/data/Search_dict/Search_quote_list.json', 'r',encoding='utf-8') as file:
    Search_lst = json.load(file)
Search_dict = {i["quote"]: i["Search"] for i in Search_lst}

## Quotation Authenticity and Quotation Creditibility
def compute_S_a_c(quote,golden_author):
    res = Search_fun(quote)
    if res != '无':
        try:
            quote_judge = 1 if eval(res)['判断'] == '是' else 0
            fact_acc = 1 if eval(res)['作者'] == golden_author else 0
        except Exception as e:
            print(f"Error during eval: {e}")
            quote_judge = 'nan'
            fact_acc = 'nan'
            print('Error : Compute S_a_c')
            return 'nan','nan'
    else:
        quote_judge = 'nan'
        fact_acc = 'nan'
        print('Error : No search result')
        return 'nan','nan'
    S_a = quote_judge
    S_c = fact_acc
    return S_a,S_c

## Semantic Mathcing(Given the previous text and quote,compute ppl of the right string)
def compute_S_m(rewrite,quote):
    try:
        quote_index = rewrite.find(quote)
        if quote_index != -1 and quote_index !=0 :
            ppl = compute_ppl(rewrite[:quote_index + len(quote)],rewrite[quote_index+len(quote):])            
        else:
            print('指标位置在最开头或结尾')
            return 'nan'

        S_m = safe_exp(0.053 * (ppl - 35.243))
        print(ppl)
        print('语义匹配度S_m', S_m)
        return S_m
    except Exception as e:
        print('计算语义匹配度出现问题:', str(e))
        ppl = 'nan'
        return 'nan'

## Semantic Fluency(Compute the ppl of entire sentence)
def compute_S_f(rewrite,quote):
    try:
        ppl = compute_ppl('',rewrite)
        S_f = safe_exp(0.5 * (ppl - 16.47))

        print('语义流畅度S_f', S_f)
        return S_f
    except Exception as e:
        print('计算语义流畅度出现问题:', str(e))
        ppl = 'nan'
        return 'nan'

## Novelty(PPL and Search)
def compute_PPL_q(quote):
    try:
        ppl = compute_ppl('',quote)
        if ppl >= 20:
            ppl = 20
        return ppl
    except:
        return 'nan'
    
def compute_S_n(quote):
    try:
        PPL_q = compute_PPL_q(quote)
        try:
            quote = quote.replace('"','')
        except:
            pass

        if quote in Search_dict:
            SearchFreq = Search_dict[quote]
        novelty = (PPL_q * 5) / math.log10(SearchFreq)
        
        exp_n = safe_exp(-0.253 * (novelty - 10.67))
        S_n = (1 / (1 + exp_n)) 
        print('ppl',PPL_q,'SearchFreq',SearchFreq) 
        print('新颖度S_n', S_n)
        return S_n
    except:
        print('计算S_n出现问题')
        return 'nan'


## Compute
def compute(df):
    author = df['作者']
    rewrite = df['ans_rewrite_list']
    quote = df['rec_quotes']
    print('Quote:',quote)
    S_a,S_c = compute_S_a_c(quote,author)
    S_m = compute_S_m(rewrite,quote)
    S_f = compute_S_f(rewrite,quote)
    S_n = compute_S_n(quote)
    return S_a,S_c,S_m,S_f,S_n

def main(args):
    prompting=args.prompt
    model_name=args.model_name
    file_path = f'QUILL/data/eval/{model_name}/naive_res_extract_quote_author_new_{model_name}_{prompting}.xlsx'
    if model_name == 'ours':
        file_path = '/QUILL/data/eval/ours/res_extract_quote_author_new_avg_novelty.xlsx'
    ## Read
    df = pd.read_excel(file_path)
    S_a_lst=[]
    S_c_lst=[]
    S_m_lst=[]
    S_f_lst=[]
    S_n_lst=[]
    ## Compute
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        S_a,S_c,S_m,S_f,S_n = compute(row)
        S_a_lst.append(S_a)
        S_c_lst.append(S_c)
        S_m_lst.append(S_m)
        S_f_lst.append(S_f)
        S_n_lst.append(S_n)
        print(f'S_a:{S_a} S_c:{S_c} S_m:{S_m} S_f:{S_f} S_n:{S_n}')
    ## List
    df['S_a']=S_a_lst
    df['S_c']=S_c_lst
    df['S_m']=S_m_lst
    df['S_f']=S_f_lst
    df['S_n']=S_n_lst
    ## Save
    df.to_excel(file_path, index=False)
    print('!!!Done!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QUILL Pipeline")

    parser.add_argument('--model_name', type=str, required=True, help="LLM name")
    parser.add_argument('--prompt', type=str, required=True, help='prompt')

    args = parser.parse_args()

    main(args)





