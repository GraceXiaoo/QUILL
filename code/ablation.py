##Ablation 1：Validating QUILL's reranker is useful i.e. ppl1,ppl2,avg,novelty

##vanilla：No rerank i.e. Top1 recalled based on similarity
##ppl1:Compute the following ppl Given the above text
##ppl2:Compute the following ppl Given the above text and first k words of the quote
##Other rerankers：Supervised(BM25、monoT5) Unsupervised(UPR、BGE) GPT(GPT3.5-turbo、GPT4o)

###########
from rag.rag_module import MyVectorDBConnector
from rag.rag_function import retrieval
import pandas as pd
import argparse
from tqdm import tqdm
import os
from eval.rerank_dcg import ndcg_at_k
from eval.rerank_score import mrr_score,hits_at_k
from reranker.chatgpt import gpt_rerank
from reranker.bge import model_bge,bge_rerank
from reranker.upr import model_upr,upr_rerank
from reranker.bm25 import model_bm25,bm25_rerank
from reranker.monoT5 import model_monoT5,monoT5_rerank
from reranker.cal_feature import *
from utils.utils import *
from app.app_compute import *

vector = MyVectorDBConnector(path='QUILL/code/rag/model/quill_final', collection_name='quill_final')

def rerank_fn(reranker,old_context,topk_list,ppl_fun=None):
    try:
        if ppl_fun==None:
            return topk_list
        if ppl_fun==gpt_rerank:
            return gpt_rerank(topk_list,old_context)
        if ppl_fun==bge_rerank:
            return bge_rerank(reranker,topk_list,old_context)
        if ppl_fun==upr_rerank:
            return upr_rerank(reranker,old_context,topk_list)
        if ppl_fun==bm25_rerank:
            return bm25_rerank(reranker,old_context,topk_list)
        if ppl_fun==monoT5_rerank:
            return monoT5_rerank(reranker,old_context,topk_list)
    except Exception as e:
        print('error',e)
        return ['error'*5]
    try:
        if isinstance(topk_list[0],str):
            topk_list[0]=eval(topk_list[0])
        rerank_list=sorted(topk_list[0], key=lambda x: ppl_fun(context=old_context,string=x), reverse=False)
        print('rerank',str(rerank_list))
        return rerank_list
    except Exception as e:
        print('error',e)
        return ['error'*5]

def ablation(reranker,data_info,ppl_fun,index):
    query = data_info['挖空语料-插入点']
    golden_author = data_info['作者']
    golden_quote = data_info['引言']
    print("Query: " + query)
    topk_list = retrieval(vector,query, 5,golden_author)
    print('The retrieval Top K：',str(topk_list))
    if ppl_fun == 'avg':
        ppl_fun = cal_feature_avg
    elif ppl_fun == 'ppl1':
        ppl_fun = cal_feature_ppl1
    elif ppl_fun == 'ppl2':
        ppl_fun = cal_feature_ppl2
    elif ppl_fun == 'vanilla':
        ppl_fun = None
    elif ppl_fun == 'ppl1_novelty':
        ppl_fun = cal_feature_ppl1_novelty
    elif ppl_fun == 'ppl2_novelty':
        ppl_fun = cal_feature_ppl2_novelty
    elif ppl_fun == 'avg_novelty':
        ppl_fun = cal_feature_avg_novelty
    elif ppl_fun == 'chatgpt':
        ppl_fun = gpt_rerank
    elif ppl_fun == 'bge':
        ppl_fun = bge_rerank
    elif ppl_fun == 'upr':
        ppl_fun = upr_rerank
    elif ppl_fun == 'bm25':
        ppl_fun = bm25_rerank
    elif ppl_fun == 'monoT5':
        ppl_fun = monoT5_rerank
    rerank_list = rerank_fn(reranker,query, topk_list, ppl_fun)
    if ppl_fun == None:
        rerank_list = rerank_list[0]
    quote = rerank_list[0]
    mrr=mrr_score(golden_quote,rerank_list)
    hit1=hits_at_k(golden_quote,rerank_list,1)
    hit3=hits_at_k(golden_quote,rerank_list,3)
    ndcg_1=ndcg_at_k(rerank_list,Search_quote_rel,index,k=1) 
    ndcg_3=ndcg_at_k(rerank_list,Search_quote_rel,index,k=3)
    return quote,mrr,hit1,hit3,ndcg_1,ndcg_3,rerank_list

def main(args):
    file_name=args.file_name
    ppl_fun=args.rerank_fun
    if ppl_fun == 'bge':
        reranker=model_bge()
    elif ppl_fun == 'upr':
        reranker=model_upr()
    elif ppl_fun == 'bm25':
        reranker=model_bm25()
    elif ppl_fun == 'monoT5':
        reranker=model_monoT5()
    else: 
        reranker=None
    file_path = f'QUILL/data/dev/{file_name}.xlsx'
    df = pd.read_excel(file_path)
    rerank_quote=[]
    mrr_list=[]
    hit1_list=[]
    hit3_list=[]
    ndcg1_list = []
    ndcg3_list = []
    rerank_all_list =[]
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        quote,mrr,hit1,hit3,ndcg_1,ndcg_3,rerank_list=ablation(reranker,row,ppl_fun,index)
        rerank_quote.append(quote)
        mrr_list.append(mrr)
        hit1_list.append(hit1)
        hit3_list.append(hit3)
        ndcg1_list.append(ndcg_1)
        ndcg3_list.append(ndcg_3)
        rerank_all_list.append(rerank_list)
    df['rerank_all']=rerank_all_list
    df['rerank_quote']=rerank_quote
    df['mrr']=mrr_list
    df['hit1']=hit1_list
    df['hit3']=hit3_list
    df['dcg1']=ndcg1_list
    df['dcg3']=ndcg3_list
    file_path = f'/QUILL/data/eval/ablation/res_{file_name}_{ppl_fun}.xlsx'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_excel(file_path, index=False)
    print("The new Excel file is saved!!!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Pipeline")

    parser.add_argument('--rerank_fun', type=str, required=True, help="ablation index")
    parser.add_argument('--file_name', type=str, required=True, help="dev file name")
    args = parser.parse_args()

    main(args)




