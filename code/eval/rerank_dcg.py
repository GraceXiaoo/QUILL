## Only need 1 time to compute the Search_quote_rel, Specifically QUILL/eval/rerank_dcg4rel.py to calculate the dict!!

import numpy as np

def get_relevances(rerank_list,Search_quote_rel,index):
    relevances = []
    for quote in eval(rerank_list):
        rel_here = Search_quote_rel[index]
        if quote in rel_here:
            relevances.append(rel_here[quote])
        else:
            relevances.append(0)
    return relevances

def dcg(relevances):
    return np.sum(relevances / np.log2(np.arange(1, len(relevances) + 1) + 1))

def ndcg_at_k(rerank_list,Search_quote_rel,index,k):
    relevances = get_relevances(rerank_list,Search_quote_rel,index)
    relevances_k = relevances[:k]
    dcg_value = dcg(relevances_k)
    idcg_value = dcg(sorted(relevances, reverse=True)[:k])
    return dcg_value / idcg_value if idcg_value > 0 else 0

