import os
from RAGchain.reranker import UPRReranker
from RAGchain.schema import Passage

def model_upr():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ['LINKER_TYPE'] = "json"
    os.environ["JSON_LINKER_PATH"] = "file.json"

    reranker = UPRReranker(use_gpu=False)
    return reranker


def upr_reranker(reranker,query, passages):
    reranked_passages = reranker.rerank(query=query, passages=passages)
    lst = []
    for i in reranked_passages:
        lst.append(i.content)
    return lst

def upr_rerank(reranker,query,list):
    list =list[0]
    passages = [
        Passage(filepath=f"passage_{i+1}", content=content) 
        for i, content in enumerate(list)
    ]
    result = upr_reranker(reranker,query,passages)
    return result


if __name__ == "__main__":
    model_upr()