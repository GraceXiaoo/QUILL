import os
from RAGchain.schema import Passage
from RAGchain.reranker import MonoT5Reranker

reranker = MonoT5Reranker()


def model_monoT5():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ['LINKER_TYPE'] = "json"
    os.environ["JSON_LINKER_PATH"] = "file.json"
    reranker = MonoT5Reranker()
    return reranker


def monoT5_reranker(reranker,query, passages):
    reranked_passages = reranker.rerank(query=query, passages=passages)
    lst = []
    for i in reranked_passages:
        lst.append(i.content)
    return lst

def monoT5_rerank(reranker,query,list):
    list =list[0]
    passages = [
        Passage(filepath=f"passage_{i+1}", content=content) 
        for i, content in enumerate(list)
    ]
    result = monoT5_reranker(reranker,query,passages)
    return result


if __name__ == "__main__":
    query = "What causes global warming?"
    passages = [
    Passage(filepath="passage_1", content="CO2 is global warming"),
    Passage(filepath="passage_2", content="abcd is just a random string."),
    Passage(filepath="passage_3", content="People are responsible for many activities."),
    Passage(filepath="passage_4", content="This is my question."),
    ]

    reranker = model_monoT5()
    reranked_passages = monoT5_reranker(reranker,query, passages)
    print(reranked_passages)