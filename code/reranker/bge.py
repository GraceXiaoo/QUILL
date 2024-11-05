import os
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

def model_bge():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    reranker = FlagEmbeddingReranker(
        top_n=5, 
        model="BAAI/bge-reranker-large",  
        use_fp16=False
    )
    return reranker

def bge(reranker,documents,query):
    nodes = [NodeWithScore(node=TextNode(text=doc)) for doc in documents]
    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle)
    return [i.node.get_content() for i in ranked_nodes]

def bge_rerank(reranker,data,query):
    data = data[0]
    return bge(reranker,data,query)