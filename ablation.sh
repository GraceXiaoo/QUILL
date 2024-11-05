#### QUILL's Reranker
CUDA_VISIBLE_DEVICES=0 python /code/ablation.py  --file_name 'quote_author' --rerank_fun 'avg_novelty'

#### Other Rerankers
CUDA_VISIBLE_DEVICES=0 LINKER_TYPE="json" JSON_LINKER_PATH="JSON_LINKER.json"  python /code/ablation.py  --file_name 'quote_author' --rerank_fun 'bm25'




