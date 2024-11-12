<p align="left">
    <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish&nbsp
</p>
<br>

# QUILL

## Website&Demo
See our quill website and demo: [QUILL website](https://gracexiaoo.github.io/quill.github.io/).

## Install Dependencies

```
pip install -r requirements.txt
```

## Note

Before proceeding, Run the following scirpt `app.sh` to compute the PPL and extract the quote.  (Confirm that the model path is correctly in the file)

```
cd QUILL/
CUDA_VISIBLE_DEVICES=0 python /code/app/ppl_compute.py
CUDA_VISIBLE_DEVICES=0 python /code/app/quote_extract.py
```

## Evaluation System for QG (Quotation Generation)

You can evaluate the QG task from any desired model via the following scirpt `naive.sh`:

```
cd QUILL/
model='llama2-70b-chat-hf'
num=1
memory=0.8
prompt='0_shot_quote'
CUDA_VISIBLE_DEVICES=0 python /code/naive_rewrite.py --model_name "$model" --file_name 'quote_author'  --tensor_parallel_size "$num" --gpu_memory_utilization "$memory" --prompt "$prompt"
CUDA_VISIBLE_DEVICES=0 python /code/naive_compute.py --model_name "$model" --prompt "$prompt"
```

All the model results are in the folder [data/eval](data/eval).

## Reranking Metrics

The metrics for our designed rerank metrics and Other rerankers can be calculated using the following script  `ablation.sh`:

```
cd QUILL/
#### QUILL's Reranker
CUDA_VISIBLE_DEVICES=0 python /code/ablation.py  --file_name 'quote_author' --rerank_fun 'avg_novelty'

#### Other Rerankers
CUDA_VISIBLE_DEVICES=0 LINKER_TYPE="json" JSON_LINKER_PATH="JSON_LINKER.json"  python /code/ablation.py  --file_name 'quote_author' --rerank_fun 'bm25'
```

All the rerankers model are in the folder [code/reranker](code/reranker).
All the reranking results are in the folder [data/eval/ablation](data/eval/ablation).

## Data

The collected data can be found in the [data/rag](data/rag). All samples have been anonymized.

## Citation
```
@article{xiao24quill
  author    = {JinXiao, BoweiZhang, QianyuHe, JiaqingLiang, FengWei, JingleiChen, ZujieLiang, DeqingYang, YanghuaXiao},
  title     = {QUILL: Quotation Generation Enhancement of Large Language Models},
  year      = {2024},
}
```
