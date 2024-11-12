<p align="left">
     中文</a>&nbsp ｜ <a href="README.md">English</a>&nbsp
</p>
<br>

# QUILL

## 网页与DEMO
详细的QUILL网页与DEMO，请看这里： [QUILL website](https://gracexiaoo.github.io/quill.github.io/)

## 安装依赖项

```
pip install -r requirements.txt
```

## 注意

在运行代码前，请先`sh app.sh`来保证计算ppl和提取引言的功能的正常运行 （同时请确认文件中的模型路径正确）
```
cd QUILL/
CUDA_VISIBLE_DEVICES=0 python /code/app/ppl_compute.py
CUDA_VISIBLE_DEVICES=0 python /code/app/quote_extract.py
```

## 评估系统 (QG，Quote generation)

你可以通过运行脚本`naive.sh`，来评估任何一个模型在QG任务上的表现

```
cd QUILL/
model='llama2-70b-chat-hf'
num=1
memory=0.8
prompt='0_shot_quote'
CUDA_VISIBLE_DEVICES=0 python /code/naive_rewrite.py --model_name "$model" --file_name 'quote_author'  --tensor_parallel_size "$num" --gpu_memory_utilization "$memory" --prompt "$prompt"
CUDA_VISIBLE_DEVICES=0 python /code/naive_compute.py --model_name "$model" --prompt "$prompt"
```

本文实验的所有模型结果在以下文件夹中： [data/eval](data/eval)

## 重排指标

你可以通过运行脚本`ablation.sh`，来计算各类重排方法的结果（包括我们所设计的重排方法与其他的主流方法）：

```
cd QUILL/
#### QUILL's Reranker
CUDA_VISIBLE_DEVICES=0 python /code/ablation.py  --file_name 'quote_author' --rerank_fun 'avg_novelty'

#### Other Rerankers
CUDA_VISIBLE_DEVICES=0 LINKER_TYPE="json" JSON_LINKER_PATH="JSON_LINKER.json"  python /code/ablation.py  --file_name 'quote_author' --rerank_fun 'bm25'
```

本文所使用的所有重排器模型在以下文件夹中： [code/reranker](code/reranker)，
对于重排器的所有实验结果在以下文件夹中： [data/eval/ablation](data/eval/ablation)。

## 数据

本文所收集的数据在以下文件夹中 [data/rag](data/rag)。 所有样本已做匿名处理。

## 引用
```
@article{xiao24quill
  author    = {JinXiao, BoweiZhang, QianyuHe, JiaqingLiang, FengWei, JingleiChen, ZujieLiang, DeqingYang, YanghuaXiao},
  title     = {QUILL: Quotation Generation Enhancement of Large Language Models},
  year      = {2024},
}
```
