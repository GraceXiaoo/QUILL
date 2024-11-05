### naive.py
## QUILL: model='ours'
## Model: model='[Your model name]'

cd QUILL/
model='llama2-70b-chat-hf'
num=1
memory=0.8
prompt='0_shot_quote'
CUDA_VISIBLE_DEVICES=0 python /code/naive_rewrite.py --model_name "$model" --file_name 'quote_author'  --tensor_parallel_size "$num" --gpu_memory_utilization "$memory" --prompt "$prompt"
CUDA_VISIBLE_DEVICES=0 python /code/naive_compute.py --model_name "$model" --prompt "$prompt"