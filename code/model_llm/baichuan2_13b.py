import csv
import time
import os
import random 
import openpyxl
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
#transformers 4.33.3
#torch 2.1.0
#triton 2.0
#device = "cuda"
# Note: The default behavior now has injection attack prevention off.
model_path="/mnt/nj-1/dataset/llm_ckpt/Baichuan/baichuan2-13b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
).eval()
model.generation_config = GenerationConfig.from_pretrained(model_path)

def get_response(prompt):
    messages = []
    messages.append({"role": "user", "content": prompt})
    res = model.chat(tokenizer, messages) 
    return res