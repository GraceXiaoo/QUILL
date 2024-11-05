import transformers
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4"

model_id = "/data2/hyc/models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/01622a9d125d924bd828ab6c72c995d5eda92b8e"

def get_response(prompt=None):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    res=outputs[0]["generated_text"][-1]['content']
    print(res)
    
get_response()