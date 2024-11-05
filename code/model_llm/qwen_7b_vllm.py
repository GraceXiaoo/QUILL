from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tokenizer = AutoTokenizer.from_pretrained("/data1/dcy/downloads/model/Qwen/Qwen1.5-7B-Chat")
sampling_params = SamplingParams(temperature=0.2, top_p=0.8, repetition_penalty=1.05, max_tokens=4000)


llm = LLM(model="/data1/dcy/downloads/model/Qwen/Qwen1.5-7B-Chat",tensor_parallel_size=1,gpu_memory_utilization=0.5)

def get_response(prompt,tokenizer,sampling_params,llm):
    messages = [
        {"role": "system", "content": "你是一名文学专家，擅长引经据典相关的任务。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # generate outputs
    outputs = llm.generate([text], sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
    print(generated_text)  
    return generated_text
        