from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
app = Flask(__name__)

model_path="/Qwen/Qwen2-7B-Instruct"
tokenizer1 = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model1 = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

model_path="/meta-llama/Meta-Llama-3-8B"
tokenizer2 = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model2 = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()


def compute_ppl(left_context,right_context, tokenizer, model, device='cuda'):
    context_ids = tokenizer.encode(left_context, return_tensors='pt').to(device)
    input_ids = tokenizer.encode(left_context+right_context, return_tensors='pt').to(device)
    target_ids = input_ids.clone()
    target_ids[:, :context_ids.shape[1]] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
    ppl = torch.exp(neg_log_likelihood)
    return ppl.item()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    if 'left' not in data or 'right' not in data:
        return jsonify({'error': 'Both "left" and "right" keys are required.'}), 400
    left_context = data['left']
    right_context = data['right']
    

    ppl1=compute_ppl(left_context,right_context,tokenizer1,model1)
    ppl2=compute_ppl(left_context,right_context,tokenizer2,model2)
    ppl = (ppl1+ppl2)/2

    return [ppl]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)