from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
from flask import Flask, request


with open(f'/QUILL/code/prompt/prompt_ch_extract_quote.md', 'r') as file:
    prompt = file.read()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
app = Flask(__name__)

model_name = "/model/Qwen2.5-32B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def quote_extract(quote):
    prompt_quote = prompt.replace('{quote}',quote)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_quote}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    quote = data['quote']
    response = quote_extract(quote)
    return [response]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6060)