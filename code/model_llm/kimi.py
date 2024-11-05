from openai import OpenAI
 
client = OpenAI(
    api_key="sk-VxKoFGKL0uWu4qIk73dNiU5xx1aFp8Yt98wP5hys9goEA8eo",
    base_url="https://api.moonshot.cn/v1",
)

def get_response_kimi(input):
    completion = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
        {"role": "user", "content": input}
    ],
    temperature=0,
    )
    ans=completion.choices[0].message.content
    print('kimi',ans)
    return ans
