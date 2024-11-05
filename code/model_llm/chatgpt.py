import requests
from openai import OpenAIError
import backoff
metadata = {
    'model': 'gpt-3.5-turbo',
    'temperature': 0,
    'max_tokens': 4096
}
def get_qaprompt(Q):
    messages = []
    messages.append({'role': 'user', 'content':  Q+"\nOutput:"})
    return messages
@backoff.on_exception(backoff.expo, OpenAIError)
#复杂模型的gpt调用
def get_response(query):
    query=get_qaprompt(query)
    cnt = 0 
    while True:
        cnt += 1
        print(f'Times: {cnt}')
        if cnt > 3:
            print('too much error')
            break

        response = requests.post("http://10.176.64.118:40004/ans",json={
                "name_key": "[Your name key]",
                "messages": query,
                "metadata": metadata
                }).json()
        if response['success'] == True:
            ans = response['ans'][0]
            return ans
        else: print(response)

    return None, None
