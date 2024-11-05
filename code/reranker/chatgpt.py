import requests
from openai import OpenAIError
import backoff
from rank_gpt import receive_permutation
from rank_gpt import create_permutation_instruction

def messageget(data,query):
    quotes = data[0]
    item = {
        'query': query,
        'hits': [{'content': quote} for quote in quotes]  
    }
    mess = create_permutation_instruction(item, rank_start=0, rank_end=5, model_name='gpt-3.5-turbo')
    return item,mess

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
def get_response(data,query):
    item,query=messageget(data,query)
    
    cnt = 0 
    while True:
        cnt += 1
        print(f'Times: {cnt}')
        if cnt > 3:
            print('too much error')
            break

        response = requests.post("http://10.176.64.118:40004/ans",json={
                "name_key": "[Your name key]",
                "messages": query
                }).json()
        if response['success'] == True:
            ans = response['ans'][0]
            return ans,item
    
    return None, None

def chat_rerank(data,query):
    response,item = get_response(data,query)
    item = receive_permutation(item, response, rank_start=0, rank_end=5)
    contents = [hit['content'] for hit in item['hits']]
    return contents

def gpt_rerank(data,query):
    return chat_rerank(data,query)
