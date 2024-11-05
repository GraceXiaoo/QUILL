import time
import requests

def compute_ppl(left,right): 
    data_to_send = {"left": left,"right":right}
    attempt = 0
    max_retries = 10
    backoff_factor = 1
    while attempt < max_retries:
        response = requests.post("http://10.176.40.139:8080/generate", json=data_to_send)
        if response.status_code == 200:
            return response.json()[0]
        attempt += 1
        print(f"Attempt {attempt} failed with status code: {response.status_code}. Retrying...")
        time.sleep(backoff_factor * (2 ** (attempt - 1)))
    raise Exception(f"Request failed after {max_retries} attempts")

def extract_quote(quote): 
    data_to_send = {'quote':quote}
    attempt = 0
    max_retries = 10
    backoff_factor = 1
    while attempt < max_retries:
        response = requests.post("http://10.176.40.139:6060/extract", json=data_to_send)
        if response.status_code == 200:
            return response.json()[0]
        attempt += 1
        print(f"Attempt {attempt} failed with status code: {response.status_code}. Retrying...")
        # Exponential backoff
        time.sleep(backoff_factor * (2 ** (attempt - 1)))
    raise Exception(f"Request failed after {max_retries} attempts")
