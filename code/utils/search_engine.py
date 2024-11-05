# -*- coding:utf-8 -*-
from selenium import webdriver
from selenium.webdriver.chrome.service import Service  # 使用Service类
from selenium.webdriver.chrome.options import Options  # 使用Options类
import json 
import time
import os
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
import re
from rag.rag_module import MyVectorDBConnector
from rag.rag_function import retrieval
import pandas as pd

def is_first_char_english(s):
    if not s:  
        return 0
    first_char = s[0]
    if ('a' <= first_char <= 'z') or ('A' <= first_char <= 'Z'):
        return 1
    else:
        return 0

def search_engine():
    options = Options()
    options.add_argument('--headless') 
    options.add_argument('--disable-gpu')  
    options.add_argument('--no-sandbox')  

    service = Service(executable_path=r"/chromedriver-linux64/chromedriver")
    driver = webdriver.Chrome(service=service, options=options)

    with open(r"lst.json", 'r',encoding='utf-8') as f:
        data = json.load(f)
    output_path = r'search_quote_list.json'

    if os.path.exists(output_path):
        with open(output_path, 'r',encoding='utf-8') as f:
            new_data = json.load(f)
        print(f"Exist {len(new_data)} data, continue from last break")
    else:
        new_data = []
    failure_lst = []
    existing_len = len(new_data)
    
    for i in range(existing_len, len(data)):
        item = data[i]
        search_keyword = item['quote'].replace('"','')
        print(f"Searching NO.{i+1} keyword：{search_keyword}")

        url_cn = f'https://cn.bing.com/search?q={search_keyword}&first=10&mkt=zh-CN&ensearch=0&FORM=BESBTB'
        url_en = f'https://cn.bing.com/search?q={search_keyword}&first=10&mkt=zh-CN&ensearch=1&FORM=BESBTB'

        # 打开搜狗搜索主页
        if is_first_char_english(search_keyword) == 1:
            driver.get(url_en)
            print('英文')
            time.sleep(3)
        else:
            driver.get(url_cn)
            print('中文')
            time.sleep(3)
        try:
            pause_time = random.uniform(2, 5)
            time.sleep(pause_time)

            num_tips_element = driver.find_element(By.CLASS_NAME, 'sb_count')
            num_tips_text = num_tips_element.text
            print(num_tips_text)
            
            match = re.search(r'of ([\d,]+) results', num_tips_text)
            if match:
                total_results = match.group(1)
                result_count = int(total_results.replace(',', ''))
            else:
                match = re.search(r'共 ([\d,]+) 条', num_tips_text)
                total_results = match.group(1)
                result_count = int(total_results.replace(',', ''))

            result = round(result_count,5)
            item['Search'] = result
            new_data.append(item)
            print(result)
            with open(output_path, 'w',encoding='utf-8') as f:
                json.dump(new_data, f, indent=4, ensure_ascii=False)
        except:
            failure_lst.append(search_keyword)
            print('Search Error')
            with open('failure_lst.json', 'w',encoding='utf-8') as f:
                json.dump(failure_lst, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    search_engine()