import os, requests, json, time, random
from bs4 import BeautifulSoup
import re
from ..model_llm.chatgpt import get_response as get_response_gpt


headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'}
def eliminate_zhidao(string):
	patterns = ["- 提问时间: [0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日","- 最新回答: [0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日","- 发帖时间: [0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日","[0-9]{1,2}个回答"]
	for pattern in patterns:
		p = re.compile(pattern)
		flag = True
		while flag:
			a = re.search(string=string,pattern=p)
			if a == None:
				break
			string = string[:a.span()[0]]+" "+string[a.span()[1]:]
	return string

import requests
class SearchCorpus:
	driver = None
	def __init__(self):
		random.seed()
		self.session = requests.Session()
		self.session.get("http://m.baidu.com/", timeout=5, headers=headers)
		#print(resp.text)		#self.session.headers 
		self.search_count = 0
		self.expire_count = random.randint(5,100)

	def restart_session(self):
		self.session = requests.Session()
		self.session.get("http://m.baidu.com/", timeout=5, headers=headers)
		self.search_count = 0
		self.expire_count = random.randint(5,100)

	def after_search(self):
		self.search_count += 1
		if self.search_count == self.expire_count:
			print("restarting Session")
			self.restart_session()

	def InitPhantom(self):
		os.system('pkill phantomjs')
		from selenium import webdriver
		webdriver.DesiredCapabilities.PHANTOMJS['phantomjs.page.settings.userAgent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
		self.driver = webdriver.PhantomJS('./phantomjs')
		self.driver.get("http://www.baidu.com/")

	#@cache.cache_redis("ASKBAIDUP|", 1, 2)
	def AskBaiduPhantom(self, q):
		from selenium.webdriver.common.keys import Keys
		if self.driver is None: self.InitPhantom()
		self.driver.find_element_by_id('kw').send_keys(q)
		self.driver.find_element_by_id('kw').send_keys(Keys.RETURN)
		corpus = []
		direct_ans = ''
		for _ in range(3):
			time.sleep(0.5)
			soup = BeautifulSoup(self.driver.page_source, 'html.parser')
			self.soup = soup
			rlst = self.soup.find_all('div', 'result')
			if soup.find('div', 'op_exactqa_s_answer') is not None:
				ret = soup.find('div', 'op_exactqa_s_answer').text.strip()
				print("EXACT FOUND",ret)
				corpus.append(ret)
				direct_ans = ret
			for rr in rlst:
				[x.extract() for x in rr.find_all('span', 'm')]
				zz = rr.text.replace('\xa0', ' ')
				corpus.append(zz)
			if len(corpus) > 0: break
			time.sleep(0.5)
		self.driver.get("http://www.baidu.com/")
		return corpus, direct_ans
			
	def AskBaidu(self, q, page=0):
		url = 'https://www.baidu.com/s?wd=%s' % q
		if page > 0: url += '&pn=%d' % (page*10)
		resp = self.session.get(url, timeout=5,headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'})
		#print(resp.content.decode())
		soup = BeautifulSoup(resp.text, 'html.parser')
		self.soup = soup
		rlst = self.soup.find_all('div', 'result')
		corpus = []
		direct_ans = ''
		if soup.find('div', 'op_exactqa_s_answer') is not None:
			ret = soup.find('div', 'op_exactqa_s_answer').text.strip()
			print("EXACT FOUND",ret)
			corpus.append(ret)
			direct_ans = ret
			#self.after_search()
			#return corpus
		for rr in rlst:
			[x.extract() for x in rr.find_all('span', 'm')]
			zz = rr.text.replace('\xa0', ' ')
			zz = rr.text.replace('\n', '')
			corpus.append(zz)
		self.after_search()
		return corpus
		
	def AskBaike(self,q):
		url = 'https://baike.baidu.com/search/none?word=%s' % q
		resp = requests.get(url, timeout=5, headers=headers)
		resp.encoding = "utf-8"
		soup = BeautifulSoup(resp.text, 'html.parser')
		self.soup = soup
		rlst = self.soup.find_all('p',"result-summary")
		corpus = []
		for rr in rlst:
			[x.extract() for x in rr.find_all('p', 'result-summary')]
			zz = rr.text.replace('\xa0', ' ').replace('<em>',"").replace("</em>","")
			corpus.append(zz)
		return corpus

	def AskSogouWeather(self,q):
		url = 'https://www.sogou.com/web?query=%s' % q
		resp = requests.get(url, timeout=5, headers=headers)
		#print(resp.text)
		soup = BeautifulSoup(resp.text, 'html.parser')
		self.soup = soup
		rlst = self.soup.find_all('div', 'vr-weather161227')
		corpus = []
		for rr in rlst:
			[x.extract() for x in rr.find_all('span', 'm')]
			zz = rr.text.replace('\xa0', ' ').replace("\n"," ")
			corpus.append(zz)
		return corpus

	def AskSogou(self,q):
		url = 'https://www.sogou.com/web?query=%s' % q
		resp = requests.get(url, timeout=5, headers=headers)
		#print(resp.content.decode())
		soup = BeautifulSoup(resp.text, 'html.parser')
		self.soup = soup
		print(soup)
		rlst = self.soup.find_all('div', 'vrwrap')
		corpus = []
		for rr in rlst:
			[x.extract() for x in rr.find_all('span', 'm')]
			zz = rr.text.replace('\xa0', ' ').replace("\n"," ").strip()
			zz = re.sub('[ ]+', ' ', zz)
			corpus.append(zz)
		return corpus

	def AskBing(self,q):
		# not now!
		url = 'https://www.bing.com/search?q=%s' % q
		resp = requests.get(url, timeout=5)
		soup = BeautifulSoup(resp.text)
		ans = ''
		for cls in ['b_xlText b_emphText']:
			node = soup.find('div', class_=cls)
			if node is not None: ans = node.text.strip()	
		return ans

	def MakeJson(self, query, corpus):
		ret = {'answer':'@NULL@', 'query':query, 'query_id':'0'}
		passages = []
		for text in corpus:
			z = {'url':'', 'passage_text':text}
			passages.append(z)
		ret['passages'] = passages
		return ret

	def Search(self, kw, query=None, corpus_list=None):
		if query is None: query = kw
		direct_ans = ''
		if corpus_list is None:
			#zz, direct_ans = self.AskBaiduPhantom(kw)
			zz, direct_ans = [], ''
			#if len(zz) == 0: return None
			# here to add the AskBaike
			##zz.extend(self.AskBaike(kw))
			zz.extend(self.AskBaidu(kw))# deal with weather
			print(zz)
			#if "天气" in kw: zz.extend(self.AskSogouWeather(kw))
			#if len(zz) == 0: zz.extend(self.AskSogou(kw))
			#zz = [eliminate_zhidao(string) for string in zz]
		
		else:
			zz = corpus_list
		return {'quote': kw,'搜索结果': zz}


def Search_fun(quote):
    try:
        sc=SearchCorpus()
        search=sc.Search(quote)
        if search['搜索结果']:
            with open('QUILL/code/prompt/prompt_ch_search.md', 'r') as file:
                prom_search = file.read()
                prompt_search =prom_search.format(search=str(search))
                res = get_response_gpt(prompt_search)
                if 'null' in res:
                    res = res.replace('null', 'None') 
                if 'xxx' in res:
                    res = res.replace('xxx', 'None')
                print('事实正确性',res)
                return res
        else:
            res='无'
            search='无'
    except Exception as e:
        res='无'
        search='无'
        print('搜索出现问题:',str(e))

    return res