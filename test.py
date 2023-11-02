import requests
import json

article = "India vice-captain Rohit Sharma will be opening for the first time overseas in Test cricket during the third Test against Australia, which starts in Sydney tomorrow. Rohit has opened in six Test innings, which all came in 2019. He has scored 556 runs at an average of 92.66 as an opener in Test cricket. His highest score is 212."
data = {'news_article':article}


url = 'http://localhost:8085/infer'
headers = {"Content-Type":"application/json"}

respone = requests.post(url, json=data, headers=headers)

print(respone.json())