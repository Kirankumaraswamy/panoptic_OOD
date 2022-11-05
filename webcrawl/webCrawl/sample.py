import requests

def hello():
    url = 'http://127.0.0.1:6800/schedule.json'
    myobj = {'project': 'webCrawl', 'spider': 'quotes'}
    x = requests.post(url, data=myobj)
    print(x.text)