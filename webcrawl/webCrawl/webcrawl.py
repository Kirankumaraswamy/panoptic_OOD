import time
from datetime import datetime, timedelta
from threading import Timer
from scrapy.crawler import CrawlerProcess
from webCrawl.spiders.WebCrawlSpider import WebCrawlSpider
import os
import pymysql.err
import psutil


def runWebCrawlSpider():

    command = 'curl http://localhost:6800/schedule.json -d project=webCrawl -d spider=webCrawl'
    os.system(command)



if __name__ == "__main__":

    current_time = datetime.today()
    next_run_time = current_time.replace(day=current_time.day + 1,
                                         hour=1, minute=0, second=0, microsecond=0)
    delta = next_run_time - current_time
    secs = delta.seconds

    secs = 1
    print(current_time)
    print(next_run_time)

    while True:
        scrapy_running = False
        for proc in psutil.process_iter():
            if proc.name().lower().find("scrapy") != -1:
                scrapy_running = True
                print(proc.name().lower())
                break

        if not scrapy_running:
            print("Scrapyd is not started. Please start scrapyd and then run again.")
            break

        print("current time: %s" %(datetime.today()))
        print("Going on sleep for %s seconds. Next run is at %s" % (secs, (datetime.today() + timedelta(0, secs))))
        time.sleep(secs)
        runWebCrawlSpider()
        current_time = datetime.today()
        next_run_time = current_time.replace(day=current_time.day+1, hour=1, minute=0, second=0, microsecond=0)
        delta = next_run_time - current_time
        secs = delta.seconds
        #secs = 100