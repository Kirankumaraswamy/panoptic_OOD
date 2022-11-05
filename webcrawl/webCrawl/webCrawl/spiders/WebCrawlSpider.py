import scrapy
from w3lib.html import remove_tags, remove_tags_with_content
import re
import os
from pathlib import Path
import urllib
import pymysql.err
from utils import *

class WebCrawlSpider(scrapy.Spider):
    name = "webCrawl"
    count = 0

    recursive_rule = []

    def __init__(self, *args, **kwargs):
        super(WebCrawlSpider, self).__init__(*args, **kwargs)
        state = kwargs.get('state', '')
        city = kwargs.get('city', '')
        # params always have format state_name/city_name
        # if the value is null/null it means run crawler for all entries
        self.state_tocrawl = urllib.parse.unquote(state)
        self.city_tocrawl = urllib.parse.unquote(city)
        #self.city_tocrawl = "Ansbach"


    def start_requests(self):

        path = Path(os.path.dirname(__file__)).parent.parent.parent
        f = open(path.joinpath("files/recursive_rule.txt"), "r", encoding="utf-8")
        for word in f:
            self.recursive_rule.append(word.strip())

        urls = []

        try:
            connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
            if connection is not None:
                db_Info = connection.get_server_info()
                print("Connected to MySQL Server version ", db_Info)

                cursor = connection.cursor()
                try:
                    sql = "select * from webcrawl.webcrawlUI_weblinks where city_name='Amberg';"
                    if self.city_tocrawl != "null" and self.city_tocrawl != None and self.city_tocrawl != "":
                        sql = "select * from webcrawl.webcrawlUI_weblinks where city_name='%s';" % (self.city_tocrawl)
                    elif self.state_tocrawl != "null" and self.state_tocrawl != None and self.state_tocrawl != "":
                        sql = "select w.id, w.weblink, w.city_id, w.city_name from webcrawl.webcrawlUI_cities as c, webcrawl.webcrawlUI_weblinks as w where c.name=w.city_name and c.state_name='%s';" % (self.state_tocrawl)
                    else:
                        sql = "select * from webcrawl.webcrawlUI_weblinks limit 20;"
                    print(sql)
                    n_weblinks = cursor.execute(sql)
                    weblinks = cursor.fetchall()
                except pymysql.err.DatabaseError as e:
                    print("Error while getting weblinks: ", str(e))

                for index, weblink in enumerate(weblinks):
                    # for now I have restricted crawling to only 20 weblinks otherwise google will block the IP for more crawling.
                    #if index == 20:
                    #    break
                    try:
                        sql = "delete from webcrawl.webcrawlUI_webdata where city_name = '%s';" % (weblink[3])
                        cursor.execute(sql)
                        connection.commit()
                    except pymysql.err.DatabaseError as e:
                        print("Error while deleting webdata: ", str(e))

                    print("Sending request..")
                    print(weblink[1])
                    yield scrapy.Request(url=weblink[2], callback=self.parse,
                                         meta={'weblink_id': weblink[0], 'city_name': weblink[1], 'depth': 0})
                    self.count += 1
                    #time.sleep(0.5)
        except:
            print("Error while connecting to MySQL for cities retrieval")
        finally:
            if (connection is not None):
                cursor.close()
                connection.close()
                print("MySQL connection is closed for weblinks retrieval")



    def parse(self, response):

        weblink_id = response.meta.get('weblink_id')
        city_name = response.meta.get('city_name')
        # we need this to identify the recursive call level.
        depth = response.meta.get('depth')

        print("Recursive level ", depth, " for ", city_name, ".......................")
        hrefs = response.css('a').getall()
        atext = response.css('a::text').getall()

        encoding = response.encoding
        str1 = response.body.decode(encoding)
        # adding additional space at the end of tag. We need these to have space between the texts after calling remove tags method.
        # otherwise the texts will be concatinated without space
        str1 = re.sub(">", "> ", str1)
        content_without_java_script = remove_tags(remove_tags_with_content(str1, ('script','style', 'title')))
        values = content_without_java_script.strip().split("\n")
        content = re.sub('\n', ' ', content_without_java_script)
        page_content = re.sub('\s+', ' ', content)

        try:
            connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
            if connection is not None:
                db_Info = connection.get_server_info()
                print("Connected to MySQL Server version ", db_Info)

                cursor = connection.cursor()

                try:

                    sql = "insert into webcrawl.webcrawlUI_webdata(weblink, text, city_name) values %s;" % (
                            (response.url, content, city_name),)
                    cursor.execute(sql)
                    print("Adding to webdata: ", (city_name, weblink_id))
                except pymysql.err.DatabaseError as e:
                    print("Error while inserting webdata addition: ", str(e))
                except:
                    print("Error while inserting webdata addition: ", city_name, weblink_id)
                connection.commit()

        except:
            print("Error while connecting to MySQL for cities retrieval")
        finally:
            if (connection is not None):
                cursor.close()
                connection.close()
                print("MySQL connection is closed for weblinks addition")

        #print(page_content)

        # This decides how much deep we want to crawl the webpages. Recursively we crawl up to this level
        if depth < 2:
            urls = response.css('a::attr(href)').getall()

            #to avoid duplicates
            url_list = []
            # only recursively call if the the url has following values in it
            for url in urls:
                match_count = 0
                for word in self.recursive_rule:
                    if word in url:
                        match_count = match_count + 1
                #only consider the urls with more than 1 word match. We need some kind of good intelligence to restrict the unnecessary crawling
                if match_count >= 1 and url not in url_list:
                    url_list.append(url)
                    yield response.follow(url, callback=self.parse,
                                      meta={'weblink_id': weblink_id, 'city_name': city_name, 'depth': depth + 1})
                    self.count += 1
"""
from scrapy.crawler import CrawlerProcess

c = CrawlerProcess({
    'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0',
    'FEED_FORMAT': 'csv',
    'FEED_URI': 'output.csv',
})
c.settings["LOG_ENABLED"] = False
c.crawl(WebCrawlSpider)
c.start()"""







