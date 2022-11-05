import random

import scrapy
import urllib
from scrapy.linkextractors import LinkExtractor
import pymysql.err
import time
from utils import *

class GoogleSearchSpider(scrapy.Spider):
    name = "googleSearch"
    hits = 0
    valid_hits = 0

    def __init__(self, *args, **kwargs):
        super(GoogleSearchSpider, self).__init__(*args, **kwargs)
        state = kwargs.get('state', '')
        city = kwargs.get('city', '')
        #params always have format state_name/city_name
        #if the value is null/null it means run crawler for all entries
        self.state_tocrawl = urllib.parse.unquote(state)
        self.city_tocrawl = urllib.parse.unquote(city)
        #self.city_tocrawl = "Kutzenhausen"



    def start_requests(self):
        urls = []
        try:
            connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
            if connection is not None:
                db_Info = connection.get_server_info()
                print("Connected to MySQL Server version ", db_Info)

                cursor = connection.cursor()
                try:
                    sql = "select * from webcrawl.webcrawlUI_cities where name='Amberg';"
                    if self.city_tocrawl != "null" and self.city_tocrawl != None and self.city_tocrawl != "":
                        sql = "select * from webcrawl.webcrawlUI_cities where name='%s';" % (self.city_tocrawl)
                    elif self.state_tocrawl != "null" and self.state_tocrawl != None and self.state_tocrawl != "":
                        sql = "select * from webcrawl.webcrawlUI_cities where state_name='%s';" % (self.state_tocrawl)
                    else:
                        sql = "select * from webcrawl.webcrawlUI_cities;"
                    n_cities = cursor.execute(sql)
                    cities = cursor.fetchall()
                except pymysql.err.DatabaseError as e:
                    print("Error while getting cities: ", str(e))
        except:
            print("Error while connecting to MySQL for cities retrieval")
        finally:
            if (connection is not None):
                cursor.close()
                connection.close()
                print("MySQL connection is closed for cities retrieval")
            for index, city in enumerate(cities):
                #for now I have restricted crawling to only 5 cities otherwise google will block the IP for more crawling.
                #if index == 5:
                #    break
                city_name = city[1]
                city_id = city[0]
                state_name = city[3]
                query_string = city_name + " " + "geminde baugrundstück"
                encoded_query_string = urllib.parse.quote(query_string, safe="")
                request = "https://www.google.com/search?q=" + encoded_query_string
                #request = "https://www.google.com/search?q=Aach+%28bei+Trier%29+geminde+baugrundst%C3%BCck%26token%3D77c1f767bc31859fee1ffe041343fa48%26allowcookies%3DACCEPTEER%2BALLE%2BCOOKIES&pli=1"
                urls.append((request, city_id, city_name, query_string))
            print(urls)
        for index, url_param in enumerate(urls):
            print("Sending request %s %s ............................................" %(index, url_param[2]))
            print(url_param[0])
            yield scrapy.Request(url=url_param[0], callback=self.parse,
                                 meta={'city_id': url_param[1], 'city_name': url_param[2], 'query_string': url_param[3]})
            # adding random sleep to avoid google identifying the crawler as bot
            if index % 100 == 0:

                sleep_time = random.randint(5, 10)
            else:
                sleep_time = random.randint(1, 5)
            time.sleep(sleep_time)



    def parse(self, response):

        city_id = response.meta.get('city_id')
        city_name = response.meta.get('city_name')
        query_string = response.meta.get('query_string')
        print(city_id, city_name, response.url)

        self.hits += 1

        query_words = query_string.split(" ")

        xlink = LinkExtractor()
        search_list = []
        temp = xlink.extract_links(response)
        for link in xlink.extract_links(response):
            if self.link_has_query_words(link, query_words) :
                self.valid_hits += 1
                search_list.append(link)
                print(link)

        # In some cases the domain doesn't contains city name
        if len(search_list) == 0:
            for link in xlink.extract_links(response):
                # sample valid url from google search
                # discard https://www.google.com/url?q= at the beginning
                url = urllib.parse.unquote(link.url)
                if "?q=" in url:
                    url = url.split("?q=")[1]

                # if there is http in the above url then split with :// so that we will get domain name
                # if there is no http or https in the url it means we can discard
                if "http://" in url or "https://" in url:
                    if url.find("google") == -1:
                        self.valid_hits += 1
                        search_list.append(link)
                        print(link)
                        #considering only first 2 results
                        if len(search_list) >= 2:
                            break

        print("Google search for: ", query_string)
        print("Google returned number of links: ", len(xlink.extract_links(response)))
        print("valid links: ", self.valid_hits)

        try:
            connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
            if connection is not None:
                db_Info = connection.get_server_info()
                print("Connected to MySQL Server version ", db_Info)

                cursor = connection.cursor()
                try:
                    sql = "delete from webcrawl.webcrawlUI_weblinks where city_id = %s;" % (city_id)
                    cursor.execute(sql)
                    connection.commit()
                except pymysql.err.DatabaseError as e:
                    print("Error while deleting weblinks: ", str(e))

                try:
                    for index, value in enumerate(search_list):
                        if (index == 3):
                            break
                        url = search_list[index].url
                        # example of above url:
                        # https://www.google.com/url?q=https://www.alleskralle.com/immobilien/de/grundst%25C3%25BCck%2Bfreiburg%2Bim%2Bbreisgau&sa=U&ved=2ahUKEwjGx9DU3eztAhWHlxQKHedcA1IQFjAAegQIBRAB&usg=AOvVaw0QnXGEUJm7EQ4gIRw9BdFo

                        url = urllib.parse.unquote(url)
                        if len(url.split("?q=")) > 1:
                            url = url.split("?q=")[1]
                        url = url.split("&sa=")[0]


                        # sometimes we get cache details before https:// in the URL and we have to remove it ans some unwanted characters at the end starting with +&
                        url = url[url.index("http"):]
                        url = url.split("+&")[0]

                        sql = "insert into webcrawl.webcrawlUI_weblinks(city_id, city_name, weblink) values %s;" % (
                            (city_id, city_name, url),)
                        cursor.execute(sql)
                        print("Adding to weblinks: ", (city_id, city_name, url))
                except pymysql.err.DatabaseError as e:
                    print("Error while inserting weblinks: ", str(e))
                except:
                    print("Error while inserting weblinks: ", city_name)
                connection.commit()

        except:
            print("Error while connecting to MySQL for weblinks addition")
        finally:
            if (connection is not None):
                cursor.close()
                connection.close()
                print("MySQL connection is closed for weblinks addition")



    def link_has_query_words(self, link, query_words):
        """"
        This method checks if the received URL links while scrapping have searched query words.
        Here we make sure we follow up over only the valid links returned by Google.
        """

        # sample valid url from google search
        # discard https://www.google.com/url?q= at the beginning
        url = urllib.parse.unquote(link.url)
        if "?q=" in url:
            url = url.split("?q=")[1]

        # if there is http in the above url then split with :// so that we will get domain name
        # if there is no http or https in the url it means we can discard
        if "http://" in url or "https://" in url:
            domain_name = url.split("://")[1].split("/")[0]
            for word in query_words:
                word = word.replace("(", "").replace(")", "").replace("ä", "ae").replace("ü", "ue").replace("ö", "oe")
                word = word.strip().lower()
                if word in domain_name:
                    return True
            return False
        else:
            return False



        #yield response.follow(url[i], callback=self.parse)


import time

c = CrawlerProcess({
    'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
    'FEED_FORMAT': 'csv',
    'FEED_URI': 'output.csv',
})
c.settings["LOG_ENABLED"] = False
c.crawl(GoogleSearchSpider)
s_time = time.time()
c.start()
e_time = time.time()
print("Total time: ", str(e_time - s_time))
