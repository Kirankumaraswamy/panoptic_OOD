import scrapy
import os
from pathlib import Path
import pymysql.err
from utils import *


class GenerateCityNamesSpider(scrapy.Spider):
    name = "generateCities"
    hits = 0
    valid_hits = 0

    def start_requests(self):
        urls = []
        state_map = {}
        try:
            path = Path(os.path.dirname(__file__)).parent.parent.parent
            f = open(path.joinpath("files/states.txt"), "r", encoding="utf-8")
            try:
                connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
                if connection is not None:
                    db_Info = connection.get_server_info()
                    print("Connected to MySQL Server version ", db_Info)

                    cursor = connection.cursor()
                    try:
                        sql = "delete from webcrawl.webcrawlUI_states;"
                        cursor.execute(sql)
                    except pymysql.err.DatabaseError as e:
                        print("Error while deleting from states: ", str(e))

                    index = 1
                    for wiki_state in f:
                        data = wiki_state.split("=>")
                        weblink = ""
                        name = data[0].strip()
                        if len(data) > 1:
                            weblink = data[1].strip()
                        state_map[name] = [index, weblink]
                        try:
                            sql = """delete from webcrawl.webcrawlUI_cities"""
                            cursor.execute(sql)
                            connection.commit()
                        except pymysql.err.DatabaseError as e:
                            print("Error while deleting cities: ", str(e))

                        try:
                            sql = "insert into webcrawl.webcrawlUI_states(id, name, weblink) values %s;" % (
                                (index, name, weblink),)
                            cursor.execute(sql)
                            connection.commit()
                        except pymysql.err.DatabaseError as e:
                            print("Error while inserting in to states: ", str(e))

                        index += 1

                    connection.commit()


            except pymysql.OperationalError as e:
                print("Error while connecting to MySQL for states insertion ")
            finally:
                if (connection is not None):
                    cursor.close()
                    connection.close()
                    print("MySQL connection is closed for states insertion")

        except:
            print("error")
        for key, value in state_map.items():
            if value[1] != "":
                yield scrapy.Request(url=value[1], callback=self.parse, meta={'state_id': value[0], 'state_name': key})

    def parse(self, response):

        state_id = response.meta.get('state_id')
        state_name = response.meta.get('state_name')

        # Here we get the list of cities/towns names for each state link provided from wikipedia
        # we have to change this code if wikipedia modifies its structure or if this fails for some other reason
        type1 = response.css('table td li a::attr(title)').getall()

        # for few of the states the wikipedia page structure is different. We have to extract like below
        # if len(cities) == 0:
        type2 = response.css('div.column-multiple ul li a::attr(title)').getall()
        type3 = response.css('tr td:nth-child(1) a::attr(title)').getall()
        cities = type1 + type2 + type3

        if len(cities) > 0:

            try:
                connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
                if connection is not None:
                    db_Info = connection.get_server_info()
                    print("Connected to MySQL Server version ", db_Info)

                    cursor = connection.cursor()
                    sql = "delete from webcrawl.webcrawlUI_cities where state_id = %s;" % (state_id)
                    cursor.execute(sql)
                    print(len(cities))
                    index = 1
                    for city in cities:
                        try:
                            sql = "insert into webcrawl.webcrawlUI_cities(state_id, name, state_name) values %s;" % (
                            (state_id, city.strip(), state_name),)
                            cursor.execute(sql)
                            index += 1
                            print("writing: ", city.strip())
                        except pymysql.err.DatabaseError as e:
                            print("Error while inserting in to cities: ", str(e))
                    connection.commit()

                print("complete")


            except:
                print("Error while connecting to MySQL")
            finally:
                if (connection is not None):
                    cursor.close()
                    connection.close()
                    print("MySQL connection is closed")

"""
from scrapy.crawler import CrawlerProcess

c = CrawlerProcess({
    'USER_AGENT': 'Mozilla/5.0',
    'FEED_FORMAT': 'csv',
    'FEED_URI': 'output.csv',
})
c.settings["LOG_ENABLED"] = True
c.crawl(GenerateCityNamesSpider)
c.start()
"""