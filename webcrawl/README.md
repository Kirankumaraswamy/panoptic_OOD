The following project has two sub projects. One is scrapy project to perform webcrawling and another a DJango project
to display the results from the webcrawling. The data will be saved in mysql database.
# One time initial setup
### 1. Install mysql server
```
sudo apt install mysql-server
sudo apt-get install libmysqlclient-dev
```

### 2. create python env and install the packages using requirements.txt file
```
pip install -r requirements.txt
```

### 3. create database and run migrations
```
create database webcrawl;
```

### 4. run migrations
#### These commands create the tables
```
cd webcrawlUI/webcrawlApp/
python manage.py makemigrations webcrawlUI
python manage.py migrate webcrawlUI
```

### 5. select the database in mysql and generate the entries using the mysql file provided inside the webCrawl directory
#### This step is optional. This loads the data from a pre-existed mysql file. Otherwise you have to run spiders manually to create the data.
```
use webcrawl;
source webcrawl_db.sql
```

### 6. add mysql database name, username and password in the file webcrawlUI/webcrawlApp/settings.py

### 7. update the mysql database name, username and password in webcrawlUI/webCrawl/utils.py

### 8.deploying the scrapy spiders which perform crawling
```
cd webcrawlUI/webCrawl
```
#### type scrapyd in the above directory. This will start the scrapyd server
```
scrapyd
python setup.py bdist_egg
```
#### scrapy-deploy command sometimes is not recognized. Hence give full path of the executable from your python environment.
```
python /home/kiran/kiran/webCrawlProject/WebCrawl/venv/bin/scrapyd-deploy local-target -p webCrawl 
```
#### after executing above step you should see something like below
{"node_name": "kiran-Inspiron-7591", "status": "ok", "project": "webCrawl", "version": "1624835529", "spiders": 3}



# Running the application
### 1. Start the DJango server and view the application in the browser
```
cd webcrawlUI
python manage.py runserver
```

### 2. Run the crawler process by executing following commands
```
cd webcrawlUI/webCrawl
```
#### start the scrapyd server and execute the crawling script. The crawling process runs  immediately on running the script. Later it runs everyday at 1 AM. This script can be modified based on our need.
```
scrapyd
python webcrawl.py
```


# Webcrawling information
#### There are three webcrawlers(spiders) written inside the directory "webcrawlUI/webCrawl/webCrawl/spiders"
#### 1. GenerateCitiyNamesSpider.py
This create the state and city names in the database by reading wikipedia pages. The links to wikipedia is 
provided in the file "webcrawlUI/files/states.txt". The code has to be changes if the parsing fails because of wikipedia
web page change.
#### 2. GoogleSearchSpider.py
This spider sends a google serach for each city geminde in the database and saves the top matching links into the weblinks database table.
This spider is tricky to run as sometimes google blocks the webcrawler and sometimes it fails because of cookies.
Make sure to run this crawler slowly with random delay in the requests.
#### 3. WebCrawlSpider.py
This is the main crawler which visits each of the geminde websites and recursively performs crawling( 3 levels)
and saves the html data from each response body into webdata table. The logic in this code can be modified for better performance in future.

#### NOTE: In order to run the spiders manually the commented code at the end of each file has to be uncommented. 
#### After running manually make sure to uncomment it back otherwise spider deploy will fail.


# Further improvements and bugs
#### 1. The dynamic search field in the UI is broken. It doesn't autocomplete/recommend all the city names in the search field. However if the name is entered correctly and hitting on the search button will display the result.
#### 2. Currently there is no loggers implemented. I am printing the output to standard output.
#### 3. Still there are minor bugs in the code and it has to be improved.

For any issues please write to me: kiran.scorpio27@gmail.com