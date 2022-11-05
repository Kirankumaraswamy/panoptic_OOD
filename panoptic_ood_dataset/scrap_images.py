# https://medium.com/geekculture/scraping-google-image-search-result-dfe01bcbc610
import os
import selenium
from selenium import webdriver
import base64
import time
import urllib.request
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
import random

DRIVER_PATH = '/home/kumarasw/Thesis/panoptic_OOD/webcrawl/geckodriver'
download_path = "/home/kumarasw/OOD_dataset/web_ood/web_data"

category = 'dog'
SAVE_FOLDER = os.path.join(download_path, category)
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

GOOGLE_IMAGES = 'https://www.google.com/search?q=transparent+background+'+category+'+images&client=ubuntu&hs=WxM&channel=fs&sxsrf=ALiCzsanEyalyWkwPwgXT0bQd3PPftRSKA:1667415347899&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjgq4SclpD7AhWPHOwKHRfWB28Q_AUoAXoECAEQAw&biw=1613&bih=894&dpr=1'


driver = webdriver.Firefox(executable_path=DRIVER_PATH)
driver.get(GOOGLE_IMAGES)

# Scroll to the end of the page
def scroll_to_end():
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    rand = random.randint(3, 7)
    time.sleep(rand)
    print('Sleeping for {} seconds. scroll done...'.format(rand))

counter = 0
downloaded_src = []
for i in range(1,3):
    print("Total images downloaded: ", counter)
    scroll_to_end()
    image_elements = driver.find_elements(By.CLASS_NAME, 'rg_i')
    print(len(image_elements))
    for image in image_elements:
        src= image.get_attribute('src')
        if (src is not None):
            my_image = image.get_attribute('src').split('data:image/jpeg;base64,')
            filename = os.path.join(SAVE_FOLDER, category+'_' + str(counter) + '.jpeg')
            if (len(my_image) > 1):
                print(src)
                if src not in downloaded_src:
                    with open(filename, 'wb') as f:
                        f.write(base64.b64decode(my_image[1]))
                    counter += 1
                else:
                    print("Duplicate: ", src)
            else:

                if src not in downloaded_src:
                    print(src)
                    downloaded_src.append(src)
                    urllib.request.urlretrieve(image.get_attribute('src'), filename)
                    counter += 1
                else:
                    print("Duplicate: ", src)

print("Total images downloaded: ", counter)
