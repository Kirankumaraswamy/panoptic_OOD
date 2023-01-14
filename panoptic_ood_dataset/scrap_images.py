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
split = "test"



driver = webdriver.Firefox(executable_path=DRIVER_PATH)


# Scroll to the end of the page
def scroll_to_end():
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    time.sleep(5)
    print('Sleeping for {} seconds. scroll done...'.format(5))

if split == "train":
    categories_train = [
         "tiger", "cow", "zebra",  "cat", "goat",  "pig",
         "pigeon",  "eagle", "swan",
          "table", "sofa chair",  "shoe",
          "washing machine",
          "shovel",
    ]
    categories = categories_train
else:
    '''"dog, "horse", "cow", "tiger", "buffallo", "lion", "sheep", "bear", "deer", "ostritch", "hen", "ducks", "elephant",
        "trash bin", "brick", "suitcase", "bird", "ball", "umbrella", "bottle", "helmet", "rickshaw", "wooden box",
        "road blocker", "baby stroller", "tyre", "construction items", "speaker", "barrel",  "fire extinguisher",  "buldozer", "trolley"
        ,  "fire hydrant", "drone", "skating board", "wolf", "cat", "hand bag", "barricade", "shoe" , "rucksack",
        "lawn mover", "robot", "wooden log, "sofa chair,  "boat"""'''

    categories_test = [
       "animal"
    ]
    categories = categories_test

for i, category in enumerate(categories):
    download_path = os.path.join("/home/kumarasw/OOD_dataset/web_ood/web_data" , split)
    random_time = random.randint(30, 60)
    time.sleep(random_time)
    print("sleeping for ", random_time, " seconds ...")
    print("downloading ", category, " images ..................")

    SAVE_FOLDER = os.path.join(download_path, category)
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    GOOGLE_IMAGES = 'https://www.google.com/search?q=white+or+transparent+background+real+images+of+' + category + '&client=ubuntu&hs=WxM&channel=fs&sxsrf=ALiCzsanEyalyWkwPwgXT0bQd3PPftRSKA:1667415347899&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjgq4SclpD7AhWPHOwKHRfWB28Q_AUoAXoECAEQAw&biw=1613&bih=894&dpr=1'
    driver.get(GOOGLE_IMAGES)
    counter = 0
    downloaded_src = []
    if i == 0:
        print("Please click on accept all button to continue....")
        time.sleep(10)


    for i in range(1,10):
        print("scroll ", i)
        scroll_to_end()
    image_elements = driver.find_elements(By.CLASS_NAME, 'rg_i')
    print(len(image_elements))
    for c, image in enumerate(image_elements):
        src = None
        try:
            src= image.get_attribute('src')
        except:
            a =1
        if (src is not None):
            try:
                image.click()
            except:
                print("Src click failed: ", src)
            image_src = None
            highrs_image = driver.find_elements(By.CLASS_NAME, "n3VNCb")
            for img in highrs_image:
                temp_src = highrs_image[1].get_attribute('src')
                if not "data:image" in temp_src:
                    image_src = temp_src
                    break

            '''my_image = image.get_attribute('src').split('data:image/jpeg;base64,')
            filename = os.path.join(SAVE_FOLDER, category+'_' + str(counter) + '.jpeg')
            if (len(my_image) > 1):
                print(src)
                if src not in downloaded_src:
                    with open(filename, 'wb') as f:
                        f.write(base64.b64decode(my_image[1]))
                    counter += 1
                else:
                    print("Duplicate: ", src)
            else:'''

            if image_src is not None and image_src not in downloaded_src:
                print(len(image_elements), "/", c, ": ", image_src)
                downloaded_src.append(image_src)
                extension = image_src.split(".")[-1].split("?")[0]
                filename = os.path.join(SAVE_FOLDER, category + '_' + str(counter) + "."+extension)
                try:
                    urllib.request.urlretrieve(image_src, filename)
                    counter += 1
                except:
                    print("Failed: ", image_src)
            else:
                print("Duplicate: ", image_src)

    print("Total images downloaded: ", counter)
