from random import randrange
from selenium import webdriver
import urllib.request
from selenium.webdriver.common.by import By
import time
import os


def download_image(person_name, src, seq, dir):
    try:
        filename = person_name + str(seq) + '.png' # i.e: "JohnTravolta0.png"
        image_path = os.path.abspath(os.path.join(os.getcwd(), dir, filename)) # /home/user/Desktop/dirname
        urllib.request.urlretrieve(src, image_path) # download image
    
    except Exception:
        pass


def browse_page(person_name, pages, dir):
    seq = 0 #initialize the file number. 
    for i in range(pages): # Loop for the number of pages you want to scrape.
        try:
            driver.get(url+str(i))
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);') # Scroll to the end of page.
            time.sleep(2) # Wait for all the images to load correctly.
            images = driver.find_elements(By.XPATH,"//img[contains(@class, 'MosaicAsset-module__thumb___yvFP5')]") # Find all images. TODO this might need to be changed
        except:
            continue
        for image in images: # For each image in one page:
              try:
                src = image.get_attribute('src') # Get the link
                
                download_image(person_name, src, seq, dir) # And download it to directory
              except:
                pass
              seq += 1
        time.sleep(2)
  
if __name__ == '__main__':
    person_name = ["Al_Pacino","Robert_De"]#input("Please Provide The Person's Name: \n") 
    url = ["https://www.gettyimages.ca/photos/al-pacino?assettype=image&family=editorial&numberofpeople=one&phrase=al%20pacino&sort=mostpopular&page=","https://www.gettyimages.ca/photos/robert-de-niro?assettype=image&family=editorial&numberofpeople=one&phrase=robert%20de%20niro&sort=mostpopular&page="]#input('Please Provide The Page URL: \n')
    dir = ["imgs/Al_Pacino","imgs/Robert_De"]#input('Please Provide The Directory Where The Data Will be Saved: \n')
    pages = [55,100]#int(input('Please Provide How Many Pages You Want To Be Scrapped: \n'))
    # driver = webdriver.Firefox()
    driver = webdriver.Chrome() # IF YOU ARE USING CHROME.	
    # driver.get(url)
    for di in dir:
        if not os.path.isdir(di): # If the folder does not exist in working directory, create a new one.
            os.makedirs(di)
    for i in range(2):
        browse_page(person_name[i], pages[i], dir[i])