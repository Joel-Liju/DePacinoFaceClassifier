import requests
from bs4 import BeautifulSoup
from sqlalchemy import true
num = 1
#doesn't work
url = "https://www.gettyimages.ca/photos/al-pacino?assettype=image&family=editorial&phrase=al%20pacino&sort=mostpopular&page=1"

page = requests.get(url)

soup = BeautifulSoup(page.content, "html.parser")
print(soup)
temp = soup.find_all("div",class_ = "Gallery-module__container___eT6yU")

print(temp)