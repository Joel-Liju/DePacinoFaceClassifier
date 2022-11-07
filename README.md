# DePacinoFaceClassifier
This is a project aiming to differentiate between Al Pacino and Robert De Niro

To run it, you should first make two folders.
1. images
2. finalImages

Put all the required images in the 'images' folder, and run the ML/faceDetection.py after installing the dependecies. 

In order to scrape the data. You need to have another folder called 
imgs

then you need to run the scrapingCode, which should be set up so that, both the people should be scraped and put into imgs, and into specific folders as well.

## training

To train the model, you have to first clean the model using ML/faceDetection.py , then you need to run the training.py file to make a model, then using this model file, you can run the testmodel.py. 


However, ensure that you have the chromedriver.exe installed and accessible for the code to run the selenium.
