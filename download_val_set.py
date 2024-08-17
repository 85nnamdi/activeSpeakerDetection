import requests
from cv2 import split
from utility import Utilities

import os

# Validation folder
csvDir = 'dataset/csv/train - Copy/'
movie_list = 'dataset/ava_speech_file_names_v1.txt'
url ='https://s3.amazonaws.com/ava-dataset/test/'
util = Utilities()

txt_file = open(movie_list, mode='r')
csv_files = util.readFiles(csvDir)
txt_Lines =txt_file.readlines()

for csvf in csv_files:
    for f in txt_Lines:
        if(f.split('.')[0] in csvf):
            print(f)
