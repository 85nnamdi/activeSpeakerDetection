import json
from datetime import datetime
import numpy as np
import cv2
import pandas as pd
import os
import sys

class Utilities():
    def __init__(self):
        self.path = os.path.dirname(os.path.realpath(__file__))

    def readKeypoints(self, path_to_keypoint):
        data = path_to_keypoint
        data = "dataset\output_json_folder\youtube_000000000000_keypoints.json"

        with open(data,mode='r') as f:
            data_dict = f.read()

        data_json=json.loads(data_dict)

        # for i in data_json['people'][0]['pose_keypoints_2d']:
        #     print (i)

        timestamp = 1061.37
        dt_obj = datetime.fromtimestamp(timestamp)

        print ("Date Object = ", dt_obj)
        print("type(dt_obj) =", type(dt_obj))


    def readVideoFrames(self, video_name, frame):
        self.video_name = video_name
        #Open the video file
        capture = cv2.VideoCapture(self.video_name)
        total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Frame count:', total_frame_count)
        

        #Jump to specific frames
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        print('Position:', float(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        _, frame = capture.read()

        return frame

    def saveFrame(self,video_name, csv_file):
        #read the excel file
        df = pd.read_csv(csv_file, header=None)

        #loop through the column 1 to get the frame, pass it to the readVideoFrames
        for i in df.index:
            frame =df[1][i]
            frame = readVideoFrames(video_name, frame)
            
            #Start saving the image
            filename = df[0][i]+df[1][i]+"_.jpg" #the combine the first and second column to makeup the filename
            cv2.imwrite(filename, frame)
        
            


