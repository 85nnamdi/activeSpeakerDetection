import json
from datetime import datetime
import numpy as np
import cv2
import pandas as pd
import os
import sys
import subprocess
import shlex


class Utilities():
    def __init__(self):
        self.path = os.path.dirname(os.path.realpath(__file__))

    def readKeypoints(self, path_to_keypoint):
        data = path_to_keypoint
        data = "dataset\\output_May282\\2bxKkUgcqpk910.15__keypoints.json"

        number_of_person_count = 0
        with open(data,mode='r') as f:
            data_dict = json.load(f)
            for x in data_dict['people']:
                #how many person in this keypoints
                if x['person_id']:
                    number_of_person_count += 1 
                    person =number_of_person_count
                    face=x['face_keypoints_2d']
                    handL=['hand_left_keypoints_2d']
                    handR=['hand_right_keypoints_2d']
                    print(f'Person: {number_of_person_count}\n\n {face} \n\n')

   


    def readVideoFrames(self, video_name, frame):
        self.video_name = video_name
        #Open the video file
        capture = cv2.VideoCapture(self.video_name)
        total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        #print('Frame count:', total_frame_count)
        
        #Jump to specific frames
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        #print('Position:', float(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        _, frame = capture.read()

        return frame

    
    def saveFrame(self,video_name, csv_file):
        #read the excel file
        df = pd.read_csv(csv_file, header=None)
        counter=0

        #loop through the column 1 to get the frame, pass it to the readVideoFrames
        for i in df.index:
            frameTime =df[1][i]
            frame = self.readVideoFrames(video_name, frameTime)
            
            #Start saving the image
            filename = str(df[0][i])+str(df[1][i])+"_.jpg" #combine the first and second column to makeup the filename
            cv2.imwrite(os.path.join(self.path, "dataset/videoFrames/1",filename), frame)
            counter=counter+1
            print("saved frames: ",counter)


    def callOpenPose(self):
        oppath ="openpose/"
        os.chdir(oppath)
        #get the current path
        currentPath = os.getcwd() #os.path.dirname(os.path.realpath(__file__)
        #commandLine = shlex.split("./bin/OpenPoseDemo.exe --number_people_max 3 --disable_blending 0 --image_dir ../dataset/videoFrames/ --face --hand --write_json ../dataset/output_May281/ --net_resolution 320x176")
        
        commandLine = shlex.split("./bin/OpenPoseDemo.exe --disable_blending 0 --image_dir ../dataset/videoFrames/ --face --hand --write_json ../dataset/output_May282/ --net_resolution 320x176")
        process = subprocess.call(commandLine)

        print(currentPath)
            


