import json
from datetime import datetime
import numpy as np
import cv2
import pandas as pd
import os
import sys
import subprocess
import shlex
'''
Initial pose constants
'''
POSE = [0,1,2,3,4,5,6,7,15,16,17,18]
EYES = [36,37,38,39,40,41,68, 42,43,44,45,46,47,69]
MOUTH = [48,54,60,61,62,63,64,65,66,67]
FACE = [EYES, MOUTH]
FACE = [i for subi in FACE for i in subi]

class Utilities():
    def __init__(self):
        self.path = os.path.dirname(os.path.realpath(__file__))

    '''
    This function is responsible for processing all the keypoints that were saved in a directory
    '''
    
    def readKeypoints(self, path_to_keypoint):
        data = path_to_keypoint
        data = "dataset\\output_May282\\2bxKkUgcqpk910.15__keypoints.json"

        person_count = 0
        poseList=[]
        faceList = []
        with open(data,mode='r') as f:
            data_dict = json.load(f)
            for x in data_dict['people']:
                #how many person in this keypoints
                if x['person_id']:
                    person_count += 1 
                    pose = x['pose_keypoints_2d']
                    for i in POSE:
                        poseList.append(pose[i*3]), poseList.append(pose[(i*3)+1])
                    face=x['face_keypoints_2d']
                    for i in FACE:
                        faceList.append(face[i*3]), faceList.append(face[i*3+1])
                    handL=x['hand_left_keypoints_2d']
                    handR=x['hand_right_keypoints_2d']
                print(poseList)
                del poseList[:]
                    #print(f'Person: {person_count}\n\n {pose} \n\n')
        

    '''
    Given a frame number, this function returns frames from the video
    '''
    
    def readVideoFrames(self, video_name, frame):
        self.video_name = video_name
        #Open the video file
        capture = cv2.VideoCapture(self.video_name)
        total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #Jump to specific frames
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, frame = capture.read()

        return frame

    '''
    This function reads a csv files, extracts frame number, passes the number to the readVideoFrames fucntions
    and saves the returned frame as image default is jpg
    '''
    def saveFrame(self,video_name, csv_file, path_to_savedFrame= "dataset/videoFrames/", extension=".jpg"):
        # read the excel file
        df = pd.read_csv(csv_file, header=None)

        # Sort the second column in ascending order
        df.sort_values(1, ascending=True, inplace=True) 
        counter=0
        path_to_savedFrame = path_to_savedFrame +df[0][0]

        #check if folder exist and create it
        if not os.path.exists(path_to_savedFrame):
            os.mkdir(path_to_savedFrame)
        else: path_to_savedFrame ="dataset/videoFrames/"

        #loop through the column 1 to get the frame, pass it to the readVideoFrames
        for i in df.index:
            frameTime =df[1][i]
            frame = self.readVideoFrames(video_name, frameTime)
            
            #Start saving the image
            filename = str(df[0][i])+str(df[1][i])+str('_')+extension #combine the first and second column to makeup the filename
            full_filename_path = os.path.join(self.path, path_to_savedFrame, filename)
            if not os.path.exists(full_filename_path):
                cv2.imwrite(full_filename_path, frame)
            else:
                # change the file name
                cv2.imwrite(os.path.join(self.path, path_to_savedFrame, str(df[0][i])+str(df[1][i])+'_'+str(i)+extension), frame)
            counter=counter+1
            print("saved frames: ", counter)

    '''
    Function to call open pose.exe from located in ./openpose/bin and passing all the required parameters along
    '''
    def callOpenPose(self, path_to_keypoint="../dataset/videoFrames/"):
        oppath ="openpose/"
        os.chdir(oppath)
        #get the current path
        currentPath = os.getcwd() 
        
        commandLine = shlex.split("./bin/OpenPoseDemo.exe --disable_blending 0 --image_dir "+path_to_keypoint+" --face --hand --write_json ../dataset/output_May282/ --net_resolution 320x176")
        process = subprocess.call(commandLine)

        print(currentPath)

    
    '''
    function to return the entire files in a folder
    '''
    def readDir(self, basePath="dataset/", fileExtention='.json'):
        os.chdir(basePath)
        path =  os.getcwd() 
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if fileExtention in file:
                    files.append(os.path.join(r, file))
        return files
            