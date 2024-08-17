import os
import sys
import cv2
import numpy as np
import pandas as pd
import json
from datetime import datetime
import subprocess
import shlex

'''
Initial pose constants
'''
POSE = [0,1,2,3,4,5,6,7,8,15,16,17,18]
EYES = [36,37,38,39,40,41,68, 42,43,44,45,46,47,69]
MOUTH = [48,54,60,61,62,63,64,65,66,67]
FACE = [EYES, MOUTH]
FACE = [i for subi in FACE for i in subi]

class Utilities():
    def __init__(self):
        self.path = os.path.dirname(os.path.realpath(__file__))

    '''
    This function is responsible for processing all the keypoints that were saved in a directory
    Reads keypoints from each person to a separate list which is combined in the end
    '''
    def readKeypoints(self,pathToJsonFile) -> tuple:
        '''
        the row is the position where value should be writen in csv
        '''
        poseListsList =[]
        faceListsList =[]
        with open(pathToJsonFile,mode='r') as f:
            data_dict = json.load(f)
            for k, v in enumerate(data_dict['people']):
                if v['person_id']:
                    poseList=[]
                    pose = v['pose_keypoints_2d']
                    for i in POSE:
                        poseList.append(pose[i*3]), poseList.append(pose[(i*3)+1])
                    faceList = []
                    face=v['face_keypoints_2d']
                    for i in FACE:
                        faceList.append(face[i*3]), faceList.append(face[i*3+1])
                poseListsList.append(poseList[:])
                faceListsList.append(faceList[:])
                faceListsList
            return poseListsList, faceListsList


    '''
    reads all keypoints into a single list
    '''
    def readKeypoints2(self,pathToJsonFile):
        '''
        the row is the position where value should be writen in csv
        '''
        poseList=[]
        faceList = []
        with open(pathToJsonFile,mode='r') as f:
            data_dict = json.load(f)
            for k, v in enumerate(data_dict['people']):
                pose = v['pose_keypoints_2d']
                for i in POSE:
                    poseList.append(pose[i*3]), poseList.append(pose[(i*3)+1])
                face=v['face_keypoints_2d']
                for i in FACE:
                    faceList.append(face[i*3]), faceList.append(face[i*3+1])
                
        return poseList, faceList

    '''
    Use this function to fill all CSVs with keypoints
    '''
    def keyPointToCSV(self, csvDir='dataset/csv/train/', jsonDir='dataset/Json/'):
        rootPath = os.getcwd()
        # For each CSV file in this folder loop
        list_csv  = self.readFiles(basePath = csvDir, fileExtention='.csv')
        os.chdir(rootPath)
        root_json_dir = self.readDir(jsonDir)
        for each_json_dir in root_json_dir[0]:
            readJsonPath = os.path.join(jsonDir, each_json_dir)
            for each_csv_file in list_csv:
                csv_row = 0
                if each_json_dir in each_csv_file:
                    # open csv
                    print(f'Currently processing {each_csv_file}')
                    df = pd.read_csv(each_csv_file, header=None)
                    df.sort_values(1, ascending=True, inplace=True) 
                    col_begin = len(df.columns)
                    df[col_begin] = np.nan
                    df[col_begin+1] = np.nan
                    allJson = self.readFiles(basePath=readJsonPath, fileExtention='.json')
                    os.chdir(rootPath)
                    for eachJson in allJson:
                        os.chdir(rootPath)
                        pose, face = self.readKeypoints(eachJson)
                        # jump to specific row and col to enter the pose and face
                        df.iloc[csv_row, col_begin] = pd.DataFrame([pose]) #str(pose)
                        df.iloc[csv_row, col_begin+1] = pd.DataFrame([face]) #str(face)
                        
                        # save and repeat
                        csv_row = csv_row+1
                    df.to_csv(each_csv_file, index=False,  header=False)

    '''
    Given a frame number, this function returns frames from the video
    '''
    def readVideoFrames(self, video_name, frameNumber=900):
        #Open the video file
        capture = cv2.VideoCapture(video_name)
        #total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        #Jump to specific frames
        capture.set(cv2.CAP_PROP_POS_MSEC, (frameNumber*1000))
        _, frame = capture.read()
        width =capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        capture.release()
        cv2.destroyAllWindows()
        return  frame, width, height
    
    '''
    Function to populate the CSV with video width and height
    '''
    def save_width_height(self, csv_path, video_path):
        rootPath = os.getcwd()
        # Open video dir
        list_video  = self.readFiles(basePath = video_path, fileExtention='.*')
        os.chdir(rootPath)
        list_csv = self.readFiles(basePath = csv_path, fileExtention='.csv')

        for each_video in list_video:
            csv_row = 0
            for each_csv_file in list_csv:
                if os.path.basename(each_video).split('.')[0] in each_csv_file:
                    print(f'Processing: {each_csv_file}')
                    # open csv
                    df = pd.read_csv(each_csv_file, header=None)
                    df.sort_values(1, ascending=True, inplace=True) 
                    
                    # read get the frame and dimension
                    _, width, height = self.readVideoFrames(each_video)
                    col_begin = len(df.columns)
                    df[col_begin] = np.nan
                    df[col_begin+1] = np.nan
                    for i in df.index:
                        # jump to specific row and col to enter the pose and face
                        df.iloc[i, col_begin] = width
                        df.iloc[i, col_begin+1] = height
                            
                        # # save and repeat
                        # csv_row = csv_row+1
                    df.to_csv(each_csv_file, index=False,  header=False)

    '''
    Saves the cropped image
    '''
    def saveFrame2(self, video_name, csv_file, path_to_savedFrame = 'dataset/frames/', extension='.jpg'):
        # read the excel file
        df = pd.read_csv(csv_file, header=None)

        # Sort the second column in ascending order
        df.sort_values(1, ascending=True, inplace=True) 
        counter=0
        path_to_savedFrame = path_to_savedFrame +df[0][0]

        # check if folder exist and create it
        if not os.path.exists(path_to_savedFrame):
            os.mkdir(path_to_savedFrame)
           
        # var to track duplicate files
        dublicate =0
        # loop through the column 1 to get the frame, pass it to the readVideoFrames
        for i in (df.index):
            frameTime =df[1][i]
            x1 =df[2][i]
            y1 =df[3][i]
            x2 =df[4][i]
            y2 =df[5][i]
            frame,width,height = self.readVideoFrames(video_name, frameTime)
              
            x1 =int(x1*width)
            y1 =int(y1*height)
            x2 =x1+int(x2*width)
            y2 =y1+int(y2*height)

            # Start saving the cropped image
            filename = str(df[0][i])+str(df[1][i])+extension #combine the first and second column to makeup the filename
            full_filename_path = os.path.join(self.path, path_to_savedFrame, filename)
            cropped_frame = frame[y1:y2, x1:x2]
            if not os.path.exists(full_filename_path):
                cv2.imwrite(full_filename_path, cropped_frame)
                dublicate=0
            else:
                dublicate=dublicate+1
                # change the file name
                cv2.imwrite(os.path.join(self.path, path_to_savedFrame, str(df[0][i])+str(df[1][i])+str(dublicate)+extension), cropped_frame)
            counter=counter+1
        print('saved frames: ', counter)

    '''
    This function reads a csv files, extracts frame number, passes the number to the readVideoFrames fucntions
    and saves the returned frame as image.jpg
    '''        
    def saveFrame(self, video_name, csv_file, path_to_savedFrame = 'dataset/frames/demo/', extension='.jpg'):
        # read the excel file
        df = pd.read_csv(csv_file, header=None)

        # Sort the second column in ascending order
        df.sort_values(1, ascending=True, inplace=True) 
        path_to_savedFrame = path_to_savedFrame +df[0][0]

        # check if folder exist and create it
        if not os.path.exists(path_to_savedFrame):
            os.mkdir(path_to_savedFrame)
           
        # var to track duplicate files
        dublicate =0
        # loop through the column 1 to get the frame, pass it to the readVideoFrames
        for i in df.index:
            frameTime =df[1][i]
            frame,_,_ = self.readVideoFrames(video_name, frameTime)
            
            # Start saving the image
            filename = str(df[0][i])+str(df[1][i])+extension #combine the first and second column to makeup the filename
            full_filename_path = os.path.join(self.path, path_to_savedFrame, filename)
            if not os.path.exists(full_filename_path):
                cv2.imwrite(str(full_filename_path), frame)
                dublicate=0
            else:
                dublicate=dublicate+1
                # change the file name
                cv2.imwrite(str(os.path.join(self.path, path_to_savedFrame, str(df[0][i])+str(df[1][i])+str(dublicate)+extension)), frame)
           

    '''
    Function to call open pose.exe from located in ./openpose/bin and passing all the required parameters along
    '''
    def callOpenPose(self, oppath ="openpose/", frames_path="../dataset/frames/new/", output_path="../dataset/Json/demo/"):
        oldpath = os.getcwd()
        os.chdir(oppath)
        #get the current path
        currentPath = os.getcwd()
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        commandLine = shlex.split("./bin/OpenPoseDemo.exe --disable_blending 0 --image_dir "+frames_path+" --face --hand --write_json "+ output_path+" )# --net_resolution 320x176")
        process = subprocess.call(commandLine)

        print(currentPath)
        os.chdir(oldpath)

    
    '''
    function to return the entire files in a folder
    '''
    def readFiles(self, basePath="dataset/output_June20/", fileExtention='.*'):
        
        os.chdir(basePath)
        path =  os.getcwd() 
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                #if fileExtention in file:
                files.append(os.path.join(r, file))
        
        # Get list of all files in a given directory sorted by name
        list_of_files = sorted(files)
        return list_of_files

    def readDir(self, basePath):
        return [x[1] for x in os.walk(basePath)]
