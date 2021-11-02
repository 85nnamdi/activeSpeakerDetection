import os,sys
from math import*
from time import perf_counter
import pandas as pd
import numpy as np
import csv
from utility import Utilities
import time
import cv2

class distance_measure():
    def __init__(self):
        self.bounding_box_coordinate = [0, 15, 16, 17, 18]
        self.POSECOORDS = [0,1,2,3,4,5,6,7,8,15,16,17,18]
        self.util= Utilities()

    def euclidean_distance(self, x,y):
        val =[]
        try:
            for (j,k) in zip(x,y):
                val.append(sqrt( sum(pow(a-b,2) for a, b in zip(j, k) if a!=0 if b!=0) ))
            return val
            # [[a  for a in (i,j)] for (i,j) in zip(x,y) ] a =if a !=0 if b !=0
        except TypeError:
            try:
                return sqrt( sum(pow(a-b,2) for a, b in zip(x, y) if a!=0 if b!=0) )  
            except TypeError:
                return sqrt( pow(x-y,2) )
    

    '''
    Use this function returns the pose keypoints from JSON Dir
    '''
    def calculate_distance(self, csvDir='dataset/csv/train/', jsonDir='dataset/Json/', displayFrame =False):
        frameDir = 'dataset/frames/'
        rootPath = os.getcwd()
        # For each CSV file in this folder loop
        list_csv  = self.util.readFiles(basePath = csvDir, fileExtention='.csv')
        os.chdir(rootPath)
        root_json_dir = self.util.readDir(jsonDir)
        os.chdir(rootPath)
        counter =0
        counter_none =0
        for each_json_dir in root_json_dir[0]:
            readJsonPath = os.path.join(jsonDir, each_json_dir)
            print(f'Old json: {readJsonPath}')
            
            for each_csv_file in list_csv:
                i = 0
                if each_json_dir in each_csv_file:
                    print(f'JSON: {each_json_dir} CSV: {each_csv_file}')
                    
                    # open csv
                    df = pd.read_csv(each_csv_file, header=None)
                    df.sort_values(1, ascending=True, inplace=True) 
                    width, height = int(df[8][i]), int(df[9][i])
                    x1,x2,y1,y2 =float(df[2][i]),	float(df[3][i]), float(df[4][i]),	float(df[5][i])

                    allJson = self.util.readFiles(basePath=readJsonPath, fileExtention='.json')
                    os.chdir(rootPath)
                    prevKey =[]
                    totalCount=[]
                    dublicate =0
                    for eachJson in allJson:
                        poses, _face = self.util.readKeypoints(eachJson)
                        x1, y1, x2, y2 = float(df[2][i])*width, float(df[3][i])*height, float(df[4][i])*width, float(df[5][i])*height                    
                        # x2 = x1+x2
                        # y2 = y1+y2
                        normalization_value = self.euclidean_distance(poses[2:4], poses[16:18])
                        ckeys,trunc, nose_neck = self.get_pose_keypoint(poses, x1, x2, y1, y2 )
                        
                        #Produce coresponding frames
                        if(displayFrame):
                            framePath = os.path.join(frameDir, each_json_dir)
                            framePath = framePath + '/'+ str(df[0][i])+str(df[1][i]) + '.jpg'
                            i+=1
                            try:
                                dublicate=0
                                frame = cv2.imread(framePath)
                            except:
                                dublicate=dublicate+1
                                # change the file name
                                framePath = framePath + '/'+ str(df[0][i])+str(df[1][i])+str(dublicate) + '.jpg'
                                frame = cv2.imread(framePath)
                        else:
                            i+=1

                        if ckeys==[]:
                            counter_none = counter_none+1
                            if(displayFrame):
                                imgbbox = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0),5)
                               
                            continue    
                        
                        else:
                            counter =counter+1
                            if(displayFrame):
                                imgbbox = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2), int(y2)), (0,255,0),2)
                                
                                cv2.circle(frame, (int(ckeys[0][0]), int(ckeys[0][1])),5, (0, 255, 0), cv2.FILLED)
                                cv2.circle(frame, (int(ckeys[1][0]), int(ckeys[1][1])),5, (0, 255, 0), cv2.FILLED)
                                cv2.circle(frame, (int(ckeys[2][0]), int(ckeys[2][1])),5, (0, 255, 0), cv2.FILLED)
                                cv2.circle(frame, (int(ckeys[3][0]), int(ckeys[3][1])),5, (0, 255, 0), cv2.FILLED)
                                cv2.circle(frame, (int(ckeys[4][0]), int(ckeys[4][1])),5, (0, 255, 0), cv2.FILLED)
                                cv2.circle(frame, (int(ckeys[5][0]), int(ckeys[5][1])),5, (0, 255, 0), cv2.FILLED)
                                cv2.circle(frame, (int(ckeys[6][0]), int(ckeys[6][1])),5, (0, 255, 0), cv2.FILLED)
                                cv2.circle(frame, (int(ckeys[7][0]), int(ckeys[7][1])),5, (0, 255, 0), cv2.FILLED)
                                cv2.circle(frame, (int(ckeys[8][0]), int(ckeys[8][1])),5, (0, 255, 0), cv2.FILLED)
                                cv2.circle(frame, (int(ckeys[9][0]), int(ckeys[9][1])),5, (0, 255, 0), cv2.FILLED)


                                self.POSECOORDS = [0,1,2,3,4,5,6,7,8,15,16,17,18]
                                cv2.line(frame, (int(ckeys[9][0]), int(ckeys[9][1])), (int(ckeys[11][0]), int(ckeys[11][1])),(0, 255, 255), 5)
                                cv2.line(frame, (int(ckeys[10][0]), int(ckeys[10][1])), (int(ckeys[12][0]), int(ckeys[12][1])),(0, 255, 255), 5)
                                cv2.line(frame, (int(ckeys[9][0]), int(ckeys[9][1])), (int(ckeys[0][0]), int(ckeys[0][1])),(0, 255, 255), 5)
                                cv2.line(frame, (int(ckeys[10][0]), int(ckeys[10][1])), (int(ckeys[0][0]), int(ckeys[0][1])),(0, 255, 255), 5)
                                
                                cv2.line(frame, (int(ckeys[0][0]), int(ckeys[0][1])), (int(ckeys[1][0]), int(ckeys[1][1])),(0, 255, 255), 5)
                                # cv2.line(frame, (int(ckeys[2][0]), int(ckeys[2][1])), (int(ckeys[5][0]), int(ckeys[5][1])),(0, 255, 255), 5)
                                # cv2.line(frame, (int(ckeys[2][0]), int(ckeys[2][1])), (int(ckeys[4][0]), int(ckeys[4][1])),(0, 255, 255), 5)
                                # cv2.line(frame, (int(ckeys[5][0]), int(ckeys[2][1])), (int(ckeys[7][0]), int(ckeys[7][1])),(0, 255, 255), 5)
                                # cv2.line(frame, (int(ckeys[1][0]), int(ckeys[1][1])), (int(ckeys[8][0]), int(ckeys[8][1])),(0, 255, 255), 5)
                                
                                cv2.imshow('Active speacker detection using pose estimation', imgbbox)
                                #cv2.waitKey(1)
                                if cv2.waitKey(1) == ord('s'):
                                    
                                    fullpath = os.path.join(frameDir, 'samples', str(df[0][i])+str(df[1][i]) + '.jpg')
                                    cv2.imwrite(fullpath, imgbbox)

                        if len(prevKey):
                            pass
                        else:
                            prevKey = ckeys
                            
                        distanc = self.euclidean_distance( prevKey, ckeys)
                        prevKey = ckeys
                        print(f'{i}: {distanc}')# \n\n Trunk: {trunc} \n\n Nose_Neck: {nose_neck}')

                totalCount.append(f'File: {each_json_dir} Counted: {counter} None: {counter_none}')    
                print(totalCount)

    def get_upperbody(self):
        print(f'upper body:')



    '''
    check if this list contains another list
    '''   
    def check_single_list(self, root_list):
        for element in root_list:
            if isinstance(element, list) == True:
                return True


    '''  
    Each time you pass in the pose read from csv, this function should return the
    x and y coordinates of the keypoints around the face
    '''
    def get_pose_keypoint(self, poses=[], x1=0, x2=0, y1=0, y2=0):
        
        for p in poses:
            newListx = []
            newListy = []
            trun1, trunk2 = p[2:4], p[16:18]
            nose1, nose2 =p[0:2], p[2:4]
            trunck_length = 0
            nose_neck_length = 0
            if trun1!=None and trunk2!=None:
                trunck_length = self.euclidean_distance(trun1, trunk2)
            if nose1!=None and nose2!=None:
                nose_neck_length = self.euclidean_distance(nose1, nose2)
        

            for face in self.bounding_box_coordinate:
                newListx.append(p[self.POSECOORDS.index(face)*2])
                newListy.append(p[self.POSECOORDS.index(face)*2+1])
            for posX in newListx:
                if posX ==0:
                    continue
                if posX < x1 or posX > x2:
                    break
            else:
                for posY in newListy:
                    if posY ==0:
                        continue
                    if posY < y1 or posY > y2:
                        break
                else:
                    faceKeyPoint = [[a for a in qt] for qt in zip(p[: :2], p[1 : :2])] #zip(newListx, newListy)] # get (0 and 1) and (2 and 3) in zip(p[: :2], p[1 : :2])]
                    
                    return  faceKeyPoint, trunck_length, nose_neck_length
        return [], [], []
        
            
if __name__ == '__main__':
    start = perf_counter()

    jsonPath = 'dataset/json/'
    csvDir = 'dataset/csv/train/'
    
    distance_measure().calculate_distance(csvDir, jsonPath, displayFrame=True)
    
    end = perf_counter()
    print ('Total processing time : ',end - start)
# store the len(x) to know the num of cordinate we're working with.