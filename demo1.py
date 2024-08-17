import cv2
import os
import time
from math import *
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import spatial  # consider remove
from sklearn.metrics.pairwise import cosine_similarity  # currently used
import math
from utility import Utilities


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
    
    def cosine_JointAngles(self, ckeys):
        #Computation of joint angle 
        # 0-1-2 Nose Chest RShoulder
        noseChestX = ckeys[0][0] - ckeys[1][0] 
        noseChestY = ckeys[0][1] - ckeys[1][1]
        chestRShoulderX = ckeys[2][0] - ckeys[1][0]
        chestRShoulderY = ckeys[2][1] - ckeys[1][1]
        # now time compute similarity
        similarityNCRS = cosine_similarity([[noseChestX, noseChestY]],[[chestRShoulderX, chestRShoulderY]])
        
        # 0-1-5 Nose Chest LShoulder
        chestLShoulderX = ckeys[1][0] - ckeys[5][0]
        chestLShoulderY = ckeys[1][1] - ckeys[5][1]
        # now time compute similarity
        similarityNCLS = cosine_similarity([[noseChestX, noseChestY]],[[chestLShoulderX, chestLShoulderY]])

        # 4-3-2 Right arm
        handRElbowX = ckeys[4][0] - ckeys[3][0]
        handRElbowY = ckeys[4][1] - ckeys[3][1]
        elbowRShoulderX =ckeys[3][0] - ckeys[2][0]
        elbowRShoulderY=ckeys[3][1] - ckeys[2][1]
        # now time compute similarity
        similarityRA = cosine_similarity([[handRElbowX, handRElbowY]],[[elbowRShoulderX, elbowRShoulderY]])

        # 1-2-3 Chest - Rshoulder - elbow
        chestRShoulderX =   ckeys[1][0]- ckeys[2][0]
        chestRShoulderY =   ckeys[1][1] - ckeys[2][1]
        rShoulderRElbowX = ckeys[2][0] -ckeys[3][0]
        rShoulderRElbowY = ckeys[3][1] -ckeys[2][1]
        # now time compute similarity
        similarityCRSE = cosine_similarity([[chestRShoulderX, chestRShoulderY]],[[rShoulderRElbowX, rShoulderRElbowY]])

        # 0-1-8 NOSE-CHEST-HIP
        noseChestX =   ckeys[0][0]- ckeys[1][0]
        noseChestY =   ckeys[1][1] - ckeys[1][1]
        chestHipX = ckeys[1][0] -ckeys[8][0]
        chestHipY = ckeys[1][1] -ckeys[8][1]
        # now time compute similarity
        similarityNCH = cosine_similarity([[noseChestX, noseChestY]],[[chestHipX, chestHipY]])

        # 1-5-6 CHEST-lSHOULDER-ELBOW
        chestLShoulderX =   ckeys[1][0]- ckeys[5][0]
        chestLShoulderY =   ckeys[1][1] - ckeys[5][1]
        lShoulderLelbowX = ckeys[5][0] -ckeys[6][0]
        lShoulderLelbowY = ckeys[5][1] -ckeys[6][1]
        # now time compute similarity
        similarityCLSE = cosine_similarity([[chestLShoulderX, chestLShoulderY]],[[lShoulderLelbowX, lShoulderLelbowY]])

        # 5-6-7 Left Arm
        lShoulderLelbowX = ckeys[5][0] -ckeys[6][0]
        lShoulderLelbowY = ckeys[5][1] -ckeys[6][1]
        lElbowLHandX = ckeys[6][0] -ckeys[7][0]
        lElbowLHandY = ckeys[6][1] -ckeys[7][1]
        # now time compute similarity
        similarityLA = cosine_similarity([[lShoulderLelbowX, lShoulderLelbowY]],[[lElbowLHandX, lElbowLHandY]])

        # 0-15-17 Nose R-Eye R-Ear 
        noseREyeX = ckeys[0][0] -ckeys[9][0]
        noseREyeY = ckeys[0][1] -ckeys[9][1]
        rEyeREarX = ckeys[9][0] -ckeys[11][0]
        rEyeREarY = ckeys[9][1] -ckeys[11][1]
        # now time compute similarity
        similarityNRERE = cosine_similarity([[noseREyeX, noseREyeY]],[[rEyeREarX, rEyeREarY]])

        # 18-16-0 Nose L-Eye L-Ear 
        noseLEyeX = ckeys[0][0] -ckeys[10][0]
        noseLEyeY = ckeys[0][1] -ckeys[10][1]
        lEyeLEarX = ckeys[10][0] -ckeys[12][0]
        lEyeLEarY = ckeys[10][1] -ckeys[12][1]
        # now time compute similarity
        similarityNLELE = cosine_similarity([[noseLEyeX, noseLEyeY]],[[lEyeLEarX, lEyeLEarY]])

        # 1-0-15 Chest-Nose-REye
        chestNoseX = ckeys[1][0] -ckeys[0][0]
        chestNoseY = ckeys[1][1] -ckeys[0][1]
        noseREyeX = ckeys[0][0] -ckeys[9][0]
        noseREyeY = ckeys[0][1] -ckeys[9][1]
        # now time compute similarity
        similarityCNRE = cosine_similarity([[chestNoseX, chestNoseY]],[[noseREyeX, noseREyeY]])

        # 1-0- 16 Chest-Nose-LEye
        chestNoseX = ckeys[1][0] -ckeys[0][0]
        chestNoseY = ckeys[1][1] -ckeys[0][1]
        noseLEyeX = ckeys[0][0] -ckeys[10][0]
        noseLEyeY = ckeys[0][1] -ckeys[10][1]
        # now time compute similarity
        similarityCNLE = cosine_similarity([[chestNoseX, chestNoseY]],[[noseLEyeX, noseLEyeY]])
        
        # concatinate all the similarity into a vector
        resultVector = [(similarityNCRS[0][0]), (similarityNCLS[0][0]) , ( similarityRA[0][0]) , (similarityCRSE[0][0]), (similarityNCH[0][0]) , (similarityCLSE[0][0]) , (similarityLA[0][0]), (similarityNRERE[0][0]) , (similarityNLELE[0][0]) , (similarityCNRE[0][0]) , (similarityCNLE[0][0]) ]
        
        return resultVector

    '''
    Use this function returns the pose keypoints from JSON Dir
    '''
    def calculate_distance(self, csvDir='dataset/csv/train/', jsonDir='dataset/Json/', frameDir='', displayFrame =False):
        
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
           # print(f'Old json: {readJsonPath}')
            
            for each_csv_file in list_csv:
                i = 0
                csv_row = 0
                pTime = 0
                if each_json_dir in each_csv_file:
                    # open csv
                    df = pd.read_csv(each_csv_file, header=None)
                    df.sort_values(1, ascending=True, inplace=True) 
                    width, height = int(df[8][i]), int(df[9][i])
                    x1,x2,y1,y2 =float(df[2][i]),	float(df[3][i]), float(df[4][i]),	float(df[5][i])

                    allJson = self.util.readFiles(basePath=readJsonPath, fileExtention='.json')
                    os.chdir(rootPath)
                    prevKey =[]
                    totalCount=[]
                    col_begin = len(df.columns)
                    df[col_begin] = np.nan
                    number_of_mismatchedSimilarity=0

                    for eachJson in allJson:
                        json_filename = os.path.join(readJsonPath, str(df[0][i])+str(df[1][i])+'_keypoints.json') #2PpxiG0WU18900.1_keypoints
                        ############################
                        poses, _face = self.util.readKeypoints(json_filename)
                        x1, y1, x2, y2 = float(df[2][i])*width, float(df[3][i])*height, float(df[4][i])*width, float(df[5][i])*height                    
                        ##########################
                        ########################################################################################
                        # We need to compute the diagonal of the bounding box
                        # This is because we need to replace the nose_neck distance with this diagonal. 
                        # We use this diagonal to account for cases where the nose_neck distance is unavailable.
                        #         Diagonal of a rectangle is defined as diagonal = √(a² + b²)
                        #         where a and b are the sides of the rectangle
                        #       https://www.omnicalculator.com/math/square-diagonal
                        #########################################################################################
                        diagonal = self.compute_diagonal(x1,x2, y1,y2)
                        normalization_value = self.euclidean_distance(poses[2:4], poses[16:18])
                        ckeys,_, _ = self.get_pose_keypoint(poses, x1, x2, y1, y2 )
                        
                        # Set the dy_dx value to null so that we can fill it dynamically later
                        dy_dx = None 
                        
                        # Produce coresponding frames
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
                        
                        # Produce bounding box
                        if ckeys==[]:
                            counter_none = counter_none+1
                            csv_row = csv_row+1
                            if(displayFrame):
                                imgbbox = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,0),5)
                               
                            continue    
                        
                        else:
                            counter =counter+1
                            if(displayFrame):
                                imgbbox = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2), int(y2)), (0,255,0),2)
                                
                                if not (ckeys[0].__contains__(0)):
                                    cv2.circle(frame, (int(ckeys[0][0]), int(ckeys[0][1])),3, (255, 255, 0), cv2.FILLED)
                                if not (ckeys[1].__contains__(0)):
                                    cv2.circle(frame, (int(ckeys[1][0]), int(ckeys[1][1])),3, (255, 255, 0), cv2.FILLED)
                                if not (ckeys[9].__contains__(0)):
                                    cv2.circle(frame, (int(ckeys[9][0]), int(ckeys[9][1])),3, (255, 255, 0), cv2.FILLED)
                                if not (ckeys[10].__contains__(0)):
                                    cv2.circle(frame, (int(ckeys[10][0]), int(ckeys[10][1])),3, (255, 255, 0), cv2.FILLED)
                                if not (ckeys[2].__contains__(0)):
                                    cv2.circle(frame, (int(ckeys[2][0]), int(ckeys[2][1])),3, (255, 255, 0), cv2.FILLED)
                                if not (ckeys[5].__contains__(0)):
                                    cv2.circle(frame, (int(ckeys[5][0]), int(ckeys[5][1])),3, (255, 255, 0), cv2.FILLED)
                                if not (ckeys[4].__contains__(0)):
                                    cv2.circle(frame, (int(ckeys[4][0]), int(ckeys[4][1])),3, (255, 255, 0), cv2.FILLED)
                                if not (ckeys[7].__contains__(0)):
                                    cv2.circle(frame, (int(ckeys[7][0]), int(ckeys[7][1])),3, (255, 255, 0), cv2.FILLED)
                                # if not (ckeys[6].__contains__(0)):
                                #     cv2.circle(frame, (int(ckeys[6][0]), int(ckeys[6][1])),3, (255, 255, 0), cv2.FILLED)
                                # if not (ckeys[3].__contains__(0)):
                                #     cv2.circle(frame, (int(ckeys[3][0]), int(ckeys[3][1])),3, (255, 255, 0), cv2.FILLED)
                                
                                # Connect the dots
                                if not (ckeys[0].__contains__(0) or ckeys[1].__contains__(0)):
                                    cv2.line(frame, (int(ckeys[0][0]), int(ckeys[0][1])), (int(ckeys[1][0]), int(ckeys[1][1])),(255, 100, 50), 2)
                                if not (ckeys[11].__contains__(0) or ckeys[9].__contains__(0)):
                                    cv2.line(frame, (int(ckeys[11][0]), int(ckeys[11][1])), (int(ckeys[9][0]), int(ckeys[9][1])),(255, 100, 50), 2)
                                if not (ckeys[9].__contains__(0) or ckeys[0].__contains__(0)):
                                    cv2.line(frame, (int(ckeys[9][0]), int(ckeys[9][1])), (int(ckeys[0][0]), int(ckeys[0][1])),(255, 100, 50), 2)
                                if not (ckeys[10].__contains__(0) or ckeys[0].__contains__(0)):
                                    cv2.line(frame, (int(ckeys[10][0]), int(ckeys[10][1])), (int(ckeys[0][0]), int(ckeys[0][1])),(255, 100, 50), 2)
                                if not (ckeys[12].__contains__(0) or ckeys[10].__contains__(0)):
                                    cv2.line(frame, (int(ckeys[12][0]), int(ckeys[12][1])), (int(ckeys[10][0]), int(ckeys[10][1])),(255, 100, 50), 2)
                                if not (ckeys[2].__contains__(0) or ckeys[5].__contains__(0)):
                                    cv2.line(frame, (int(ckeys[2][0]), int(ckeys[2][1])), (int(ckeys[5][0]), int(ckeys[5][1])),(255, 100, 50), 2)
                                if not (ckeys[2].__contains__(0) or ckeys[4].__contains__(0)):
                                    cv2.line(frame, (int(ckeys[2][0]), int(ckeys[2][1])), (int(ckeys[4][0]), int(ckeys[4][1])),(255, 100, 50), 2)
                                if not (ckeys[5].__contains__(0) or ckeys[7].__contains__(0)):
                                    cv2.line(frame, (int(ckeys[5][0]), int(ckeys[5][1])), (int(ckeys[7][0]), int(ckeys[7][1])),(255, 100, 50), 2)
                                if not (ckeys[1].__contains__(0) or ckeys[8].__contains__(0)):
                                    cv2.line(frame, (int(ckeys[1][0]), int(ckeys[1][1])), (int(ckeys[8][0]), int(ckeys[8][1])),(255, 100, 50), 2)
                                

                                cTime = time.time()
                                fps = 1 / (cTime - pTime)
                                pTime = cTime
                                cv2.putText(imgbbox, "fps: "+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (155, 200, 150), 2)
                                cv2.imshow('Active speacker detection using pose estimation', imgbbox)
                                
                                if cv2.waitKey(1) == ord('s'):
                                    
                                    fullpath = os.path.join(frameDir, 'samples', str(df[0][i])+str(df[1][i]) + '.jpg')
                                    cv2.imwrite(fullpath, imgbbox)

                        if len(prevKey):
                            pass
                        else:
                            prevKey = ckeys
                        
                        # compute the vertical difference in x and y for each keypoint
                        # if not nose_neck ==[]:
                        # normalized_prevKey = [[item/nose_neck for item in group] for group in prevKey if nose_neck!=0]
                        # normalized_cKey = [[item/nose_neck for item in group] for group in ckeys if nose_neck!=0]
                        normalized_prevKey = [[item/diagonal for item in group] for group in prevKey if diagonal!=0]
                        normalized_cKey = [[item/diagonal for item in group] for group in ckeys if diagonal!=0]
                        print(f'Normalized {normalized_prevKey}')
                        print(f'CKey {ckeys}')
                        
                        # Actually i intend to compute the difference between each valu in y and x in the previous frame and current frame
                        dy_dx = [[i - j for i, j in group] for group in zip(normalized_prevKey, normalized_cKey)]

                        # diff bw hpose of chest and hpose of the nose
                        # also ned the vpose of nose and chest

                        #Computation of joint angle 
                        resultVector = self.cosine_JointAngles(normalized_cKey)

                        #swap the values: Current key becomes previous key in the next iteration
                        prevKey = ckeys

                        #flatten dy_dx
                        dy_dx_flat = [i for subi in dy_dx for i in subi]
                        
                        # concatinate all the similarity into a vector
                        resultVector.extend(dy_dx_flat)
                        if(len(resultVector)<37):
                            resultVector.extend(([0]* (37-len(resultVector)) ))
                            #print(f"Similarity: {len(resultVector)} \n")
                        
                        #Add to the end of the coloumn
                        df.iloc[csv_row, col_begin] = str(resultVector)#pd.DataFrame([resultVector])
                        csv_row = csv_row+1
                    #Save the CSV in the end
                    df.to_csv(each_csv_file, index=False,  header=False)
                totalCount.append(f'File: {each_json_dir} Counted: {counter} None: {counter_none}')   
                print(f'Mismatched similarity: {number_of_mismatchedSimilarity}') 
                print(totalCount)

    '''
    check if this list contains another list
    '''   
    def check_single_list(self, root_list):
        for element in root_list:
            if isinstance(element, list) == True:
                return True

    def compute_diagonal(self, x1, x2, y1, y2):
        ########################################################################################
        # We need to compute the diagonal of the bounding box
        # THis is because we need to replace the nose_neck distance with this diagonal. 
        # We use this diagonal to account for cases where the nose_neck distance is unavailable.
        #         Diagonal of a rectangle is defined as diagonal = √(a² + b²)
        #         where a and b are the sides of the rectangle
        #       https://www.omnicalculator.com/math/square-diagonal
        #########################################################################################
        horizontal_side = x2-x1
        verical_side = y2-y1
        return math.sqrt(math.pow(horizontal_side,2) + math.pow(verical_side,2))

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

    '''
    LIST FLATTENER
    '''
    def flatten(self, list):
        return [i for sub_i in list for i in sub_i]   
            
if __name__ == '__main__':
    start = perf_counter()

    jsonPath = 'dataset/json/val/'
    csvDir ='dataset/STE_Forward/val_pose/'#'dataset/csv/train/'
    frameDir = 'dataset/frames/val/'
    distance_measure().calculate_distance(csvDir, jsonPath, frameDir, displayFrame=True)
    
    end = perf_counter()
    print ('Total processing time : ',end - start)
# store the len(x) to know the num of cordinate we're working with.
