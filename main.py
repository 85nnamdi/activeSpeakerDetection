from downloader import  VideoData
from utility import Utilities
import cv2
from time import perf_counter 
import concurrent.futures
import asyncio

# Paths to training and test data
Base_train_url ="https://s3.amazonaws.com/ava-dataset/trainval/"
#Base_test_url ="https://s3.amazonaws.com/ava-dataset/test/"
dataset = "dataset/sampleData.txt"
videoFile = "dataset/-5KQ66BBWC4.mkv"
DataPath = "dataset/"
csvPath = "D:\\Users\\Nnamdi\\University of Hamburg\\SoSe2021\\Thesis\\activeSpeakerDetection\\dataset\\4gVsDd8PV9U-activespeakerDemo.csv"
jsonPath = "dataset/"


def main():

    # 1 Download videos
    # insr = VideoData(dataset, Base_train_url)
    # insr.download()

    # 2 Save video frames
    util = Utilities()
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.submit(util.saveFrame,videoFile,csvPath) #imag
        #util.saveFrame(videoFile, csvPath)

    # 3 Save keypoint using openpose
    util.callOpenPose()

    # 4 Read all the saved keypoints
    # for i in util.readDir("dataset/output_June20/"):
    #     util.readKeypoints(csvPath,i)

    #process all the keypoint.json file
    # counter = 0
    # for i in util.readDir():
    #     print("==========Printing===========\n")
    #     print(i)
    #     util.readKeypoints(i)
    #     counter=counter+1
        
    #     print(f"There are {counter} keyframes ")
    

if __name__ == '__main__':
    start = perf_counter()
    main()
    end = perf_counter()
    print ('Total processing time : ',end - start)
    