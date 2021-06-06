from downloader import  VideoData
from utility import Utilities
import cv2
from time import perf_counter 

# Paths to training and test data
Base_train_url ="https://s3.amazonaws.com/ava-dataset/trainval/"
#Base_test_url ="https://s3.amazonaws.com/ava-dataset/test/"
dataset = "dataset/sampleData.txt"
videoFile = "dataset/4gVsDd8PV9U.mp4"
DataPath = "dataset/"
csvPath = "dataset/4gVsDd8PV9U-activespeaker.csv"

def main():

    # 1 Download videos
    # insr = VideoData(dataset, Base_train_url)
    # insr.download()

    # 2 Save video frames
    util = Utilities()
    #util.saveFrame(videoFile, csvPath)

    # 3 Save keypoint using openpose
    # util.callOpenPose()

    # 4 Read all the saved keypoints
    util.readKeypoints(csvPath)
    
    

if __name__ == '__main__':
    start = perf_counter()
    main()
    end = perf_counter()
    print ('Total processing time : ',end - start)
    