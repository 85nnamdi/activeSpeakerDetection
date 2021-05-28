from downloader import  VideoData
from utility import Utilities
import cv2

# Paths to training and test data
Base_train_url ="https://s3.amazonaws.com/ava-dataset/trainval/"
#Base_test_url ="https://s3.amazonaws.com/ava-dataset/test/"
dataset = "dataset/sampleData.txt"
videoFile = "dataset/2bxKkUgcqpk.mp4"
DataPath = "dataset/"
excelPath = "dataset/2bxKkUgcqpk-activespeaker.csv"



def main():
    
    #Download videos
    # insr = VideoData(dataset, Base_train_url)
    # insr.download()

    # read video frames
    util = Utilities()
    util.readKeypoints(excelPath)
    # frame = 1380.64
    # id = util.readVideoFrames(videoFile, frame)
    # cv2.imshow('Frame number',id)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    # df = util.readExcel(excelPath) 
    # for i in df.index:
    #     print(df[1][i])

    #util.saveFrame(videoFile, excelPath)
    #util.callOpenPose()

if __name__ == '__main__':
    main()