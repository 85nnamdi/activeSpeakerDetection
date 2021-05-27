from downloader import  VideoData
from utility import Utilities
import cv2

# Paths to training and test data
Base_test_url ="https://s3.amazonaws.com/ava-dataset/test/"
dataset = "dataset/sampleData.txt"
videToCapture = "dataset/2PpxiG0WU18.mkv"
DataPath = "dataset/"
excelPath = "dataset/2bxKkUgcqpk-activespeaker.csv"

def main():
    #Download videos
    # insr = VideoData(dataset, Base_train_url)
    # insr.download()

    # read video frames
    util = Utilities()
    print(util.path)
    # frame = 1081.93
    # id = util.readVideoFrames(videToCapture, frame)
    # cv2.imshow('Frame number',id)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    # df = util.readExcel(excelPath) 
    # for i in df.index:
    #     print(df[1][i])

if __name__ == '__main__':
    main()