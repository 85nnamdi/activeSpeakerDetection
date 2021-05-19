from downloader import  VideoData
from utility import Utilities
import cv2

# Paths to training and test data
Base_train_url="https://s3.amazonaws.com/ava-dataset/trainval/"
Base_test_url ="https://s3.amazonaws.com/ava-dataset/test/"
dataset = "dataset/sampleData.txt"
videToCapture = "dataset/2PpxiG0WU18.mkv"

def main():
    # insr = VideoData(dataset, Base_train_url)
    # insr.download()
    vid = Utilities()
    frame = 1080.03
    id = vid.readVideoFrames(videToCapture, frame)
    cv2.imshow('Frame number',id)
    cv2.waitKey()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()