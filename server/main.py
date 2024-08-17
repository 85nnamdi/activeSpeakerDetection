import cv2
import os
import pandas as pd
import numpy as np
from time import perf_counter
import concurrent.futures
from data_preparation.downloader import  VideoData
from utility import Utilities

# Paths to training and test data
Base_train_url ='https://s3.amazonaws.com/ava-dataset/trainval/'
#Base_test_url ="https://s3.amazonaws.com/ava-dataset/test/"

txtvideoSet = 'dataset/ava_speech_file_names_v1.txt'
jsonPath = 'dataset/json/'
csvDir = 'dataset/csv/train/'
videos = 'dataset/videos/'

def main():
    # # 1 Download videos
    # videoFile = VideoData(txtvideoSet, Base_train_url)
    # videoFile.download()

    
    util = Utilities()
    rootPath = os.getcwd()
    # 2 Save video frames
    list_videos  =util.readFiles(videos,fileExtention='.*')
    os.chdir(rootPath)
    list_csv = util.readFiles(basePath = csvDir, fileExtention='.csv')
    for each_video_file in list_videos:
        a_vid_file = os.path.basename(each_video_file).split('.',1)[0]
        for each_csv_file in list_csv:
            a_csv_file = os.path.basename(each_csv_file).split('.', 1)[0]
            if a_vid_file in a_csv_file:
                os.chdir(rootPath)
                # util.saveFrame(each_video_file, each_csv_file)
                # worker = Multicore()
                # worker.execute(util.saveFrame, each_video_file, each_csv_file)
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    executor.submit(util.saveFrame, each_video_file, each_csv_file)

    # # 3 Save keypoint using openpose
    # dirpath = 'dataset/frames/'
    # dir = util.readDir(dirpath)
    # for d in dir[0]:
    #     framePath = os.path.join(dirpath,d)
    #     output_path =d
    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         executor.submit(util.callOpenPose, frames_path ='../'+framePath, output_path='../dataset/json/'+output_path)

    # # Save the video dimensions
    # util.save_width_height(csvDir, videos)
    # # 4. keypoints to CSV
    # #util.keyPointToCSV(csvDir=csvDir, jsonDir=jsonPath)


if __name__ == '__main__':
    start = perf_counter()
    main()
    end = perf_counter()
    print ('Total processing time : ',end - start)
