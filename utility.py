import json
from datetime import datetime
import numpy as np
import cv2

class Utilities():
    # def __init__(self):
        

    def readKeypoints(self):
        data = "dataset\output_json_folder\youtube_000000000000_keypoints.json"

        with open(data,mode='r') as f:
            data_dict = f.read()

        data_json=json.loads(data_dict)

        # for i in data_json['people'][0]['pose_keypoints_2d']:
        #     print (i)

        timestamp = 1061.37
        dt_obj = datetime.fromtimestamp(timestamp)

        print ("Date Object = ", dt_obj)
        print("type(dt_obj) =", type(dt_obj))


    def readVideoFrames(self, video_name, frame):
        self.video_name = video_name
        #Open the video file
        capture = cv2.VideoCapture(self.video_name)
        total_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Frame count:', total_frame_count)
        

        #Jump to specific frames
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        print('Position:', float(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        _, frame = capture.read()
        return frame

