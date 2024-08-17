import os
import sys
import copy
import mmcv
import numpy as np
import pandas as pd
import pickle as pkl
import subprocess
import warnings
import datetime

import torch

from mmpose.apis import process_mmdet_results
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)

warnings.filterwarnings('ignore')

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def get_poses(det_model, pose_model, csv_folder, video_folder, frame_folder, poses_folder, video_name, device):
    csv_header_list = ['Video ID', 'Time Stamp', 'Face X1', 'Face Y1', 'Face X2', 'Face Y2', 'Label', 'Face Track ID']
    csv_ending = '-activespeaker.csv'
    
    pkl_filename = video_name[: video_name.rfind('.')] + '.pkl'
    
    video = mmcv.VideoReader(os.path.join(video_folder, video_name))
    fps = video.fps
    width = video.width
    height = video.height
    
    print(f'\nStarting to work on video {video_name} ({fps} frames per second) at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
    
    video_id = video_name[: video_name.rfind('.')]
    frame_folder = os.path.join(frame_folder, video_id)
    
    df = pd.read_csv(os.path.join(csv_folder, f'{video_id}{csv_ending}'), names = csv_header_list)
    all_ts = sorted(df['Time Stamp'].unique())
    ts_to_frame_idx = {ts : [frm for frm in range(int(np.floor(fps * ts - 0.5)), int(np.ceil(fps * ts + 0.5)) + 1)] for ts in all_ts}
    all_frames = sorted(list(set([frm for ts in ts_to_frame_idx for frm in ts_to_frame_idx[ts]])))
    
    head_bboxes_per_ts = {ts : [[int(np.round(x)) for x in (row[['Face X1', 'Face Y1', 'Face X2', 'Face Y2']].values * [width, height, width, height])] for _, row in df[df['Time Stamp'] == ts].iterrows()] for ts in all_ts}
        
    poses_per_frame = {}
    start_time = datetime.datetime.now()
    for idx_frm, frm in enumerate(all_frames):
        mmdet_results = inference_detector(det_model, video[frm])
        person_results = process_mmdet_results(mmdet_results, 1)
        
        pose_results, returned_outputs = inference_top_down_pose_model(pose_model, video[frm], person_results, bbox_thr = 0.3, format = 'xyxy')
        
        torch.cuda.synchronize(device = device)
        poses_per_frame[frm] = (pose_results, returned_outputs)
        
        if (idx_frm + 1) % 100 == 0:
            current_time = datetime.datetime.now()
            print(f'{idx_frm + 1}/{len(all_frames)} frames processed. Average frame processing time: {int((current_time - start_time).total_seconds() * 1000 / (idx_frm + 1)) / 1000} seconds. Expected to end the processing of all frames of {video_name} by {(start_time + (current_time - start_time) * len(all_frames) / (idx_frm + 1)):%Y-%m-%d %H:%M:%S}')
    
    with open(os.path.join(poses_folder, pkl_filename), 'wb') as poses_per_frame_pklfile:
        pkl.dump((ts_to_frame_idx, head_bboxes_per_ts, poses_per_frame), poses_per_frame_pklfile, protocol = pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('This program needs four arguments: the first being the split from which the videos belong (train, val or test), the second the device where the pose estimation model will run (cuda:X or cpu), and the other two are numbers identifying the start and the end of the iteration through the videos. For the train split, those numbers can be between 0 and 119, for the val split between 0 and 32, and for the test split between 0 and 108. It is expected the first of these numbers to be smaller than or equal to the second.')
    else:
        split = sys.argv[1]
        device = sys.argv[2]
        video_idx_start = int(sys.argv[3])
        video_idx_end = int(sys.argv[4])
        
        det_model = init_detector('./demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py', 
                                'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
                                device = device)
        pose_model = init_pose_model('./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py', 
                                    'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth', 
                                    device = device)
        
        if split == 'train':
            csv_folder = './ava_activespeaker_train_v1.0'
        elif split == 'val':
            csv_folder = './ava_activespeaker_test_v1.0'
        elif split == 'test':
            csv_folder = './ava_activespeaker_test_for_activitynet2019'
        video_folder = os.path.join('./videos', split)
        frame_folder = os.path.join('./AVA', split, 'frames')
        poses_folder = os.path.join('./poses', split)
        
        for video_name in sorted(os.listdir(video_folder))[video_idx_start : video_idx_end + 1]:
            get_poses(det_model, pose_model, csv_folder, video_folder, frame_folder, poses_folder, video_name, device)
