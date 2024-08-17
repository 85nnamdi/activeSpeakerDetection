import os
import subprocess

base_url = 'https://s3.amazonaws.com/ava-dataset/'
trainval = 'trainval'
test = 'test'

folds = {'train' : {}, 'val' : {}, 'test' : {}}
folds['train']['url_subfolder'] = 'trainval'
folds['val']['url_subfolder'] = 'trainval'
folds['test']['url_subfolder'] = 'test'
folds['train']['video_folder'] = './dataset/videos/train'
folds['val']['video_folder'] = './dataset/videos/val'
folds['test']['video_folder'] = './dataset/videos/test'
folds['train']['videolist_filename'] = 'trainval-videolist.txt'
folds['val']['videolist_filename'] = 'trainval-videolist.txt'
folds['test']['videolist_filename'] = 'test-videolist.txt'
folds['train']['csv_folder'] = './ava_activespeaker_train_v1.0'
folds['val']['csv_folder'] = './ava_activespeaker_test_v1.0'
folds['test']['csv_folder'] = './ava_activespeaker_test_for_activitynet2019'
print(folds)

for f in folds:
    videolist = []
    with open(folds[f]['videolist_filename'], 'r') as videolist_filelist:
        for row in videolist_filelist:
            stripped_row = row.strip()
            if stripped_row != '':
                videolist.append(stripped_row)
    for csv_filename in os.listdir(folds[f]['csv_folder']):
        selected_video = [x for x in videolist if x[: 11] == csv_filename[: 11]][0]
        commands = ['wget', '-P', folds[f]['video_folder'], os.path.join(base_url, folds[f]['url_subfolder'], selected_video)]
        retcode = subprocess.call(commands, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if retcode == 0:
            print(f'SUCCESS. Fold {f}, video {selected_video}')
