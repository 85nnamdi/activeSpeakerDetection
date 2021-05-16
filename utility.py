import json

data = "dataset\output_json_folder\youtube_000000000000_keypoints.json"

with open(data,mode='r') as f:
    data_dict = f.read()

data_json=json.loads(data_dict)

for i in data_json['people'][0]['pose_keypoints_2d']:
    print (i)

