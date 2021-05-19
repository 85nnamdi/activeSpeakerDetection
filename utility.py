import json


def readKeypoints():
    data = "dataset\output_json_folder\youtube_000000000000_keypoints.json"

    with open(data,mode='r') as f:
        data_dict = f.read()

    data_json=json.loads(data_dict)

    # for i in data_json['people'][0]['pose_keypoints_2d']:
    #     print (i)

    from datetime import datetime

    timestamp = 1061.37
    dt_obj = datetime.fromtimestamp(timestamp)

    print ("Date Object = ", dt_obj)
    print("type(dt_obj) =", type(dt_obj))


