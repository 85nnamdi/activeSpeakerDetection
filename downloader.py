import requests
import sys

class VideoData():
    def __init__(self, url):
        self.url=url
        

    def download(self):
        chunk_size =256
        r = requests.get(url, stream=True)
        with open("videoName.mp4", "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)

    
    def extractPose(self, video_file):
        return self