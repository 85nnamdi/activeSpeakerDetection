import requests
import sys


class VideoData():
    def __init__(self,  dataset, url):
        self.url=url
        self.dataset = dataset
        

    def download(self):
        chunk_size =256

        #read the text file to find filename which has to be downloaded
        try:
            fileFromTxt = open(self.dataset, mode='r')
            Lines =fileFromTxt.readlines()
            count = 0
            for line in Lines:
                line_striped=line.strip();
                print(line_striped)
                
                url_download=self.url+line_striped
                print(url_download)
                r = requests.get(url_download, stream=True)

                with open("dataset/"+line_striped, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                    count=count+1
            print("Downloaded video: ", count)
        except OSError as error:
            print("Could not read file: {0}".format(error))
        except IOError as exc:
            print("An error occurred while downloading the file: {0}".format(error))

    
    #Extract pose using openPose
    def extractPose(self, video_file):
        return self