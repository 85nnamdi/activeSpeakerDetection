from downloader import  VideoData

# Paths to training and test data
Base_train_url="https://s3.amazonaws.com/ava-dataset/trainval/_a9SWtcaNj8"
Base_test_url ="https://s3.amazonaws.com/ava-dataset/test/"
dataset = "dataset/sampleData.txt"

def main():
    insr = VideoData(dataset, Base_train_url)
    insr.download()



if __name__ == '__main__':
    main()