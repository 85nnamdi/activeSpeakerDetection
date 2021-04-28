from downloader import  VideoData

Base_train_url="https://s3.amazonaws.com/ava-dataset/trainval/"
Base_test_url ="https://s3.amazonaws.com/ava-dataset/test/"
dataset = "dataset/sampleData.txt"

insr = VideoData(dataset, Base_train_url)
insr.download()