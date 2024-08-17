import os

'''
function to return the entire files in a folder
'''
def readFiles(basePath, fileExtention='csv.*'):
    os.chdir(basePath)
    path =  os.getcwd() 
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            #if fileExtention in file:
            files.append(os.path.join(r, file))
    
    # Get list of all files in a given directory sorted by name
    list_of_files = sorted(files)
    return files


def writeToFile(basePath='/'):
    list_file_paths = readFiles(basePath)
    # for l in list_file_paths:
    #     print(l.split('/')[-1].split('.')[0])
    with open('../../STE_val.txt', 'a') as f:
        for l in list_file_paths:
            f.write(l.split('/')[-1].split('.')[0]+'\n')

if __name__ == '__main__':
    basePath = '/home/nnamdi_20/activeSpeakersContext/dataset/STE_forward/val'
    writeToFile(basePath)
    

