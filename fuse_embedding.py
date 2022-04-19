from asyncio.windows_events import NULL
import os, sys
import numpy as np
from utility import Utilities
import ast
import csv,operator
import itertools

class DataProcessing():
    def __init__(self):
        self.currentdir = os.path.dirname(os.path.realpath(__file__))
        self.parentdir = os.path.dirname(self.currentdir)

    def generatecsv(self, pose_csv,  ste_foward, targetDir):
        print("__Started generating the dataset__")
        # Paths
        rootPath = os.getcwd()

        util = Utilities()
        ste_foward_data  =util.readFiles(ste_foward,fileExtention='.*')
        os.chdir(rootPath)
        pose_csv_data  = util.readFiles(pose_csv,fileExtention='.*')

        for ste in ste_foward_data:
            ste_csv_name = os.path.basename(ste).split('.',1)[0]
            for pose in pose_csv_data:
                if ste_csv_name in pose:
                    print(f'Generating: {ste_csv_name}augumented.csv')
                    # # Open the pose CSV
                    zip_longest = itertools.zip_longest
                    f_pose = open(pose)
                    f_ste = open(ste)

                    csv_f1 = csv.reader(f_pose)
                    csv_f2 = csv.reader(f_ste)
                    data_pose = sorted(csv_f1, key=operator.itemgetter(1)) 
                    data_ste = sorted(csv_f2, key=operator.itemgetter(1)) 
                    targetDir = os.path.join(self.currentdir, targetDir)
                    
                    with open(os.path.join(targetDir, ste_csv_name+'-augumented3.csv'), mode='w', newline='', encoding='utf-8') as f:
                        f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        for row_pose, row_ste in zip_longest(data_pose, data_ste):
                            # merge_list =[]
                            merge_list =[]
                            merged_for_writing = np.array([])
                            try:
                                merge_list.extend(ast.literal_eval(row_ste[6])) 
                                merge_list.extend(ast.literal_eval(row_pose[10]))
                                # ste_embeding =ast.literal_eval(row_ste[6])
                                # pose_embeding =np.array(ast.literal_eval(row_pose[10])).resize((512,))
                                
                                # merged_ = np.append(merge_list, ste_embeding ,0)
                                # merged_for_writing = np.append(merged_, pose_embeding ,0)
                                
                                # print(len(merged_for_writing))
                            except:
                                merge_list.extend(([0]*37))
                                # merged_for_writing = np.append(merge_list, [],0)
                                # merged_for_writing.resize((512,))
                                # print(len(merged_for_writing))
                         
                            f_writer.writerow( [row_ste[0], row_ste[1], row_ste[2], row_ste[3], row_ste[4], row_ste[5], merge_list] )
                    
                    f_ste.close()
                            
                        
if __name__ == '__main__':
    
    ste_foward = 'dataset/STE_Forward/val/'
    pose_csv = 'dataset/STE_Forward/val_Pose/'
    asc_input = 'dataset/ASC_Input/val_fused/'
    startProcessing = DataProcessing()
    startProcessing.generatecsv(pose_csv, ste_foward, asc_input)
    
#https://stackoverflow.com/questions/58568595/wrote-a-pandas-df-with-a-column-with-lists-to-a-file-how-to-read-it-back