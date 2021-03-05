import os
import sys
import numpy as np
from scipy import spatial as ss

import cv2
from misc.utils import hungarian,read_pred_and_gt,AverageMeter,AverageCategoryMeter

#dataset = 'NWPU'
#dataRoot = '../ProcessedData/' + dataset
#gt_file = dataRoot + '/val_gt_loc.txt'
img_path ='./JHU/images'
gt_file="a.txt"

exp_name = './'
pred_file = './saved_exp_results/JHU_HR_Net_test.txt'

flagError = False
id_std = ['a','b','IMG_89','c']
#id_std[59] = 3098

if not os.path.exists(exp_name):
    os.mkdir(exp_name)

def main():
    
    pred_data, gt_data = read_pred_and_gt(pred_file,gt_file)
    print(pred_data)
    for i_sample in id_std:
        print(i_sample)        
        img = cv2.imread(img_path + '/' + str(i_sample) + '.jpg')#bgr
        point_r_value = 5
        thickness = 3
        if pred_data[i_sample]['num'] !=0:
            pred_p=pred_data[i_sample]['points']
            for i in range(pred_data[i_sample]['num']):
                cv2.circle(img,(pred_p[i][0],pred_p[i][1]),point_r_value,(0,255,0),-1)# tp: green
        cv2.imwrite(exp_name+'/'+str(i_sample)+ '_pre_' + '_rec_' + '.jpg', img)



if __name__ == '__main__':
    main()
