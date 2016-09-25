 # -*- coding: UTF-8 -*-  
import numpy as np
import cv2
import cv
import sys
import os
import shutil

def VideoToImg():
    success,frame=videoCapture.read()
    index=0
    frame_num=0
    while success:
        index+=1
        cv2.imwrite(whole_frame_path+'/'+str(index)+'.jpg',frame)
        shutil.copy(whole_frame_path+'/'+str(index)+'.jpg',dst)
        frame_num+=1
        success,frame=videoCapture.read()
    return frame_num
    
video_name=sys.argv[1]   # sys.argv包含了命令行参数的列表,也即输入的数据 ; 使用时 python ....py "1.avi"
check_name=(video_name.split('/')[-1]).split('.')[0] #get video prefix
whole_frame_path=check_name+'/'+'Imgs'
if not os.path.exists(check_name):
    os.mkdir(check_name)
if not os.path.exists(whole_frame_path):
    os.mkdir(whole_frame_path)
videoCapture = cv2.VideoCapture(video_name)
caffe_root="./"
dst=caffe_root+"examples/images"
if not os.path.exists(dst):
    os.makedirs(dst)
frame_num=VideoToImg()
print frame_num
shutil.rmtree(whole_frame_path)
