 # -*- coding: UTF-8 -*-  
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import cv
import shutil
import sys
import pylab
import linecache
import scipy.io 
from numpy import *
import math

def compute(HGray,frame):
    d=np.array_split(frame,3,2)  # split an array into 3 sections acorss axis 2(RGB)
    c=(0.299*d[0]+0.587*d[1]+0.114*d[2])
    c=np.array(c)
    a=int32(c)
    HGray.append(a)

def Cal_HGray():
    success,frame=videoCapture.read()
    index=0
    frame_num=0
    HGray=[]
    while success:
        index+=1
        cv2.imwrite(whole_frame_path+'/'+str(index)+'.jpg',frame)
        compute(HGray,frame)
        frame_num+=1
        success,frame=videoCapture.read()
    return HGray

def distance(i,j):
    #print HGray[i].shape()
    sum=0
    for h1 in range(height):
        for w1 in range(width):
            sum+=abs(HGray[i][h1][w1][0]-HGray[j][h1][w1][0])
            # sum+= (abs(HGray[i][h1][w1][0]-HGray[j][h1][w1][0])+abs(HGray[i][h1][w1][1]-HGray[j][h1][w1][1])+abs(HGray[i][h1][w1][2]-HGray[j][h1][w1][2]))/3
    return float(sum)

def Cal_HDGray():
    # one segment consists of 21 frames
    i=0
    HDGray=[]
    while(i+20<=len(HGray)-1):
        d=distance(i,i+20)
        HDGray.append(d)
        i+=20
    return HDGray

def Cal_gmean():
    sum=0.0
    for i in range(len(HDGray)):
        sum+=HDGray[i]
    return sum/len(HDGray)

# def Cal_mean_dev():
    # # compute local mean and std; one "local" means 10 segments
    # i=0
    # lmean=[]
    # ldev=[]
    # while(i<=len(HDGray)-1):
        # new_i=min(i+9,len(HDGray)-1)
        # lmean.append(np.array(HDGray[i:new_i+1]).mean())
        # ldev.append(np.array(HDGray[i:new_i+1]).std())
        # i=new_i+1
    # return lmean,ldev

# def Cal_Tl():
    # # compute local threshold
    # Tl=[]
    # for i in range(len(lmean)):
        # Tl.append(lmean[i]+0.7*ldev[i]*math.log(1+gmean/lmean[i]))
    # return Tl

def SB_segments():
    # get candidate SB segments CSB
    n=len(HDGray)
    CSB=[]
    for i in range(n):  
    #  compute local mean and std; one "local" means 5~6 segments
        left=max(0,i-2)
        right=min(n-1,i+2)
        lmean=np.array(HDGray[left:right+1]).mean()
        lstd=np.array(HDGray[left:right+1]).std()
        Tl=lmean+0.7*lstd*math.log(1+gmean/lmean)
        # HDGray > Tl : candidate segment
        if(HDGray[i]>Tl):
            left=i*20
            right=i*20+20
            adder=[left,right]
            cmp=[HDGray[i]]
            cmp.append(adder)
            CSB.append(cmp)
        # when HDGray is much larger than neighbor segments, it can also be candidate segment
        elif i>0 and i<len(HDGray)-1 and ((HDGray[i]>3*HDGray[i-1])or(HDGray[i]>3*HDGray[i+1])) and HDGray[i]>0.8*gmean:
            left=i*20
            right=i*20+20
            adder=[left,right]
            cmp=[HDGray[i]]
            cmp.append(adder)
            CSB.append(cmp)
    return CSB

def Bisc1():
    i=0
    #print "Bisc1"
    while i <len(CSB):
    #print i
        tmp=CSB.pop(i)
        d=tmp[0]   # d is HDGray[i]
        left=tmp[1][0]    
        right=tmp[1][1]
        df=distance(left,(left+right)/2)
        db=distance((left+right)/2,right)
  
        if df/(db+1.0)>1.5 and df/(d+1.0)>0.7:
            cmp=[0,df]   # flag = 0 : can be CT candidate
            adder=[left,(left+right)/2]
            cmp.append(adder)
            CSB.insert(i,cmp)
            i+=1
        elif db/(df+1.0)>1.5 and db/(d+1.0)>0.7:
            cmp=[0,db]   # flag = 0 : can be CT candidate
            adder=[(left+right)/2,right]
            cmp.append(adder)
            CSB.insert(i,cmp)
            i+=1
        elif db/(d+1.0)<0.3 and df/(d+1.0)<0.3:
            continue
        else:
            cmp=[1,tmp[1]]   # flag = 1 : GT candidate
            CSB.insert(i,cmp)
            i+=1

def Bisc2():
    # Main purpose of Bisc2() is to find CT
    #print  "Bisc2"
    i=0
    while i < len(CSB):
        #print i
        tmp=CSB.pop(i)
        type=tmp[0]
        if(type==1):
            CSB.insert(i,tmp)  # flag = 1: HDGray[i] is GT candidate; preserved
            i+=1
        else:
            d=tmp[1]
            left=tmp[2][0]
            right=tmp[2][1]
            df=distance(left,(left+right)/2)
            db=distance((left+right)/2,right)
            if df/(db+1.0)>1.5 and df/(d+1.0)>0.7:
                cmp=[0,df]   # flag = 0; is CT candidate
                adder=[left,(left+right)/2]
                cmp.append(adder)
                CSB.insert(i,cmp)
                i+=1
            elif db/(df+1.0)>1.5 and db/(d+1.0)>0.7:
                cmp=[0,db]   # flag = 0; is CT candidate
                adder=[(left+right)/2,right]
                cmp.append(adder)
                CSB.insert(i,cmp)   
                i+=1
            elif db/(d+1.0)<0.3 and df/(d+1.0)<0.3:
                continue
            else:
                add=[tmp[2][0],tmp[2][1]]  # To lower the miss rate
                cmp=[1,add]   # flag is set to 1, means HDGray[i] is GT candidate;
                CSB.insert(i,cmp)
                i+=1
        # CT: [0,distance_value,[left,right]]
        # GT: [1,[left,right]]


# Cosine distance to measure the similarity of frame[i] and frame[i+1]
# vec[i]: CNN fc7_features
'''
Candidate segments need to be stored in CT-candidate-folder and GT-candidate-folder respectively
'''

def convert(lst):
    N=len(lst)
    for k in range(N):
        lst[k] = round(lst[k],2)  # limit the accuracy and reduce its computation and storation
    return lst

def min_index(lst):
    index=0
    i=1
    while(i<len(lst)):
        if(lst[i]<lst[index]):
            index=i
            i+=1
        else:
            i+=1
    return index

def dist_cos(i,j):
    k = dot(vec[i],vec[j])/(linalg.norm(vec[i])*linalg.norm(vec[j]))
    return k


def CT_detection(s,e):
    i=s
    G=dist_cos(s,e)
    if(G<0.95):
        thre=0.48+0.52*G
        dis=[]
        while(i<=e):
            d=dist_cos(i,i+1)
            dis.append(d)
            i+=1
        if(min(dis)<thre):
            index=min_index(dis)
            tmax=index+s+1
            return tmax
        else:
            return -1  # means it can be lengthened to GT candidates
    else:
        return 0


def GT_detection(s,e):
    ft=[]
    n=e-s+1
    start=s-1
    end=e+1
    T1=0.25   # T1:max-min>T1; to reduce miss rate, it should be smaller
    T2=0.15   # bias rate
    T3=0.3
    # T1=0.25
    # T2=0.15
    # T3=20
    if(s-n<0 or s+2*n-1>len(HGray)-1 ):
        return -1    # -1 means discard it
    G=dist_cos(s,e)
    if(G<0.9):
        for t in range(n):
            d=abs(dist_cos(start,s+t)-dist_cos(s+t,end))
            ft.append(d) 
        if((max(ft)-min(ft))>T1):
            t_min=min_index(ft)
            if(float(abs((t_min-(n+1)/2)))/n<=T2):
                return 1   # Don't check abnormal points
            else:
                if(n==21):
                    L=t_min-(n+1)/2
                    return (L+100)
                else:
                    return 0
        else:
            return -1
    else:
        return -1

    # for t in range(2*n):  
        # d=distance(s-n+t,s+t)
        # if(t==0):
            # dmax=d
            # dmin=d
            # tmax=t
        # elif(d>dmax):
            # dmax=d
            # tmax=t
        # elif(d<dmin):
            # dmin=d
        # ft.append(d)  
    # # bias is normal; peak-peak-value > threshold
    # if (float(abs(tmax-n))/2/n<T1) and dmax-dmin>T3:
        # up_ab=0
        # dw_ab=0
        # for t in range(0,tmax):
            # if(ft[t]>ft[t+1]):
                # up_ab+=1
        # for t in range(tmax+1,2*n-1):
            # if(ft[t]<ft[t+1]):
                # dw_ab+=1
        # # number of abnormal points is normal
        # if(up_ab+dw_ab<2*n*T2):
            # left1=handle_image(whole_frame_path+'/'+str(s-5)+'.jpg')
            # left2=handle_image(whole_frame_path+'/'+str(s-3)+'.jpg')
            # left3=handle_image(whole_frame_path+'/'+str(s-1)+'.jpg')
            # right1=handle_image(whole_frame_path+'/'+str(e+1)+'.jpg')
            # right2=handle_image(whole_frame_path+'/'+str(e+3)+'.jpg')
            # right3=handle_image(whole_frame_path+'/'+str(e+5)+'.jpg')
            # merge2_1 = [val for val in left1 if val in left2]
            # merge2_2 = [val for val in right1 if val in right2]
            # merge3_1 = [val for val in merge2_1 if val in left3]
            # merge3_2 = [val for val in merge2_2 if val in right3]
            
            # if len(list(set(merge3_1+merge3_2)))>=len(merge3_1)+len(merge3_2)-1:
                # return 1
            # else:
                # return -1
        # else:
            # return -1
    # else:
        # return -1



def handle_image(IMAGE_FILE):
    input_image = caffe.io.load_image(IMAGE_FILE)
    prediction = net.predict([input_image])
    plt.plot(prediction[0]) 
    preDict = {}
    for i in xrange(len(prediction[0])):
        preDict[prediction[0][i]] = i
    five_labels=[]
    for i in xrange(5):
        val = sorted(preDict.keys())[-i -1]
        #tmp=linecache.getline(SYNSET, preDict[val]).split(' ')[0]
        tmp=linecache.getline(SYNSET, preDict[val]).split(' ')[-1].split('\n')[0] #This is just for observe
        five_labels.append(tmp)
    return five_labels



def DLHandle():
    i=0
    #CT detection
    while(i<len(CSB)):
        tmp=CSB.pop(i)
        type=tmp[0]
        if(type==0):
            CT_location=CT_detection(tmp[2][0],tmp[2][1])
            if (CT_location==-1):
                cmp=[tmp[2][0]-5,tmp[2][1]+5]
                new=[1,cmp]
                CSB.insert(i,new)
                i+=1
            elif (CT_location==0):
                continue
            else:
                new=[0,CT_location,CT_location+1]
                # CT_Results.write(str(0)+" "+"["+str(CT_location)+","+str(CT_location+1)+'\n')
                CSB.insert(i,new)
                i+=1
        else:
            CSB.insert(i,tmp)
            i+=1

    #GT detection
    i=0 
    while(i<len(CSB)):      
        tmp=CSB.pop(i)
        type=tmp[0]
        if(type==1):
            flag=GT_detection(tmp[1][0],tmp[1][1])
            if(flag==1):
                add=[tmp[1][0]+1,tmp[1][1]+1]
                CSB.insert(i,[1,add])
                i+=1
            elif(flag==-1):
                continue
            elif(flag==0):
                s=tmp[1][0]-5
                e=tmp[1][1]+5
                flag_2=GT_detection(s,e)  # lengthen the 11-frame segment to 21 and do GT again
                if(flag_2==1):
                    cmp=[s+1,e+1]
                    add=[1,cmp]
                    CSB.insert(i,add)
                    i+=1
                elif((flag_2==-1) or (flag_2==0)):
                    continue
                else:
                    L=flag_2-100
                    flag_2_1=GT_detection(s+L,e+L)
                    if(flag_2_1==1):
                        cmp=[s+L+1,e+L+1]
                        adder=[1,cmp]
                        CSB.insert(i,adder)
                        i+=1
                    else:
                        continue
            else:
                L=flag-100
                flag=GT_detection(tmp[1][0]+L,tmp[1][1]+L)
                if(flag==1):
                    cmp=[tmp[1][0]+L+1,tmp[1][1]+L+1]
                    adder=[1,cmp]
                    CSB.insert(i,adder)
                    i+=1 
                else:
                    continue
        else:
            CSB.insert(i,tmp)
            i+=1
    Last_s=len(HDGray)*20
    Last_e=len(HGray)-4-2  # need to modify
    Lst_seg=[Last_s,Last_e]
    CT_location=CT_detection(Last_s,Last_e)
    if(CT_location==-1 or CT_location==0):
        flag_GT=GT_detection(Last_s,Last_e)
        if(flag==1):
            cmp=[Last_s+1,Last_e+1]
            add=[1,cmp]
            CSB.append(add)
        else:
            print("The last segment is not Transition")
    else:
        cmp=[0,[CT_location+1,CT_location+2]]
        CSB.append(cmp)

#mkdirs
video_name=sys.argv[1]   # sys.argv包含了命令行参数的列表,也即输入的数据 ; 使用时 python ....py "1.avi"
check_name=(video_name.split('/')[-1]).split('.')[0] #get video prefix
whole_frame_path=check_name+'/'+'whole'+check_name
if not os.path.exists(check_name):
    os.mkdir(check_name)
if not os.path.exists(whole_frame_path):
    os.mkdir(whole_frame_path)
if not os.path.exists(check_name+'/'+check_name):
    os.mkdir(check_name+'/'+check_name)
if not os.path.exists(check_name+'/final_'+check_name):
    os.mkdir(check_name+'/final_'+check_name)
if not os.path.exists(check_name+'/'+'shot_bound_frames'):
    os.mkdir(check_name+'/'+'shot_bound_frames')

#read video
videoCapture = cv2.VideoCapture(video_name)
height = int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
width = int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
HGray=Cal_HGray()
HDGray=Cal_HDGray()
gmean=Cal_gmean()
CSB=SB_segments()
Bisc1()
Bisc2()
print(CSB)

caffe_root = './'
matfile=caffe_root+'examples/_temp/features_fc7_anni001.mat'
data=scipy.io.loadmat(matfile)
Feats=data['feats']
(m,n)=shape(Feats)
vec=[]
for i in range(m):
    temp=Feats[i,:]
    convert(temp)
    vec.append(temp)

# CT_Results = open(check_name+'/'+"CT_Results.txt",'w')
# GT_Results = open(check_name+'/'+"GT_Results.txt",'w')
DLHandle()
# CT_Results.close()
# GT_Results.close()'''
print CSB
