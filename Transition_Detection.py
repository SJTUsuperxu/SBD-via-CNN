 # -*- coding: UTF-8 -*-  
import numpy as np
from numpy import *

# Cosine distance to measure the similarity of frame[i] and frame[i+1]
# vec[i]: CNN fc7_features
'''
Candidate segments need to be stored in CT-candidate-folder and GT-candidate-folder respectively
'''
def min_index(lst):
    index=0
    i=1
    while(i<len(lst)):
        if(list[i]<list[index]):
            index=i
        else:
            continue
        i+=1
    return index

def dist_cos(i,j):
    dot_product = 0
    norm_i = 0
    norm_j = 0
    for a,b in zip(vec[i],vec[j]):
        dot_product+=a*b
        norm_i+=a**2
        norm_j+=b**2    
    if norm_i==0 or norm_j==0:
        return None
    else:
        return dot_product/((norm_i*norm_j)**0.5) 


#CT Detection: 
#(1)G<0.95 ; (2)dist_cos(t)<adaptive_threshold

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
            tmax=index+s
            return tmax
        else:
            return -1  # means it can be lengthened to GT candidates
    else:
        return 0

# GT Detection
# (1)G<0.9 ; (2) abs(max(dt)-min(dt))>T1; (3) abs((t_min-(n+1)/2)/n)<=T2 < bias:L >

def GT_detection(s,e):
    ft=[]
    n=e-s+1
    start=s-1
    end=e+1
    T1=0.33
    T2=0.25
    T3=0.3
    # if(s-n<0 or s+2*n-1>len(HGray)-1 ):
        # return -1    # -1 means discard it
    # # if(s-5<0 or e+5>len(HGray)):
        # # return -1
    G=dist_cos(s,e)
    if(G<0.9):
        for t in range(n):
            d=abs(dist_cos(start,s+t)-dist_cos(s+t,end))
            ft.append(d)
        
        if((max(ft)-min(ft))>T1):
            t_min=s+min_index(ft)
            if(abs((t_min-(n+1)/2)/n)<=T2):
                return 1   # Don't check abnormal points
            else:
                L=t_min-(n+1)/2
                return (L+100)
        else:
            return -1
    else:
        return -1
        
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
            # return 1
            # else:
                # return -1
        # else:
            # return -1
    # else:
        # return -1

def Detection():

    i=0
    #CT detection
    while(i<len(CSB)):
        tmp=CSB.pop(i)
        type=tmp[0]
        if type==0:
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
    #Get CT-Results here
    
    #GT detection
    i=0 
    while(i<len(CSB)):      
        tmp=CSB.pop(i)
        type=tmp[0]
        if type==1:
            flag=GT_detection(tmp[1][0],tmp[1][1])
            if(flag==1):
                # GT_Results.write(str(1)+" "+"["+str(tmp[1][0])+","+str(tmp[1][1])+"]"+'\n')
                CSB.insert(i,tmp)
                i+=1
            elif(flag==-1):
                continue
            else:
                L=flag-100
                flag=GT_detection(tmp[1][0]+L,tmp[1][1]+L)
                if(flag==1):
                    cmp=[tmp[1][0]+L,tmp[1][1]+L]
                    adder=[1,cmp]
                    CSB.insert(i,adder)
                else:
                    continue  # should do CT; here we simpify this
        else:
            CSB.insert(i,tmp)
            i+=1

Detection()
print CSB

