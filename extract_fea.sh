#!/usr/bin/env sh
# args for EXTRACT_FEATURE
TOOL=./build/tools
MODEL=./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel #下载得到的caffe model
PROTOTXT=./examples/_temp/imagenet_val.prototxt # 网络定义
LAYER=fc6 # 提取层的名字，如提取fc7等    
LEVELDB=./examples/_temp/features_fc6 # 保存的leveldb路径   
BATCHSIZE=33  #
# 160*71 anni005
# 13*70 anni001
# 30*53 anni007
# 37*75 anni008
# 33*55 BOR_1
# DIM=290400 # feature长度，conv1
# DIM=43264 # conv5
# args for LEVELDB to MAT
DIM=4096 # 需要手工计算feature长度
OUT=./examples/_temp/features_fc7_BOR1_fc6.mat #.mat文件保存路径   #
BATCHNUM=55 # Eg: BATCHNUM*BATCH_SIZE = Number_Imgs = 910       #

$TOOL/extract_features.bin  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHSIZE lmdb 
python lmdb2mat.py $LEVELDB $BATCHNUM  $BATCHSIZE $DIM $OUT
