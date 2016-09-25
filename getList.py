import os
caffe_root='./'
folder = caffe_root+"examples/images"
if not os.path.exists(folder):
    os.mkdir(folder)

index=1815
#12306 anni009
#11360  anni005
#910   anni001
#1590    anni007
# 2775 anni008
#1815 BOR10_001

notes=open(os.path.join(caffe_root+"examples/_temp",'temp.txt'),'w')
for i in range(index):
    notes.write('/home/digits/XJW/caffe-new/examples/images'+'/'+str(i+1)+'.jpg'+'\n')

notes.close()
