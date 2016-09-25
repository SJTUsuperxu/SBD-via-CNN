import scipy.io 
import numpy as np
from numpy import *

matfile = "features_fc7.mat"
data = scipy.io.loadmat(matfile)
Feats = data['feats']
#print type(Feats)
#print Feats[1,:]
#print shape(Feats)
(m,n)=shape(Feats)
vec=[]
for i in range(m):
    vec.append(Feats[i,:])
print m
np.set_printoptions(threshold='nan')
print vec[0]
