import numpy as np
import scipy
from math import ceil,floor
import cv2
path=r'C:\Program Files\Crazy_Pic\pic\test\train_data'
datapath=open(path+'\\data.csv','r')
data=[]
def mapminmax(x):
    x=(x-np.min(x))*(1/(np.max(x)-np.min(x)))
    return x
def ReLU(x):
    return x * (x > 0)
def dReLU(x):
    return 1.* (x > 0)
for line in datapath:
    #num=[str(x) for x in line]
    data.append(line)
data=cv2.imread(r'C:\Program Files\Crazy_Pic\pic\test\train_data\1.jpg',0)
data=mapminmax(data)
def convolutionlayer(imgdata,kernelsize,layernum,stride):
    if layernum==0:
        return
    [r,c]=np.shape(imgdata)
    kernel=np.random.rand(kernelsize[0],kernelsize[1])
    samptc=int((c-kernelsize[1])/stride)+1
    samptr=int((r-kernelsize[0])/stride)+1
    #sc=floor(28/kernelsize[1])
    #sr=floor(28/kernelsize[0])
    addczero=c-int((c-kernelsize[1])/stride)+1
    addrzero=r-int((r-kernelsize[0])/stride)+1
    imgdata=np.concatenate((imgdata,np.zeros([r,addczero],int)),axis=1)
    imgdata=np.concatenate((imgdata,np.zeros([addrzero,c+addczero],int)),axis=0)
    pool=[]
    for i in range(0,int(samptr),stride):
        for j in range(0,int(samptc),stride):
            sampling=np.convolve(imgdata[i:i+kernelsize[0],j:j+kernelsize[0]],kernel,'full')
            maxnum=int(np.max(sampling))
            fv=ReLU(maxnum)
            pool=np.append(pool,fv)
    fm=np.reshape(pool,[samptr,samptc])
    fm=np.matrix(fm)
    try:
        convolutionlayer(fm,kernelsize,layernum-1,stride)
    except:
        return
    # cv2.imshow('test',fm)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #return fm
convolutionlayer(data,[5,5],2,1)
