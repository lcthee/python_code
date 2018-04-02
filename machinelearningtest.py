from numpy import *
from matplotlib.pyplot import *
import operator
from math import log
dataMat=[]
labelMat=[]
fr=open(r'C:\Users\lenovo\Desktop\machinelearninginaction\Ch06\testSet.txt')
for line in fr.readlines():
    lineArr=line.strip().split('\t')
    dataMat.append([float(lineArr[0]),float(lineArr[1])])
    labelMat.append(float(lineArr[2]))
datamat=mat(dataMat)
labelmat=mat(labelMat)
minvals=datamat.min(0)
maxvals=datamat.max(0)
ranges=maxvals-minvals
normdata=zeros(shape(datamat))
m=datamat.shape[0]
n=datamat.shape[1]
normdata=datamat-tile(minvals,(m,1))
normdata=normdata/tile(ranges,(m,1))
plot(normdata[:,0],normdata[:,1],'.')
# grid(True)
# show()
#knn test---------------------------------------------------------------------
group=array([[1.0,1.0],[1.0,1.1],[0,0],[0,0.1]])
labels=['A','A','B','B']
datasetsize=normdata.shape[0]
for i in range(datasetsize):
    diffMat=tile(array(normdata[i,:]),(4,1))-group
    sqdiff=diffMat**2
    distances=sqdiff.sum(axis=1)
    sortindex=distances.argsort()
    classcount={}
    for j in range(3):
        votelabel=labels[sortindex[j]]
        classcount[votelabel]=classcount.get(votelabel,0)+1
    sortedclasscount=sorted(classcount.iteritems(),key=operator.itemgetter(1),reverse=True)
    print sortedclasscount[0][0]
#-----------------------------------------------------------------------------
#tree test
dataSet=[[1, 1, 'yes'],
         [1, 1, 'yes'],
         [1, 0, 'no'],
         [0, 1, 'no'],
         [0, 1, 'no']]
labels = ['no surfacing','flippers']
numEntries=len(dataSet)
labelCount={}
for featVec in dataSet:
    currentLabel=featVec[-1]
    labelCount[currentLabel]=labelCount.get(currentLabel,0)+1
EntD=0.0
for key in labelCount:
    prob=float(labelCount[key])/numEntries
    EntD = EntD-prob*log(prob,2)
#Gain
datam=mat(dataSet)
rows,column=shape(dataSet)
GainD=[];
for i in range(column-1):
    data=datam[:,i]
    label=datam[:,-1]
    data=c_[data,label]
    labelC = {}
    d=data[:,0]
    uniquelabel=set(d)
    l=len(uniquelabel)
for k in range(l):
    for f in data:
        currentLabel=f[uniquelabel[l]]
        labelC[currentLabel]=labelC.get(currentLabel,0)+1
    Gain=0.0
    for key in labelC:
        prob=float(labelC[key])/rows
        Gain = Gain-prob*log(prob,2)
    GainD.append(Gain)
GainD=array(GainD)
order=GainD.argsort()
treeroot=order[-1]