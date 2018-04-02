import numpy as np
def ReLU(x):
    return x*(x>0)
def dReLU(x):
    return 1*(x>0)
def bplayer(datain,dataout,n_in,n_out,node):
    vh=np.ones([len(datain)*n_in,1],float)#initialize the input weight matrix
    wh=np.ones([len(dataout)*node,1],float)#initialize...hidden layer's output weight
    hidb=np.ones([node,1],float)#init...hidden bias
    ob=np.ones([n_out,1],float)#init...output bias
    alpha=np.multiply(vh,datain)

    return
