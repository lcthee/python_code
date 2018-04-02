#coding=gbk
import numpy as np
import os
import cv2
path=r'C:\Program Files\Crazy_Pic\pic\bike'
i=1
pathdir=os.listdir(path)
for imgname in pathdir:
    oldpath=os.path.join(path,imgname)
    newpath=os.path.join(path,str(i)+".jpg")
    i=i+1
    try:
        if (os.path.splitext(imgname)[-1]==".jpg"):
            os.rename(oldpath,newpath)
    except:
        break
pathdir=os.listdir(path)
i=1
for img in pathdir:
    Sr_img=cv2.imread(os.path.join(path,img),0)
    #Sr_img=cv2.cvtColor(Sr_img,cv2.COLOR_RGB2GRAY)
    Sr_img=cv2.resize(Sr_img,(28,28))
    #cv2.imshow('test',Sr_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(path,"train_data\\"+str(i)+".jpg"),Sr_img)
    i=i+1
