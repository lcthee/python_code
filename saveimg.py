import os
import cv2
import numpy as np
path=r'C:\Program Files\Crazy_Pic\pic\test'
pathdir=os.listdir(path)
i=1
datamat=np.zeros((28,28),dtype=int)
for img in pathdir:
    Sr_img=cv2.imread(os.path.join(path,img),0)
    #Sr_img=cv2.cvtColor(Sr_img,cv2.COLOR_RGB2GRAY)
    try:
        Sr_img=cv2.resize(Sr_img,(28,28),interpolation=cv2.INTER_CUBIC)
        
        #cv2.imshow('test',Sr_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        npath=path+'\\train_data\\'
        isExists=os.path.exists(npath)
        if not isExists:
            os.makedirs(npath)
        cv2.imwrite(npath+str(i)+".jpg",Sr_img)
        i=i+1

    except:
        continue
    data=np.reshape(Sr_img,(28,28))
    datamat=np.concatenate((datamat,data),axis=1)
#cv2.imshow('test',datamat)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
np.savetxt(npath+'data.csv',datamat,delimiter=',')
