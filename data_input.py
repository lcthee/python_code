#encoding=utf-8
import os
import numpy as np
import cv2
Path=r'C:\Program Files\Crazy_Pic\pic\train'
#os.mkdir('data')
i=0
for file in os.listdir(Path):
    file_path=os.path.join(Path,file)
    if os.path.isdir(file_path):
        continue
    else:
        i=i+1
        try:
            img_source=cv2.imread(file_path)
            img_gray=cv2.cvtColor(img_source,cv2.COLOR_BGR2GRAY)
            img=cv2.resize(img_gray,(16,16),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(Path+'\data\\'+'train_'+str(i)+'.png',img)
            #cv2.imshow('test',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        except:
            continue
