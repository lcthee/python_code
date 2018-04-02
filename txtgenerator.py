#encoding=utf-8
import os
path=r'C:\Users\lenovo\Desktop\python_action\data\data1'
label={}
i=0
for file in os.listdir(path):
    file_path=os.path.join(path,file)
    if os.path.isdir(file_path):
        continue
    else:
        filename=file.split('_')
        name=filename[0]
        if label.has_key(name):
            n=label.get(name)
        else:
            i=i+1
            label[name]=i
            n=label.get(name)
        with open('train.txt','a') as f:
            f.write('data1\\'+file+'\t'+str(n)+'\n')
