import matplotlib.pyplot as plt  
import caffe   
caffe.set_device(0)  
caffe.set_mode_gpu()   
# ʹ��SGDSolver��������ݶ��½��㷨  
solver = caffe.SGDSolver('/home/xxx/mnist/solver.prototxt')  
  
# �ȼ���solver�ļ��е�max_iter�������������  
niter = 10000 

# ÿ��100���ռ�һ��loss����  
display= 100  
  
# ÿ�β��Խ���100�ν��� 
test_iter = 100

# ÿ500��ѵ������һ�β���
test_interval =500
  
#��ʼ�� 
train_loss = zeros(ceil(niter * 1.0 / display))   
test_loss = zeros(ceil(niter * 1.0 / test_interval))  
test_acc = zeros(ceil(niter * 1.0 / test_interval))  
  
# ��������  
_train_loss = 0; _test_loss = 0; _accuracy = 0  
# ���н���  
for it in range(niter):  
    # ����һ�ν���  
    solver.step(1)  
    # ͳ��train loss  
    _train_loss += solver.net.blobs['SoftmaxWithLoss1'].data  
    if it % display == 0:  
        # ����ƽ��train loss  
        train_loss[it // display] = _train_loss / display  
        _train_loss = 0  
  
    if it % test_interval == 0:  
        for test_it in range(test_iter):  
            # ����һ�β���  
            solver.test_nets[0].forward()  
            # ����test loss  
            _test_loss += solver.test_nets[0].blobs['SoftmaxWithLoss1'].data  
            # ����test accuracy  
            _accuracy += solver.test_nets[0].blobs['Accuracy1'].data  
        # ����ƽ��test loss  
        test_loss[it / test_interval] = _test_loss / test_iter  
        # ����ƽ��test accuracy  
        test_acc[it / test_interval] = _accuracy / test_iter  
        _test_loss = 0  
        _accuracy = 0  
  
# ����train loss��test loss��accuracy����  
print '\nplot the train loss and test accuracy\n'  
_, ax1 = plt.subplots()  
ax2 = ax1.twinx()  
  
# train loss -> ��ɫ  
ax1.plot(display * arange(len(train_loss)), train_loss, 'g')  
# test loss -> ��ɫ  
ax1.plot(test_interval * arange(len(test_loss)), test_loss, 'y')  
# test accuracy -> ��ɫ  
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')  
  
ax1.set_xlabel('iteration')  
ax1.set_ylabel('loss')  
ax2.set_ylabel('accuracy')  
plt.show()