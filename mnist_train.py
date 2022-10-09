import numpy as np
import sys
from model_layer import *
from load_dataset import *
from tqdm import tqdm

def eval(model, x_test, y_test):
    model.eval()
    prob = model.forward(x_test)
    predict = np.argmax(prob, axis=1)
    label = np.argmax(y_test, axis=1)
    model.train()
    return sum(predict == label)/x_test.shape[0]

if __name__ == '__main__':
    x_train_path='dataset/train-images.idx3-ubyte.gz'
    y_train_path='dataset/train-labels.idx1-ubyte.gz'
    x_test_path='dataset/t10k-images.idx3-ubyte.gz'
    y_test_path='dataset/t10k-labels.idx1-ubyte.gz'
    (x_train,y_train), (x_test,y_test) = load_mnist(x_train_path, y_train_path, x_test_path, y_test_path)
    print('trainset size: ', x_train.shape[0])
    print('testset size:', x_test.shape[0])
    
    x_train_shuffle = np.random.randn(x_train.shape[0], x_train.shape[1])
    y_train_shuffle = np.random.randn(y_train.shape[0], y_train.shape[1])

    
    id_list = [i for i in range(y_train.shape[0])]
    np.random.shuffle(id_list)
    for i, id in enumerate(id_list):
        x_train_shuffle[id] = x_train[i]
        y_train_shuffle[id] = y_train[i]
    
    
    np.random.seed(2333)
    # x_train_shuffle = np.random.randn(300, 784)# only test
    # y_train_shuffle = np.random.randn(300, 10)
    # fcnn = FCNN([784, 512, 10], 0.01)
    fcnn = FCNN([784, 512, 512, 10], 0.01)
    batch_size = 1000
    max_test_acc = 0
    for epoch in tqdm(range(400)):
        k=0
        for i in range(0, 60000, batch_size):
            k+=1
            X = x_train_shuffle[i:i+batch_size]
            label = y_train_shuffle[i:i+batch_size]
            
            Y = fcnn.forward(X)
            Y = softmax(Y)
            
            grad_y = Y - label
            fcnn.backward(grad_y)
            
            if k%100==1:
                print('train: epoch={}, loss={:.10f}'.format(epoch, np.mean((Y-label)*(Y-label))))
                fcnn.eval()
                test_acc = eval(fcnn, x_test, y_test)
                if test_acc >= max_test_acc:
                    max_test_acc = test_acc
                print('test: cur_acc={:.4f}, best_acc={:.4f}'.format(test_acc, max_test_acc))
                fcnn.train()
                
