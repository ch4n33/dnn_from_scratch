from dnn import *
from sklearn.datasets import fetch_openml 
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import cupy as cp
import numpy as np
from matplotlib.pylab import plt
import pickle
from tqdm import tqdm
import json


with open('3d\config.json','r') as conf:
    config = json.load(conf)
epoch_size = config ['epoch_size']
save_every_n_epoch = config ['save_every_n_epoch']

input_size = config ['input_size']
layer_num = config ['layer_num']
hidden_size = config ['hidden_size']
output_size = config ['output_size']
batch_size = config ['batch_size']

learning_rate = config ['learning_rate']
momentum = config ['momentum']
weight_decay = config ['weight_decay']

data_mean = 0.13092535192648574
data_std = 0.30844852402703143

def load_data(batch_size=10):
    # 데이터 불러오기
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

    mnist.target = mnist.target.astype(np.int8)
    X = mnist.data / 255  # 0-255값을 [0,1] 구간으로 정?규화
    y = mnist.target
    
    data_mean = np.mean(X)
    data_std =  np.std(X)
    X -= data_mean
    X /= data_std

    # 훈련 데이터, 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=0)

    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    ds_train = TensorDataset(X_train, y_train)
    ds_test = TensorDataset(X_test, y_test)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    return loader_train, loader_test 
    

def test(model, loader_test, batch_size=1):
    correct_num = 0
    for data, targets in loader_test:
        data = cp.array(data.numpy())
        targets_np = cp.array(targets.numpy())
        
        targets_encode = one_hot_encode(targets_np, num_classes=output_size)
        output = model.forward(data)
        
        predicted = cp.argmax(output, axis=1)
        correct_num += len(predicted[(targets_np == predicted)])
    print(f'test batch{batch_size}, {len(loader_test)}')
    return correct_num / (batch_size * len(loader_test))


def train(model, optimizer:Optimizer, loader_train, loader_test, epoch, batch_size=1):
    print(f'epoch : {epoch}, batch_size : {batch_size}')
    softmax = Softmax()
    error_list = []
    error_list_each = []
    acc_list = []
    test_acc_list = []
    for e in range(epoch):
        error_list_epoch = []
        correct_num=0
        for data, targets in tqdm(loader_train, mininterval=2):
            data = cp.array(data.numpy())
            targets_np = cp.array(targets.numpy())
            
            targets_encode = one_hot_encode(targets_np, num_classes=output_size)
            
            optimizer.zero_grad()
            output = model.forward(data)
            result = softmax.forward(output) # loss value
            error = softmax.error(targets_encode)            
            error_list_epoch.append(error.get())
            back = softmax.backward()
            model.backward(back)
            optimizer.step()

            predicted = cp.argmax(output, axis=1)
            correct_num += len(predicted[(targets_np == predicted)])
        if (e+1)%save_every_n_epoch == 0:
            with open(f'C:\\Users\\Jongchan\\Desktop\\Folder\\jolf\\dnn_saved_models\\{model.name}_epoch{e}.pickle', 'wb') as fw:
                pickle.dump(model, fw)
            
        test_acc_list.append(test(model, loader_test, batch_size))
        acc_list.append(correct_num/(batch_size * len(loader_train)))
        error_list_each += error_list_epoch
        error_list.append(np.mean(error_list_epoch))
        print(f'epoch{e} end, train_acc:{acc_list[-1]}, test_acc:{test_acc_list[-1]}, train_loss_mean:{error_list[-1]}')
    
    fig,ax=plt.subplots(1,3)
    ax[0].plot(error_list, label='loss')
    ax[1].plot(error_list_each, label='loss')
    ax[2].plot(acc_list, label='train')
    ax[2].plot(test_acc_list, label='test')
    
    ax[0].set_xlabel('epoch')
    ax[1].set_xlabel('batch')
    ax[2].set_xlabel('epoch')
    
    ax[2].set_ylabel('accuracy')
    
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    
    plt.show()        
    

def main():
    #np.seterr(invalid='raise')
    print('start')
    model = nn('MNIST Classifier Model')
    
    model.addModule(Linear(input_size, hidden_size, name=f'layer{0}'))
    model.addModule(Sigmoid())
    for i in range(1,layer_num-1):
        model.addModule(Linear(hidden_size, hidden_size, name=f'layer{i}'))
        model.addModule(ReLU())
    model.addModule(Linear(hidden_size, output_size, name=f'layer{layer_num-1}'))
    
    #optimizer = SGD(model.parameters(), learning_rate=learning_rate)
    optimizer = SGD(model.parameters(), learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    print(model)

    loader_train, loader_test  = load_data(batch_size=batch_size)
  
    train(model, optimizer, loader_train, loader_test, epoch=epoch_size, batch_size=batch_size)


if __name__ == '__main__':
    main()