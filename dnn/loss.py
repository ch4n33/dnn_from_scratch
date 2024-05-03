from typing import Any
import cupy as cp


class Loss:
    def __call__(self, y_, y) -> Any: # y : 모델 추론 결과, y_ : 정답 레이블
        pass


class MSE(Loss):   
    def __call__(self, y_, y) -> Any:
        return cp.mean((y_ - y)**2, axis=0)


class CrossEntropyLoss(Loss):
    def __call__(self, y_, y) -> Any:
        return -cp.sum(y_ * cp.log(y + 1e-9)) / y.shape[0]

def one_hot_encode(labels, num_classes):
    """
    :param labels: 레이블 배열
    :param num_classes: 클래스의 수
    :return: one-hot encoding된 레이블 배열
    """
    num_labels = labels.shape[0]  # 레이블의 수
    index_offset = cp.arange(num_labels)
    one_hot_labels = cp.zeros((num_labels, num_classes))
    one_hot_labels[index_offset, labels] = 1
    return one_hot_labels


class Softmax: # 리팩토링 필요 
    def __init__(self):
        self.loss = None  # 손실
        self.y = None     # softmax의 출력
        self.t = None     # 정답 레이블(원-핫 벡터)
        self.crossentropyloss = CrossEntropyLoss()
    
    def softmax(self, x):
        exp = cp.exp(x - cp.max(x, axis = 1, keepdims = True))
        out = exp / cp.sum(exp, axis = 1, keepdims = True)
        return out
    
    def error(self, y_):
        self.y_ = y_
        self.loss = self.crossentropyloss(y_, self.y)
        return self.loss
    
    def forward(self, x):
        self.y = self.softmax(x)
        return self.y
    
    def backward(self, d_y=1):
        batch_size = self.y_.shape[0]
        d_x = (self.y - self.y_)/batch_size
        return d_x
    