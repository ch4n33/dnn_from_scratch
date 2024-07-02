import cupy as cp
from .optim import Updatable
from .nn import nnModule

class Linear(Updatable):
    def __init__(self, input_size, output_size, name = ''):
        stddev = cp.sqrt(2/input_size)
        super().__init__(f'Linear ({str(input_size)} -> {str(output_size)}) {name}', 
                         input_size, 
                         output_size,
                         weight = stddev * cp.random.randn(input_size, output_size), 
                         bias = stddev * cp.zeros(output_size))   
    
    def forward(self, x):
        self.x = x
        self.y = cp.dot(x, self.weight) + self.bias
        return self.y
    
    def backward(self, d_y):
        self.d_W = cp.dot(self.x.T, d_y)
        self.d_b = cp.sum(d_y, axis=0)
        d_x = cp.dot(d_y, self.weight.T)        
        
        return d_x


class LayerNorm():
    pass


# 사용되지 않음, 작동 보장되지 않음
class Conv3x3(Updatable):
    def init(self, num_filters=8, name = ''):
        filter_size=3
        stride=1
        stddev = cp.sqrt(1/9)
        
        # input size, output size를 입력받는 메커니즘을 삭제해야 conv layer를 구현할 수 있음
        input_size = (123, 123)
        output_size = (123 -2, 123 -2, num_filters)
        
        super().__init__(f'Conv{str(filter_size)}x{str(filter_size)}, filter*{str(num_filters)} {name}', 
                         input_size, 
                         output_size,
                         weight = stddev * cp.random.randn(filter_size, filter_size, num_filters), 
                         bias = None)   
        self.stride = stride
        
    def forward(self, x):
        self.x = x
        y = cp.zeros(self.output_size)
        for i in self.output_size[0]:
            for j in self.output_size[1]:
                for filter in self.output_size[2]:
                    y[i, j, filter] = cp.sum(x[i:i+3, j:j+3] * self.weight[:, :, filter])
        return y
    
    def backward(self, d_y):
        self.d_W = cp.zeros_like(self.weight)
        d_x = cp.zeros_like(self.x)
        for i in self.output_size[0]:
            for j in self.output_size[1]:
                for filter in self.output_size[2]:
                    self.d_W[:, :, filter] += cp.sum(x[i:i+3, j:j+3] * d_y[i, j, filter])
                    d_x[i:i+3, j:j+3] += self.weight[:, :, filter] * d_y[i, j, filter]
        return d_x