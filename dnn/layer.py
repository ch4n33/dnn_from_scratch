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