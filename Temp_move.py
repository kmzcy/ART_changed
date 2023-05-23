from multiprocessing.dummy import active_children
import torch.nn as nn
import torch
import numpy as np
    
class T_change(nn.Module):
    def __init__(self, input_size, output_B = 64):
        """
        用于模板偏移, 目前也包含1个中间层, 中间层包含256, 中间层的激活函数为ReLU
        输入：
        input: 
          input_size:输入向量的尺寸
          output_B:输出的二进制哈希码的长度
        """
        super(T_change, self).__init__()
        
        # self.W1 = nn.Parameter(torch.randn(input_size, 4096, requires_grad=True))
        # self.b1 = nn.Parameter(torch.zeros(4096, requires_grad=True))
        # self.W2 = nn.Parameter(torch.randn(4096, output_B, requires_grad=True))
        # self.b2 = nn.Parameter(torch.zeros(output_B, requires_grad=True))

        self.W1 = nn.Parameter(torch.randn(input_size, output_B, requires_grad=True))
        self.b1 = nn.Parameter(torch.zeros(output_B, requires_grad=True))
        
        self.coefficient1 = nn.Parameter(torch.Tensor([0.1]))
        self.coefficient2 = nn.Parameter(torch.Tensor([0.9]))
        self.coefficient1.requires_grad_()
        self.coefficient2.requires_grad_()
        
        # self.params = [self.W1, self.b1, self.W2, self.b2, self.coefficient1, self.coefficient2]
        self.params = [self.W1, self.b1, self.coefficient1, self.coefficient2]

    def relu(self, X):
        a = torch.zeros_like(X)
        return torch.max(X, a)

    def forward(self, x):
        x1 = torch.tanh(x@self.W1 + self.b1)
        x2 = self.coefficient1 * x1 + self.coefficient2 * x
        x3 = torch.sigmoid(x2)
        return x3
    
    def parameters(self):
        return self.params
