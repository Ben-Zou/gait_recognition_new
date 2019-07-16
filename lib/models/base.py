import torch.nn as nn
import numpy as np

from abc import abstractmethod

class BaseModel(nn.Module):
    """
    BaseModel,
    """

    def __init__(self,):
        super().__init__()

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def init_states(self):
        pass

    @abstractmethod
    def forward_train(self,x):
        pass

    @abstractmethod
    def forward_test(self,x):
        pass

    def forward(self,x,train_mode):
        if train_mode:
            return self.forward_train(x)
        else:
            return self.forward_test(x)

















##
