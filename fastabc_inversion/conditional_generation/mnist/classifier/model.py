"""
Adapted from https://github.com/aaron-xichen/pytorch-playground/blob/master/mnist/model.py
Adapted by Eliane Maalouf
"""
import torch
import torch.nn as nn
from collections import OrderedDict
#import torch.utils.model_zoo as model_zoo # commented by Eliane Maalouf
#from fastabc_inversion.conditional_generation.utee import misc # commented by Eliane Maalouf
#print = misc.logger.info # commented by Eliane Maalouf

model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
}

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model= nn.Sequential(layers)
        print(self.model)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)

def mnist(input_dims=1024, n_hiddens=[256, 256], n_class=10, pretrained=None): # original code input_dims=784
    model = MLP(input_dims, n_hiddens, n_class)
    """ # original code to load pretrained weights; commented by Eliane Maalouf
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    """
    if pretrained is not None:
        # pretrained variable is expected to be the path to the weights
        state_dict = pretrained
        model.load_state_dict(torch.load(state_dict))
    return model
