#%%
from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import einops
import numpy as np
import pandas as pd

from local_models.KronLinear import KronLinear
from local_models.KronLinear import Kron1Linear, VeKronLinearRank1

def kron_decompose_model(model, layer_config=None):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = kron_decompose_model(module, layer_config)
            
        elif isinstance(module, nn.Conv2d):
            # todo: do the decomposition for nn.Conv2d
            pass
        elif isinstance(module, nn.Linear):
            linear_layer = module
            patch_size = layer_config['block_size']
            rank = layer_config['rank']
            if linear_layer.in_features % patch_size[0] or linear_layer.out_features % patch_size[1]:
                # e.g. a 100-class head with 8x8 blocks: keep layers that do not tile evenly dense
                print(f"keeping {name} dense: {linear_layer.in_features}x{linear_layer.out_features} "
                      f"is not divisible into {patch_size[0]}x{patch_size[1]} blocks")
                continue
            decomposed = linear2kronlinear(linear_layer, rank=rank, patch_size = patch_size)
            print(decomposed.a.shape, decomposed.b.shape)
            model._modules[name] = decomposed
    return model

def linear2kronlinear(linear_layer, rank=4, patch_size=None):
    """transfer a linear layer to a kronecker product decomposed layer

    Args:
        linear_layer (_type_): _description_
        rank (_type_, optional): _description_. Defaults to None.
        patch_size (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    assert patch_size is not None, "Patch size must be specified"
    assert rank is not None, "Rank must be specified"
    
    from local_utils.tensorops import gkpd
    
    kronlinear = KronLinear(linear_layer.weight.shape[1], linear_layer.weight.shape[0], structured_sparse=True, bias=True, patch_size=patch_size, rank=rank)
    weight_shape = linear_layer.weight.shape
    weight_shape = [weight_shape[1], weight_shape[0]]
    a_shape = [weight_shape[0]//patch_size[0], weight_shape[1]//patch_size[1]]
    b_shape = patch_size
    print(a_shape, b_shape)
    weight = linear_layer.weight
    weight = torch.transpose(weight, 0, 1)
    print(weight.shape)
    a, b = gkpd(weight, a_shape=a_shape, b_shape=b_shape)
    kronlinear.a.data = a[:rank]
    kronlinear.b.data = b[:rank]
    kronlinear.s.data = torch.ones(kronlinear.s.shape)
    if linear_layer.bias is not None:
        kronlinear.bias.data = linear_layer.bias
    return kronlinear




def k1l_decompose_model(model, layer_config=None):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = k1l_decompose_model(module, layer_config)
            
        elif isinstance(module, nn.Conv2d):
            # todo: do the decomposition for nn.Conv2d
            pass
        elif isinstance(module, nn.Linear):
            linear_layer = module
            decomposed = linear2k1l(linear_layer, config=layer_config)
            model._modules[name] = decomposed
    return model



def linear2k1l(linear_layer, config=None):
    if config is not None:
        shape_bias = config['shape_bias'] if 'shape_bias' in config else 0
        structured_sparse = config['structured_sparse'] if 'structured_sparse' in config else False
        print(config['structured_sparse'])
    kronlinear = Kron1Linear(linear_layer.weight.shape[1], linear_layer.weight.shape[0], structured_sparse=structured_sparse, shape_bias=shape_bias)
    
    return kronlinear


def k1l_update_model(model, layer_config=None):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = k1l_update_model(module, layer_config)
            
        elif isinstance(module, Kron1Linear or VeKronLinearRank1):
            module.update_W_0()
            print('update W_0')
    return model

def k1l_freezes(model):
    total_params = 0
    zero_params = 0
    #calcu the total params and zero params of s

    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            out = k1l_freezes(module)
            total_params += out[0]
            zero_params += out[1]
            
        elif isinstance(module, Kron1Linear):
            module.freeze_S()
            # return s numels and zeros numels(zero <1e-5)
            # return module.s.numel(), module.s[module.s<1e-5].numel()
            
    return total_params, zero_params


def s_l1_loss(model):
    loss = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            loss += s_l1_loss(module)
            
        elif isinstance(module, Kron1Linear):
            loss += module.s.abs().sum()
    return loss

def s_l2_loss(model):
    loss = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            loss += s_l2_loss(module)
            
        elif isinstance(module, Kron1Linear):
            loss += module.s.pow(2).sum()
    return loss



#####

def linear2vekronlinear(linear_layer, config=None):
    if config is not None:
        shape_bias = config['shape_bias'] if 'shape_bias' in config else 0
        structured_sparse = config['structured_sparse'] if 'structured_sparse' in config else False
        print(config['structured_sparse'])
    vekronlinear = VeKronLinearRank1(linear_layer.weight.shape[1], linear_layer.weight.shape[0], structured_sparse=structured_sparse, shape_bias=shape_bias, rank=1000)
    vekronlinear.W_0.data = linear_layer.weight.data
    return vekronlinear 

def vekron_decompose_model(model, layer_config=None):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = vekron_decompose_model(module, layer_config)
            
        elif isinstance(module, nn.Conv2d):
            # todo: do the decomposition for nn.Conv2d
            pass
        elif isinstance(module, nn.Linear):
            linear_layer = module
            decomposed = linear2vekronlinear(linear_layer, config=layer_config)
            model._modules[name] = decomposed
    return model








def kron_ensemble_model(model, layer_config=None):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = kron_ensemble_model(module, layer_config=layer_config)
            
        elif isinstance(module, KronLinear):
            ensemble = kronlinear2linear(module, config=layer_config)
            model._modules[name] = ensemble
    

def svd_decomposed_linear_model(linear_layer, rank, config):
    # todo : get the config settings
    # Weight should be size of (in_features, out_features)
    # A should have a size of (in_1, out_1)
    # B should have a size of (in_2, out_2)
    # input should have a size of (in_features)
    in_1, out_1 = config['a_shape']
    in_2, out_2 = config['b_shape']
    
    first_layer = nn.Linear(in_features = in_1, out_features = out_1, bias=False)
    second_layer = nn.Linear(in_features = in_1, out_features = out_2, bias=False)
    
    
    
    
    
    

#%%

    
    
def kronlinear2linear(kronlinear_layer, config=None):
    from gkpd.tensorops import kron
    a = kronlinear_layer.a * kronlinear_layer.s
    b = kronlinear_layer.b
    weight = kron(a, b)
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
    linear.weight = nn.Parameter(weight)
    linear.bias = kronlinear_layer.bias
    return linear
            

def freeze_A(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_A(module)
        else:
            if isinstance(module, KronLinear):
                module.a.requires_grad = False
            else:
                continue
    
def unfreeze_A(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_A(module)
        else:
            if isinstance(module, KronLinear):
                module.a.requires_grad = True
            else:
                continue
            
def freeze_B(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_B(module)
        else:
            if isinstance(module, KronLinear):
                module.b.requires_grad = False
            else:
                continue
            
def unfreeze_B(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_B(module)
        else:
            if isinstance(module, KronLinear):
                module.b.requires_grad = True
            else:
                continue
            
def freeze_S(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_A(module)
        else:
            if isinstance(module, KronLinear):
                module.s.requires_grad = False
            else:
                continue
            
def unfreeze_S(model):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            freeze_A(module)
        else:
            if isinstance(module, KronLinear):
                module.s.requires_grad = True
            else:
                continue
