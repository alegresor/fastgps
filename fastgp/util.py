import torch

def optional_requires_grad_func(func):
    def func_requires_grad(*args,requires_grad=False,**kwargs):
        if requires_grad:
            return func(*args,**kwargs)
        else:
            with torch.no_grad():
                return func(*args,**kwargs)
    return func_requires_grad