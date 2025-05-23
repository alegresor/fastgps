import torch 
import numpy as np
from typing import Union

class Polynomial(torch.nn.Module):
    def __init__(self,
            d:int,
            powers:torch.Tensor = None,
            coeffs:Union[torch.Tensor,float] = 1., 
            shape_coeffs:torch.Size = None,
            requires_grad_coeffs:bool = True,
            device:torch.device = "cpu",
        ):
        r"""
        Args:
            powers (torch.Tensor[P,d]): powers of the polynomial prior mean where `P` is the number of terms and `d` is the dimension.
                Some examples are below

                - `powers=None` or `powers="ZERO"` has $M(\boldsymbol{x}) = 0$
                ```python
                powers = torch.zeros((0,d),dtype=int)
                ```
                - `powers="CONSTANT"` has $M(\boldsymbol{x}) = c$
                ```python
                powers = torch.zeros((1,d),dtype=int)
                ```
                - `powers="LINEAR"` is equivalent to has $M(\boldsymbol{x}) = c_0+\sum_{j=1}^d c_j x_j$
                ```python
                powers = torch.cat([torch.zeros((1,d),dtype=int),torch.eye(d,dtype=int)],axis=0)
                ```
                - `powers="DEGREEp"` for integer `p>=0` has $M(\boldsymbol{x}) = c_0+\sum_{j=1}^d \sum_{i=1}^k c_{ij} x_j^p$
                ```python
                powers = torch.cat([torch.zeros((1,d),dtype=int)]+[i*torch.eye(d,dtype=int) for i in range(1,p+1)],axis=0)
                ```
                For example, `"DEGREE0"` is equivalent to `"CONSTANT"`, `"DEGREE1"` is equivalent to `"LINEAR"`, and `"DEGREE2"` uses $M(\boldsymbol{x}) = c_0+\sum_{j=1}^d (c_{1j} x_j + c_{2j}x_j^2)$
            poly_mean_coeffs (torch.Tensor)
        """
        super().__init__()
        assert np.isscalar(d) and d%1==0 and d>0, "dimension d must be a positive integer"
        self.d = int(d)
        self.device = torch.device(device)
        assert powers is None or np.isscalar(powers) or isinstance(powers,torch.Tensor), "powers must be None, a scalar, or a torch.Tensor"
        if powers is None:
            powers = torch.zeros((0,self.d),dtype=int,device=self.device)
        elif np.isscalar(powers):
            assert powers%1==0 and powers>=0, "scalar powers must be a non-negative int"
            powers = int(powers)
            powers = torch.cat([torch.zeros((1,self.d),dtype=int,device=self.device)]+[i*torch.eye(self.d,dtype=int,device=self.device) for i in range(1,powers+1)],axis=0)
        assert isinstance(powers,torch.Tensor) and powers.dtype==torch.int64 and powers.ndim==2 and powers.size(1)==d
        self.powers = powers
        self.r = self.powers.size(0) 
        assert np.isscalar(coeffs) or isinstance(coeffs,torch.Tensor), "coeffs must be a scalar or torch.Tensor"
        if isinstance(coeffs,torch.Tensor): shape_coeffs = coeffs.shape 
        if shape_coeffs is None: shape_coeffs = torch.Size([self.r])
        if isinstance(shape_coeffs,(list,tuple)): shape_coeffs = torch.Size(shape_coeffs)
        assert isinstance(shape_coeffs,torch.Size) and (shape_coeffs[-1]==self.r or shape_coeffs[-1]==1)
        if np.isscalar(coeffs): coeffs = coeffs*torch.ones(shape_coeffs,device=self.device)
        self.coeffs = torch.nn.Parameter(coeffs,requires_grad=requires_grad_coeffs)
    def __call__(self, x):
        assert isinstance(x,torch.Tensor) and x.ndim>=1 and x.size(-1)==self.d
        xpows = (x[...,None,:]**self.powers).prod(-1)
        idx = [Ellipsis]+[None]*(xpows.ndim-1)+[slice(None)]
        y = (self.coeffs[idx]*xpows).sum(-1)
        return y
    def integral01d(self):
        return self.coeffs/(self.powers+1).prod(-1)
        