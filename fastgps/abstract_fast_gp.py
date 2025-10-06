import torch
import numpy as np
from typing import Union,List
from .abstract_gp import AbstractGP
from .util import _K1PartsSeq,_FastInverseLogDetCache,_LamCaches,_YtildeCache
import os 

class AbstractFastGP(AbstractGP):
    def __init__(self,
            ft,
            ift,
            omega,
            *args,
            **kwargs
        ):
        super().__init__(*args,**kwargs)
        # fast transforms 
        self.ft_unstable = ft
        self.ift_unstable = ift
        self.omega = omega
        # storage and dynamic caches
        self.k1parts_seq = np.array([[_K1PartsSeq(self,l0,l1,*self.derivatives_cross[l0][l1]) if l1>=l0 else None for l1 in range(self.num_tasks)] for l0 in range(self.num_tasks)],dtype=object)
        self.lam_caches = np.array([[_LamCaches(self,l0,l1,*self.derivatives_cross[l0][l1],self.derivatives_coeffs_cross[l0][l1]) if l1>=l0 else None for l1 in range(self.num_tasks)] for l0 in range(self.num_tasks)],dtype=object)
        self.ytilde_cache = np.array([_YtildeCache(self,i) for i in range(self.num_tasks)],dtype=object)
    def get_x_next(self, n:Union[int,torch.Tensor], task:Union[int,torch.Tensor]=None):
        n_og = n 
        if isinstance(n,(int,np.int64)): n = torch.tensor([n],dtype=int,device=self.device) 
        if isinstance(n,list): n = torch.tensor(n,dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and torch.logical_or(n==0,n&(n-1)==0).all(), "maximum sequence index must be a power of 2"
        return super().get_x_next(n=n_og,task=task)
    def add_y_next(self, y_next:Union[torch.Tensor,List], task:Union[int,torch.Tensor]=None):
        super().add_y_next(y_next=y_next,task=task)
        assert torch.logical_or(self.n==0,(self.n&(self.n-1)==0)).all(), "total samples must be power of 2"
    def post_var(self, x:torch.Tensor, task:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True):
        n_og = n 
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and (n&(n-1)==0).all() and (n>=self.n).all(), "require n are all power of two greater than or equal to self.n"
        return super().post_var(x=x,task=task,n=n_og,eval=eval)
    def post_cov(self, x0:torch.Tensor, x1:torch.Tensor, task0:Union[int,torch.Tensor]=None, task1:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True):
        n_og = n 
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and (n&(n-1)==0).all() and (n>=self.n).all(), "require n are all power of two"
        return super().post_cov(x0=x0,x1=x1,task0=task0,task1=task1,n=n_og,eval=eval)
    def get_inv_log_det_cache(self, n=None):
        if n is None: n = self.n
        assert isinstance(n,torch.Tensor) and n.shape==(self.num_tasks,) and (n>=self.n).all()
        ntup = tuple(n.tolist())
        if ntup not in self.inv_log_det_cache_dict.keys():
            self.inv_log_det_cache_dict[ntup] = _FastInverseLogDetCache(n)
        return self.inv_log_det_cache_dict[ntup]
    def post_cubature_mean(self, task:Union[int,torch.Tensor]=None, eval:bool=True):
        kmat_tasks = self.kernel.taskmat
        if self.solo_task: # numerically stable computation
            lam = self.get_lam(0,0)[...,[0]].real+self.noise/torch.sqrt(self.n)
        else:
            coeffs = self.coeffs
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int,device=self.device)
        if isinstance(task,list): task = torch.tensor(task,dtype=int,device=self.device)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        if self.solo_task: # numerically stable computation
            pcmean = self.prior_mean[...,task]+(self._y[0].mean(-1,keepdim=True)-self.prior_mean[...,task])*torch.sqrt(self.n)/lam
        else:
            coeffs_split = coeffs.split(self.n.tolist(),-1)
            coeffs_split_scaled = [(self.kernel.base_kernel.scale*coeffs_split[l])[...,None,:]*kmat_tasks[...,task,l,None] for l in range(self.num_tasks)]
            pcmean = self.prior_mean[...,task]+torch.cat(coeffs_split_scaled,-1).sum(-1)
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pcmean[...,0] if inttask else pcmean
    def post_cubature_var(self, task:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True):
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and (n&(n-1)==0).all() and (n>=self.n).all(), "require n are all power of two greater than or equal to self.n"
        kmat_tasks = self.kernel.taskmat
        if self.solo_task: # numerically stable computation
            lamm1 = self.get_lam_m1(0,0,n=n)[...,[0]].real+self.noise/torch.sqrt(n)
        else:
            inv_log_det_cache = self.get_inv_log_det_cache(n)
            inv = inv_log_det_cache(self)[0]
            to = inv_log_det_cache.task_order
            nord = n[to]
            mvec = torch.hstack([torch.zeros(1,device=self.device),(nord/nord[-1]).cumsum(0)]).to(int)[:-1]
            nsqrts = torch.sqrt(nord[:,None]*nord[None,:])
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int,device=self.device)
        if isinstance(task,list): task = torch.tensor(task,dtype=int,device=self.device)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        s = self.kernel.base_kernel.scale
        if self.solo_task: # stable computation
            pcvar = s*lamm1/(lamm1+s*torch.sqrt(n))
        else:
            inv_cut = inv[...,mvec,:,:][...,:,mvec,:][...,0]
            kmat_tasks_left = kmat_tasks[...,task,:][...,:,to].to(self._FTOUTDTYPE)
            kmat_tasks_right = kmat_tasks[...,to,:][...,:,task].to(self._FTOUTDTYPE)
            term = torch.einsum("...ij,...jk,...ki->...i",kmat_tasks_left,nsqrts*inv_cut,kmat_tasks_right).real
            pcvar = s*kmat_tasks[...,task,task]-s**2*term
        pcvar[pcvar<0] = 0.
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pcvar[...,0] if inttask else pcvar
    def post_cubature_cov(self, task0:Union[int,torch.Tensor]=None, task1:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True):
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and (n&(n-1)==0).all() and (n>=self.n).all(), "require n are all power of two greater than or equal to self.n"
        kmat_tasks = self.kernel.taskmat
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        inv = inv_log_det_cache(self)[0]
        to = inv_log_det_cache.task_order
        nord = n[to]
        mvec = torch.hstack([torch.zeros(1,device=self.device),(nord/nord[-1]).cumsum(0)]).to(int)[:-1]
        nsqrts = torch.sqrt(nord[:,None]*nord[None,:])
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task0 is None: task0 = self.default_task
        inttask0 = isinstance(task0,int)
        if inttask0: task0 = torch.tensor([task0],dtype=int,device=self.device)
        if isinstance(task0,list): task0 = torch.tensor(task0,dtype=int,device=self.device)
        assert task0.ndim==1 and (task0>=0).all() and (task0<self.num_tasks).all()
        if task1 is None: task1 = self.default_task
        inttask1 = isinstance(task1,int)
        if inttask1: task1 = torch.tensor([task1],dtype=int,device=self.device)
        if isinstance(task1,list): task1 = torch.tensor(task1,dtype=int,device=self.device)
        assert task1.ndim==1 and (task1>=0).all() and (task1<self.num_tasks).all()
        equal = torch.equal(task0,task1)
        inv_cut = inv[...,mvec,:,:][...,:,mvec,:][...,0]
        kmat_tasks_left = kmat_tasks[...,task0,:][...,:,to].to(self._FTOUTDTYPE)
        kmat_tasks_right = kmat_tasks[...,to,:][...,:,task1].to(self._FTOUTDTYPE)
        term = torch.einsum("...ij,...jk,...kl->...il",kmat_tasks_left,nsqrts*inv_cut,kmat_tasks_right).real
        s = self.kernel.base_kernel.scale
        pccov = s[...,None]*kmat_tasks[...,task0,:][...,:,task1]-s[...,None]**2*term
        if equal:
            tvec = torch.arange(pccov.size(-1))
            diag = pccov[...,tvec,tvec]
            diag[diag<0] = 0. 
            pccov[...,tvec,tvec] = diag
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        if inttask0 and inttask1:
            return pccov[...,0,0]
        elif inttask0 and not inttask1:
            return pccov[...,0,:]
        elif not inttask0 and inttask1:
            return pccov[...,:,0]
        else: #not inttask0 and not inttask1
            return pccov       
    def get_lam(self, task0, task1, n=None):
        assert 0<=task0<self.num_tasks
        assert 0<=task1<self.num_tasks
        if n is None: m = int(self.m[task0])
        else: m = -1 if n==0 else int(np.log2(int(n)))
        return self.lam_caches[task0,task1].getitem(self,m)
    def get_lam_m1(self, task0, task1, n=None):
        assert 0<=task0<self.num_tasks
        assert 0<=task1<self.num_tasks
        if n is None: m = int(self.m[task0])
        else: m = -1 if n==0 else int(np.log2(int(n)))
        return self.lam_caches[task0,task1].getitem_m1(self,m)
    def get_k1parts(self, task0, task1, n=None):
        assert 0<=task0<self.num_tasks
        assert 0<=task1<self.num_tasks
        if n is None: n = self.n[task0]
        assert n>=0
        return self.k1parts_seq[task0,task1].getitem(self,slice(0,n))
    def get_ytilde(self, task):
        assert 0<=task<self.num_tasks
        return self.ytilde_cache[task](self)
    def get_inv_log_det(self, n=None):
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        return inv_log_det_cache(self)
    def ft(self, x):
        """
        One dimensional fast transform along the last dimenions. 
            For `FastGPLattice` this is the orthonormal Fast Fourier Transform (FFT). 
            For `FastGPDigitalNetB2` this is the orthonormal Fast Walsh Hadamard Transform (FWHT). 
        
        Args: 
            x (torch.Tensor): inputs to be transformed along the last dimension. Require `n = x.size(-1)` is a power of 2. 
        
        Returns: 
            y (torch.Tensor): transformed inputs with the same shape as `x` 
        """
        xmean = x.mean(-1)
        y = self.ft_unstable(x-xmean[...,None])
        y[...,0] += xmean*np.sqrt(x.size(-1))
        return y
    def ift(self, x):
        """
        One dimensional inverse fast transform along the last dimenions. 
            For `FastGPLattice` this is the orthonormal Inverse Fast Fourier Transform (IFFT). 
            For `FastGPDigitalNetB2` this is the orthonormal Fast Walsh Hadamard Transform (FWHT). 
        
        Args: 
            x (torch.Tensor): inputs to be transformed along the last dimension. Require `n = x.size(-1)` is a power of 2. 
        
        Returns: 
            y (torch.Tensor): transformed inputs with the same shape as `x` 
        """
        xmean = x.mean(-1)
        y = self.ift_unstable(x-xmean[...,None])
        y[...,0] += xmean*np.sqrt(x.size(-1))
        return y
