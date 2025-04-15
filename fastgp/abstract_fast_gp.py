from .util import (
    _FastInverseLogDetCache,
    _K1PartsSeq,
    _LamCaches,
    _YtildeCache)
import torch
import numpy as np 
import os
from typing import Union,List
from .abstract_gp import AbstractGP

class AbstractFastGP(AbstractGP):
    def __init__(self,
            alpha,
            ft,
            ift,
            *args,
            **kwargs
        ):
        super().__init__(*args,**kwargs)
        # alpha
        assert (np.isscalar(alpha) and alpha%1==0) or (isinstance(alpha,torch.Tensor) and alpha.shape==(self.d,)), "alpha should be an int or a torch.Tensor of length d"
        if np.isscalar(alpha):
            alpha = int(alpha)*torch.ones(self.d,dtype=int,device=self.device)
        self.alpha = alpha
        # fast transforms 
        self.ft_unstable = ft
        self.ift_unstable = ift
        # storage and dynamic caches
        self.k1parts_seq = np.array([[_K1PartsSeq(self,self.xxb_seqs[l0],self.xxb_seqs[l1],self.derivatives[l0],self.derivatives[l1]) if l1>=l0 else None for l1 in range(self.num_tasks)] for l0 in range(self.num_tasks)],dtype=object)
        self.lam_caches = np.array([[_LamCaches(self,l0,l1,self.derivatives[l0],self.derivatives[l1],self.derivatives_coeffs[l0],self.derivatives_coeffs[l1]) if l1>=l0 else None for l1 in range(self.num_tasks)] for l0 in range(self.num_tasks)],dtype=object)
        self.ytilde_cache = np.array([_YtildeCache(self,i) for i in range(self.num_tasks)],dtype=object)
    def fit(self,
        iterations:int = 5000,
        lr:float = 1e-1,
        optimizer:torch.optim.Optimizer = None,
        stop_crit_improvement_threshold:float = 1e-1,
        stop_crit_wait_iterations:int = 10,
        store_mll_hist:bool = True, 
        store_scale_hist:bool = True, 
        store_lengthscales_hist:bool = True,
        store_noise_hist:bool = True,
        store_task_kernel_hist:bool = True,
        verbose:int = 5,
        verbose_indent:int = 4,
        ):
        """
        Args:
            iterations (int): number of optimization iterations
            lr (float): learning rate for default optimizer
            optimizer (torch.optim.Optimizer): optimizer defaulted to `torch.optim.Rprop(self.parameters(),lr=lr)`
            stop_crit_improvement_threshold (float): stop fitting when the maximum number of iterations is reached or the best mll is note reduced by `stop_crit_improvement_threshold` for `stop_crit_wait_iterations` iterations 
            stop_crit_wait_iterations (int): number of iterations to wait for improved mll before early stopping, see the argument description for `stop_crit_improvement_threshold`
            store_mll_hist (bool): if `True`, store and return iteration data for mll
            store_scale_hist (bool): if `True`, store and return iteration data for the kernel scale parameter
            store_lengthscales_hist (bool): if `True`, store and return iteration data for the kernel lengthscale parameters
            store_noise_hist (bool): if `True`, store and return iteration data for noise
            store_task_kernel_hist (bool): if `True`, store and return iteration data for the task kernel
            verbose (int): log every `verbose` iterations, set to `0` for silent mode
            verbose_indent (int): size of the indent to be applied when logging, helpful for logging multiple models
            
        Returns:
            data (dict): iteration data which, dependeing on storage arguments, may include keys in 
                ```python
                ["mll_hist","scale_hist","lengthscales_hist","noise_hist","task_kernel_hist"]
                ```
        """
        assert (self.n>0).any(), "cannot fit without data"
        assert isinstance(iterations,int) and iterations>=0
        if optimizer is None:
            assert np.isscalar(lr) and lr>0, "require lr is a positive float"
            optimizer = torch.optim.Rprop(self.parameters(),lr=lr)
        assert isinstance(optimizer,torch.optim.Optimizer)
        assert isinstance(store_mll_hist,bool), "require bool store_mll_hist" 
        assert isinstance(store_scale_hist,bool), "require bool store_scale_hist" 
        assert isinstance(store_lengthscales_hist,bool), "require bool store_lengthscales_hist" 
        assert isinstance(store_noise_hist,bool), "require bool store_noise_hist"
        assert isinstance(store_task_kernel_hist,bool), "require bool store_task_kernel_hist"
        assert (isinstance(verbose,int) or isinstance(verbose,bool)) and verbose>=0, "require verbose is a non-negative int"
        assert isinstance(verbose_indent,int) and verbose_indent>=0, "require verbose_indent is a non-negative int"
        assert np.isscalar(stop_crit_improvement_threshold) and 0<stop_crit_improvement_threshold, "require stop_crit_improvement_threshold is a positive float"
        assert isinstance(stop_crit_wait_iterations,int) and stop_crit_wait_iterations>0
        logtol = np.log(1+stop_crit_improvement_threshold)
        if store_mll_hist:
            mll_hist = torch.empty(iterations+1)
        store_scale_hist = store_scale_hist and self.raw_scale.requires_grad
        store_lengthscales_hist = store_lengthscales_hist and self.raw_lengthscales.requires_grad
        store_noise_hist = store_noise_hist and self.raw_noise.requires_grad
        store_task_kernel_hist = store_task_kernel_hist and (self.raw_factor_task_kernel.requires_grad or self.raw_noise_task_kernel.requires_grad)
        if store_scale_hist: scale_hist = torch.empty(torch.Size([iterations+1])+self.raw_scale.shape)
        if store_lengthscales_hist: lengthscales_hist = torch.empty(torch.Size([iterations+1])+self.raw_lengthscales.shape)
        if store_noise_hist: noise_hist = torch.empty(torch.Size([iterations+1])+self.raw_noise.shape)
        if store_task_kernel_hist: task_kernel_hist = torch.empty(torch.Size([iterations+1])+self.gram_matrix_tasks.shape)
        if verbose:
            _s = "%16s | %-10s | %-10s | %-10s"%("iter of %.1e"%iterations,"NMLL","norm term","logdet term")
            print(" "*verbose_indent+_s)
            print(" "*verbose_indent+"~"*len(_s))
        mll_const = self.d_out*self.n.sum()*np.log(2*np.pi)
        stop_crit_best_mll = torch.inf 
        stop_crit_save_mll = torch.inf 
        stop_crit_iterations_without_improvement_mll = 0
        ytildes = [self.get_ytilde(i) for i in range(self.num_tasks)]
        ytildescat = torch.cat(ytildes,dim=-1)
        os.environ["FASTGP_FORCE_RECOMPILE"] = "True"
        inv_log_det_cache = self.get_inv_log_det_cache()
        for i in range(iterations+1):
            ztildes,logdet = inv_log_det_cache._gram_matrix_solve_tilde_to_tilde(ytildes)
            ztildescat = torch.cat(ztildes,dim=-1)
            norm_term = (ytildescat.conj()*ztildescat).real.sum()
            logdet_term = self.d_out/torch.tensor(logdet.shape).prod()*logdet.sum()
            mll = norm_term+logdet_term+mll_const
            if mll.item()<stop_crit_best_mll:
                stop_crit_best_mll = mll.item()
            if (stop_crit_save_mll-mll.item())>logtol:
                stop_crit_iterations_without_improvement_mll = 0
                stop_crit_save_mll = stop_crit_best_mll
            else:
                stop_crit_iterations_without_improvement_mll += 1
            break_condition = i==iterations or stop_crit_iterations_without_improvement_mll==stop_crit_wait_iterations
            if store_mll_hist: mll_hist[i] = mll.item()
            if store_scale_hist: scale_hist[i] = self.scale.detach().to(scale_hist.device)
            if store_lengthscales_hist: lengthscales_hist[i] = self.lengthscales.detach().to(lengthscales_hist.device)
            if store_noise_hist: noise_hist[i] = self.noise.detach().to(noise_hist.device)
            if store_task_kernel_hist: task_kernel_hist[i] = self.gram_matrix_tasks.detach().to(task_kernel_hist.device)
            if verbose and (i%verbose==0 or break_condition):
                _s = "%16.2e | %-10.2e | %-10.2e | %-10.2e"%(i,mll.item(),norm_term.item(),logdet_term.item())
                print(" "*verbose_indent+_s)
            if break_condition: break
            mll.backward()
            optimizer.step()
            optimizer.zero_grad()
        del os.environ["FASTGP_FORCE_RECOMPILE"]
        data = {}
        if store_mll_hist: data["mll_hist"] = mll_hist
        if store_scale_hist: data["scale_hist"] = scale_hist
        if store_lengthscales_hist: data["lengthscales_hist"] = lengthscales_hist
        if store_noise_hist: data["noise_hist"] = noise_hist
        if store_task_kernel_hist: data["task_kernel_hist"] = task_kernel_hist
        return data
    def post_cubature_mean(self, task:Union[int,torch.Tensor]=None, eval:bool=True):
        """
        Posterior cubature mean. 

        Args:
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`
            task (Union[int,torch.Tensor[T]]): task indices

        Returns:
            pcmean (torch.Tensor[...,T]): posterior cubature mean
        """
        kmat_tasks = self.gram_matrix_tasks
        coeffs = self.coeffs
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int)
        if isinstance(task,list): task = torch.tensor(task,dtype=int)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        coeffs_split = coeffs.split(self.n.tolist(),-1)
        coeffs_split_scaled = [(self.scale*coeffs_split[l])[...,None,:]*kmat_tasks[...,task,l,None] for l in range(self.num_tasks)]
        pcmean = torch.cat(coeffs_split_scaled,-1).sum(-1)
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pcmean[...,0] if inttask else pcmean
    def post_cubature_var(self, task:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True):
        """
        Posterior cubature variance. 

        Args:
            task (Union[int,torch.Tensor[T]]): task indices
            n (Union[int,torch.Tensor[num_tasks]]): number of points at which to evaluate the posterior cubature variance.
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`

        Returns:
            pcvar (torch.Tensor[T]): posterior cubature variance
        """
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and (n&(n-1)==0).all() and (n>=self.n).all(), "require n are all power of two greater than or equal to self.n"
        kmat_tasks = self.gram_matrix_tasks
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        inv = inv_log_det_cache()[0]
        to = inv_log_det_cache.task_order
        nord = n[to]
        mvec = torch.hstack([torch.zeros(1,device=self.device),(nord/nord[-1]).cumsum(0)]).to(int)[:-1]
        nsqrts = torch.sqrt(nord[:,None]*nord[None,:])
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int)
        if isinstance(task,list): task = torch.tensor(task,dtype=int)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        inv_cut = inv[...,mvec,:,:][...,:,mvec,:][...,0]
        kmat_tasks_left = kmat_tasks[...,task,:][...,:,to].to(self._FTOUTDTYPE)
        kmat_tasks_right = kmat_tasks[...,to,:][...,:,task].to(self._FTOUTDTYPE)
        term = torch.einsum("...ij,...jk,...ki->...i",kmat_tasks_left,nsqrts*inv_cut,kmat_tasks_right).real
        pcvar = self.scale*kmat_tasks[...,task,task]-self.scale**2*term
        pcvar[pcvar<0] = 0.
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pcvar[...,0] if inttask else pcvar
    def post_cubature_cov(self, task0:Union[int,torch.Tensor]=None, task1:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True):
        """
        Posterior cubature covariance. 

        Args:
            task0 (Union[int,torch.Tensor[T1]]): task indices
            task1 (Union[int,torch.Tensor[T2]]): task indices
            n (Union[int,torch.Tensor[num_tasks]]): number of points at which to evaluate the posterior cubature covariance.
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`

        Returns:
            pcvar (torch.Tensor[T1,T2]): posterior cubature covariance
        """
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and (n&(n-1)==0).all() and (n>=self.n).all(), "require n are all power of two greater than or equal to self.n"
        kmat_tasks = self.gram_matrix_tasks
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        inv = inv_log_det_cache()[0]
        to = inv_log_det_cache.task_order
        nord = n[to]
        mvec = torch.hstack([torch.zeros(1,device=self.device),(nord/nord[-1]).cumsum(0)]).to(int)[:-1]
        nsqrts = torch.sqrt(nord[:,None]*nord[None,:])
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task0 is None: task0 = self.default_task
        inttask0 = isinstance(task0,int)
        if inttask0: task0 = torch.tensor([task0],dtype=int)
        if isinstance(task0,list): task0 = torch.tensor(task0,dtype=int)
        assert task0.ndim==1 and (task0>=0).all() and (task0<self.num_tasks).all()
        if task1 is None: task1 = self.default_task
        inttask1 = isinstance(task1,int)
        if inttask1: task1 = torch.tensor([task1],dtype=int)
        if isinstance(task1,list): task1 = torch.tensor(task1,dtype=int)
        assert task1.ndim==1 and (task1>=0).all() and (task1<self.num_tasks).all()
        equal = torch.equal(task0,task1)
        inv_cut = inv[...,mvec,:,:][...,:,mvec,:][...,0]
        kmat_tasks_left = kmat_tasks[...,task0,:][...,:,to].to(self._FTOUTDTYPE)
        kmat_tasks_right = kmat_tasks[...,to,:][...,:,task1].to(self._FTOUTDTYPE)
        term = torch.einsum("...ij,...jk,...kl->...il",kmat_tasks_left,nsqrts*inv_cut,kmat_tasks_right).real
        pccov = self.scale[...,None]*kmat_tasks[...,task0,:][...,:,task1]-self.scale[...,None]**2*term
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
        return self.lam_caches[task0,task1][m]
    def get_k1parts(self, task0, task1, n=None):
        assert 0<=task0<self.num_tasks
        assert 0<=task1<self.num_tasks
        if n is None: n = self.n[task0]
        assert n>=0
        return self.k1parts_seq[task0,task1][:n]
    def get_ytilde(self, task):
        assert 0<=task<self.num_tasks
        return self.ytilde_cache[task]()
    def get_inv_log_det_cache(self, n=None):
        if n is None: n = self.n
        assert isinstance(n,torch.Tensor) and n.shape==(self.num_tasks,) and (n>=self.n).all()
        ntup = tuple(n.tolist())
        if ntup not in self.inv_log_det_cache_dict.keys():
            self.inv_log_det_cache_dict[ntup] = _FastInverseLogDetCache(self,n)
        return self.inv_log_det_cache_dict[ntup]
    def get_inv_log_det(self, n=None):
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        return inv_log_det_cache()
    def _kernel_parts(self, x, z, beta0, beta1):
        assert x.size(-1)==self.d and z.size(-1)==self.d and beta0.ndim==2 and beta0.size(1)==self.d and beta1.ndim==2 and beta1.size(1)==self.d
        delta = self._ominus(x,z)
        parts = torch.empty(list(delta.shape)[:-1]+[len(beta0),len(beta1)]+[delta.size(-1)])
        for _t0 in range(len(beta0)):
            for _t1 in range(len(beta1)):
                parts[...,_t0,_t1,:] = self._kernel_parts_from_delta(delta,beta0[_t0],beta1[_t1])
        return parts
    def _kernel_from_parts(self, parts, beta0, beta1, c0, c1):
        assert c0.ndim==1 and c1.ndim==1
        assert beta0.shape==(len(c0),self.d) and beta1.shape==(len(c1),self.d)
        assert parts.shape[-3:]==(len(c0),len(c1),self.d)
        ndim = parts.ndim
        scale = self.scale.reshape(self.scale.shape+torch.Size([1]*(ndim-2))) 
        lengthscales = self.lengthscales.reshape(self.lengthscales.shape[:-1]+torch.Size([1]*(ndim-1)+[self.lengthscales.size(-1)]))
        ind = ((beta0[:,None,:]+beta1[None,:,:])==0).to(torch.int64)
        terms = scale*(ind+lengthscales*parts).prod(-1)
        vals = ((terms*c1).sum(-1)*c0).sum(-1)
        return vals
    def _kernel(self, x:torch.Tensor, z:torch.Tensor, beta0:torch.Tensor, beta1: torch.Tensor, c0:torch.Tensor, c1:torch.Tensor):
        assert c0.ndim==1 and c1.ndim==1
        assert beta0.shape==(len(c0),self.d) and beta1.shape==(len(c1),self.d)
        assert x.size(-1)==self.d and z.size(-1)==self.d
        return self._kernel_from_parts(self._kernel_parts(x,z,beta0,beta1),beta0,beta1,c0,c1)
    def ft(self, x):
        xmean = x.mean(-1)
        y = self.ft_unstable(x-xmean[...,None])
        y[...,0] += xmean*np.sqrt(x.size(-1))
        return y
    def ift(self, x):
        xmean = x.mean(-1)
        y = self.ift_unstable(x-xmean[...,None])
        y[...,0] += xmean*np.sqrt(x.size(-1))
        return y
