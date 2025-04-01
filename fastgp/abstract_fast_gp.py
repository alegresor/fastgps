from .util import (
    _XXbSeq,
    _CoeffsCache,
    _InverseLogDetCache,
    _K1PartsSeq,
    _LamCaches,
    _TaskCovCache,
    _YtildeCache)
import torch
import numpy as np 
import scipy.stats 
import os
from typing import Union,List

class AbstractFastGP(torch.nn.Module):
    def __init__(self,
            seqs,
            num_tasks,
            default_task,
            solo_task,
            alpha,
            scale,
            lengthscales,
            noise,
            factor_task_kernel,
            rank_factor_task_kernel,
            noise_task_kernel,
            device,
            tfs_scale,
            tfs_lengthscales,
            tfs_noise,
            tfs_factor_task_kernel,
            tfs_noise_task_kernel,
            requires_grad_scale, 
            requires_grad_lengthscales, 
            requires_grad_noise,
            requires_grad_factor_task_kernel,
            requires_grad_noise_task_kernel,
            shape_batch,
            shape_scale, 
            shape_lengthscales,
            shape_noise,
            shape_factor_task_kernel, 
            shape_noise_task_kernel,
            ft,
            ift,
        ):
        super().__init__()
        assert torch.get_default_dtype()==torch.float64, "fast transforms do not work without torch.float64 precision" 
        assert isinstance(num_tasks,int) and num_tasks>0
        self.num_tasks = num_tasks
        self.default_task = default_task
        self.solo_task = solo_task
        self.device = torch.device(device)
        assert isinstance(seqs,np.ndarray) and seqs.shape==(self.num_tasks,)
        self.d = seqs[0].d
        assert all(seqs[i].d==self.d for i in range(self.num_tasks))
        self.seqs = seqs
        self.n = torch.zeros(self.num_tasks,dtype=int,device=self.device)
        self.m = -1*torch.ones(self.num_tasks,dtype=int,device=self.device)
        # alpha
        assert (np.isscalar(alpha) and alpha%1==0) or (isinstance(alpha,torch.Tensor) and alpha.shape==(self.d,)), "alpha should be an int or a torch.Tensor of length d"
        if np.isscalar(alpha):
            alpha = int(alpha)*torch.ones(self.d,dtype=int,device=self.device)
        self.alpha = alpha
        # shape_batch 
        if isinstance(shape_batch,(list,tuple)): shape_batch = torch.Size(shape_batch)
        assert isinstance(shape_batch,torch.Size)
        self.shape_batch = shape_batch
        self.ndim_batch = len(self.shape_batch)
        # scale
        assert np.isscalar(scale) or isinstance(scale,torch.Tensor), "scale must be a scalar or torch.Tensor"
        if isinstance(scale,torch.Tensor): shape_scale = scale.shape
        if isinstance(shape_scale,(list,tuple)): shape_scale = torch.Size(shape_scale)
        assert isinstance(shape_scale,torch.Size) and shape_scale[-1]==1
        if len(shape_scale)>1: assert shape_scale[:-1]==shape_batch[-(len(shape_scale)-1):]
        if np.isscalar(scale): scale = scale*torch.ones(shape_scale,device=self.device)
        assert (scale>0).all(), "scale must be positive"
        assert len(tfs_scale)==2 and callable(tfs_scale[0]) and callable(tfs_scale[1]), "tfs_scale should be a tuple of two callables, the transform and inverse transform"
        self.tf_scale = tfs_scale[1]
        self.raw_scale = torch.nn.Parameter(tfs_scale[0](scale),requires_grad=requires_grad_scale)
        # lengthscales
        assert np.isscalar(lengthscales) or isinstance(lengthscales,torch.Tensor), "lengthscales must be a scalar or torch.Tensor"
        if isinstance(lengthscales,torch.Tensor): shape_lengthscales = lengthscales.shape 
        if shape_lengthscales is None: shape_lengthscales = torch.Size([self.d])
        if isinstance(shape_lengthscales,(list,tuple)): shape_lengthscales = torch.Size(shape_lengthscales)
        assert isinstance(shape_lengthscales,torch.Size) and (shape_lengthscales[-1]==self.d or shape_lengthscales[-1]==1)
        if len(shape_lengthscales)>1: assert shape_lengthscales[:-1]==shape_batch[-(len(shape_lengthscales)-1):]
        if np.isscalar(lengthscales): lengthscales = lengthscales*torch.ones(shape_lengthscales,device=self.device)
        assert (lengthscales>0).all(), "lengthscales must be positive"
        assert len(tfs_lengthscales)==2 and callable(tfs_lengthscales[0]) and callable(tfs_lengthscales[1]), "tfs_lengthscales should be a tuple of two callables, the transform and inverse transform"
        self.tf_lengthscales = tfs_lengthscales[1]
        self.raw_lengthscales = torch.nn.Parameter(tfs_lengthscales[0](lengthscales),requires_grad=requires_grad_lengthscales)
        # noise
        assert np.isscalar(noise) or isinstance(noise,torch.Tensor), "noise must be a scalar or torch.Tensor"
        if isinstance(noise,torch.Tensor): shape_noise = noise.shape
        if isinstance(shape_noise,(list,tuple)): shape_noise = torch.Size(shape_noise)
        assert isinstance(shape_noise,torch.Size) and shape_noise[-1]==1
        if len(shape_noise)>1: assert shape_noise[:-1]==shape_batch[-(len(shape_noise)-1):]
        if np.isscalar(noise): noise = noise*torch.ones(shape_noise,device=self.device)
        assert (noise>0).all(), "noise must be positive"
        assert len(tfs_noise)==2 and callable(tfs_noise[0]) and callable(tfs_noise[1]), "tfs_scale should be a tuple of two callables, the transform and inverse transform"
        self.tf_noise = tfs_noise[1]
        self.raw_noise = torch.nn.Parameter(tfs_noise[0](noise),requires_grad=requires_grad_noise)
        # factor_task_kernel
        assert np.isscalar(factor_task_kernel) or isinstance(factor_task_kernel,torch.Tensor), "factor_task_kernel must be a scalar or torch.Tensor"
        if isinstance(factor_task_kernel,torch.Tensor): shape_factor_task_kernel = factor_task_kernel.shape
        if shape_factor_task_kernel is None:
            if rank_factor_task_kernel is None: rank_factor_task_kernel = 0 if self.num_tasks==1 else 1 
            assert isinstance(rank_factor_task_kernel,int) and 0<=rank_factor_task_kernel<=self.num_tasks
            shape_factor_task_kernel = torch.Size([self.num_tasks,rank_factor_task_kernel])
        if isinstance(shape_factor_task_kernel,(list,tuple)): shape_factor_task_kernel = torch.Size(shape_factor_task_kernel)
        assert isinstance(shape_factor_task_kernel,torch.Size) and 0<=shape_factor_task_kernel[-1]<=self.num_tasks and shape_factor_task_kernel[-2]==self.num_tasks
        if len(shape_factor_task_kernel)>2: assert shape_factor_task_kernel[:-2]==shape_batch[-(len(shape_factor_task_kernel)-2):]
        if np.isscalar(factor_task_kernel): factor_task_kernel = factor_task_kernel*torch.ones(shape_factor_task_kernel,device=self.device)
        assert len(tfs_factor_task_kernel)==2 and callable(tfs_factor_task_kernel[0]) and callable(tfs_factor_task_kernel[1]), "tfs_factor_task_kernel should be a tuple of two callables, the transform and inverse transform"
        self.tf_factor_task_kernel = tfs_factor_task_kernel[1]
        if requires_grad_factor_task_kernel is None: requires_grad_factor_task_kernel = self.num_tasks>1
        self.raw_factor_task_kernel = torch.nn.Parameter(tfs_factor_task_kernel[0](factor_task_kernel),requires_grad=requires_grad_factor_task_kernel)
        # noise_task_kernel
        assert np.isscalar(noise_task_kernel) or isinstance(noise_task_kernel,torch.Tensor), "noise_task_kernel must be a scalar or torch.Tensor"
        if isinstance(noise_task_kernel,torch.Tensor): shape_noise_task_kernel = noise_task_kernel.shape 
        if shape_noise_task_kernel is None: shape_noise_task_kernel = torch.Size([self.num_tasks])
        if isinstance(shape_noise_task_kernel,(list,tuple)): shape_noise_task_kernel = torch.Size(shape_noise_task_kernel)
        assert isinstance(shape_noise_task_kernel,torch.Size) and (shape_noise_task_kernel[-1]==self.num_tasks or shape_noise_task_kernel[-1]==1)
        if len(shape_noise_task_kernel)>1: assert shape_noise_task_kernel[:-1]==shape_batch[-(len(shape_noise_task_kernel)-1):]
        if np.isscalar(noise_task_kernel): noise_task_kernel = noise_task_kernel*torch.ones(shape_noise_task_kernel,device=self.device)
        assert (noise_task_kernel>0).all(), "noise_task_kernel must be positive"
        assert len(tfs_noise_task_kernel)==2 and callable(tfs_noise_task_kernel[0]) and callable(tfs_noise_task_kernel[1]), "tfs_noise_task_kernel should be a tuple of two callables, the transform and inverse transform"
        self.tf_noise_task_kernel = tfs_noise_task_kernel[1]
        if requires_grad_noise_task_kernel is None: requires_grad_noise_task_kernel = self.num_tasks>1
        self.raw_noise_task_kernel = torch.nn.Parameter(tfs_noise_task_kernel[0](noise_task_kernel),requires_grad=requires_grad_noise_task_kernel)
        # fast transforms 
        self.ft = ft
        self.ift = ift
        # storaget and dynamic caches
        self._y = [torch.empty(0,device=self.device) for l in range(self.num_tasks)]
        self.xxb_seqs = np.array([_XXbSeq(self,self.seqs[i]) for i in range(self.num_tasks)],dtype=object)
        self.k1parts_seq = np.array([[_K1PartsSeq(self,self.xxb_seqs[l0],self.xxb_seqs[l1]) for l1 in range(self.num_tasks)] for l0 in range(self.num_tasks)],dtype=object)
        self.lam_caches = np.array([[_LamCaches(self,l0,l1) for l1 in range(self.num_tasks)] for l0 in range(self.num_tasks)],dtype=object)
        self.ytilde_cache = np.array([_YtildeCache(self,i) for i in range(self.num_tasks)],dtype=object)
        self.task_cov_cache = _TaskCovCache(self)
        self.coeffs_cache = _CoeffsCache(self)
        self.inv_log_det_cache_dict = {}
        # MLL setup
        self.d_out = int(torch.tensor(self.shape_batch).prod())
    def get_x_next(self, n:Union[int,torch.Tensor], task:Union[int,torch.Tensor]=None):
        """
        Get the next sampling locations. 

        Args:
            n (Union[int,torch.Tensor]): maximum sample index per task
            task (Union[int,torch.Tensor]): task index
        
        Returns:
            x_next (Union[torch.Tensor,List]): next samples in the sequence
        """
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device) 
        if isinstance(n,list): n = torch.tensor(n,dtype=int)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int)
        if isinstance(task,list): task = torch.tensor(task,dtype=int)
        assert isinstance(n,torch.Tensor) and isinstance(task,torch.Tensor) and n.ndim==task.ndim==1 and len(n)==len(task)
        assert (n>=self.n[task]).all() and torch.logical_or(n==0,n&(n-1)==0).all(), "maximum sequence index must be a power of 2 greater than the current number of samples"
        x_next = [self.xxb_seqs[l][self.n[l]:n[i]][0] for i,l in enumerate(task)]
        return x_next[0] if inttask else x_next
    def add_y_next(self, y_next:Union[torch.Tensor,List], task:Union[int,torch.Tensor]=None):
        """
        Add samples to the GP. 

        Args:
            y_next (Union[torch.Tensor,List]): new function evaluations at next sampling locations
            task (Union[int,torch.Tensor]): task index
        """
        if isinstance(y_next,torch.Tensor): y_next = [y_next]
        if task is None: task = self.default_task
        if isinstance(task,int): task = torch.tensor([task],dtype=int)
        if isinstance(task,list): task = torch.tensor(task,dtype=int)
        assert isinstance(y_next,list) and isinstance(task,torch.Tensor) and task.ndim==1 and len(y_next)==len(task)
        assert all(y_next[i].shape[:-1]==self.shape_batch for i in range(len(y_next)))
        for i,l in enumerate(task):
            self._y[l] = torch.cat([self._y[l],y_next[i]],-1)
        self.n = torch.tensor([self._y[i].size(-1) for i in range(self.num_tasks)],dtype=int,device=self.device)
        assert torch.logical_or(self.n==0,(self.n&(self.n-1)==0)).all(), "total samples must be power of 2"
        self.m = torch.where(self.n==0,-1,torch.log2(self.n)).to(int)
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
    def post_mean(self, x:torch.Tensor, task:Union[int,torch.Tensor]=None, eval:bool=True):
        """
        Posterior mean. 

        Args:
            x (torch.Tensor[N,d]): sampling locations
            task (Union[int,torch.Tensor[T]]): task index
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`
        
        Returns:
            pmean (torch.Tensor[...,T,N]): posterior mean
        """
        coeffs = self.coeffs
        kmat_tasks = self.gram_matrix_tasks
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int)
        if isinstance(task,list): task = torch.tensor(task,dtype=int)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        kmat = torch.cat([self._kernel(x[:,None,:],self.get_xb(l)[None,:,:])[...,None,:,:]*kmat_tasks[...,task,l,None,None] for l in range(self.num_tasks)],dim=-1)
        pmean = (kmat*coeffs[...,None,None,:]).sum(-1)
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pmean[...,0,:] if inttask else pmean
    def post_var(self, x:torch.Tensor, task:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True):
        """
        Posterior variance.

        Args:
            x (torch.Tensor[N,d]): sampling locations
            task (Union[int,torch.Tensor[T]]): task indices
            n (Union[int,torch.Tensor[num_tasks]]): number of points at which to evaluate the posterior cubature variance.
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`

        Returns:
            pvar (torch.Tensor[T,N]): posterior variance
        """
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and (n&(n-1)==0).all() and (n>=self.n).all(), "require n are all power of two greater than or equal to self.n"
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        kmat_tasks = self.gram_matrix_tasks
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int)
        if isinstance(task,list): task = torch.tensor(task,dtype=int)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        kmat_new = self._kernel(x,x)[...,None,:]*kmat_tasks[...,task,task,None]
        kmat = torch.cat([self._kernel(x[:,None,:],self.get_xb(l,n=n[l])[None,:,:])[...,None,:,:]*kmat_tasks[...,task,l,None,None] for l in range(self.num_tasks)],dim=-1)
        kmat_perm = torch.permute(kmat,[-3,-2]+[i for i in range(kmat.ndim-3)]+[-1])
        t_perm = self.get_inv_log_det_cache(n).gram_matrix_solve(kmat_perm)
        t = torch.permute(t_perm,[2+i for i in range(t_perm.ndim-3)]+[0,1,-1])
        diag = kmat_new-(t*kmat).sum(-1)
        diag[diag<0] = 0 
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return diag[...,0,:] if inttask else diag
    def post_cov(self, x0:torch.Tensor, x1:torch.Tensor, task0:Union[int,torch.Tensor]=None, task1:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True):
        """
        Posterior covariance. 

        Args:
            x0 (torch.Tensor[N,d]): left sampling locations
            x1 (torch.Tensor[M,d]): right sampling locations
            task0 (Union[int,torch.Tensor[T1]]): left task index
            task1 (Union[int,torch.Tensor[T2]]): right task index
            n (Union[int,torch.Tensor[num_tasks]]): number of points at which to evaluate the posterior cubature variance.
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`
        
        Returns:
            pcov (torch.Tensor[T1,T2,N,M]): posterior covariance matrix
        """
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and (n&(n-1)==0).all() and (n>=self.n).all(), "require n are all power of two greater than or equal to self.n"
        assert x0.ndim==2 and x0.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        assert x1.ndim==2 and x1.size(1)==self.d, "z must a torch.Tensor with shape (-1,d)"
        kmat_tasks = self.gram_matrix_tasks
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
        equal = torch.equal(x0,x1) and torch.equal(task0,task1)
        kmat_new = self._kernel(x0[:,None,:],x1[None,:,:])[...,None,None,:,:]*kmat_tasks[...,task0,:][...,task1][...,None,None]
        kmat1 = torch.cat([self._kernel(x0[:,None,:],self.get_xb(l,n=n[l])[None,:,:])[...,None,:,:]*kmat_tasks[...,task1,l,None,None] for l in range(self.num_tasks)],dim=-1)
        kmat2 = kmat1 if equal else torch.cat([self._kernel(x1[:,None,:],self.get_xb(l,n=n[l])[None,:,:])[...,None,:,:]*kmat_tasks[...,task1,l,None,None] for l in range(self.num_tasks)],dim=-1)
        kmat2_perm = torch.permute(kmat2,[-3,-2]+[i for i in range(kmat2.ndim-3)]+[-1])
        t_perm = self.get_inv_log_det_cache(n).gram_matrix_solve(kmat2_perm)
        t = torch.permute(t_perm,[2+i for i in range(t_perm.ndim-3)]+[0,1,-1])
        kmat = kmat_new-(kmat1[...,:,None,:,None,:]*t[...,None,:,None,:,:]).sum(-1)
        if equal:
            tmesh,nmesh = torch.meshgrid(torch.arange(kmat.size(0),device=self.device),torch.arange(x0.size(0),device=x0.device),indexing="ij")            
            tidx,nidx = tmesh.ravel(),nmesh.ravel()
            diag = kmat[...,tidx,tidx,nidx,nidx]
            diag[diag<0] = 0 
            kmat[...,tidx,tidx,nidx,nidx] = diag 
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        if inttask0 and inttask1:
            return kmat[...,0,0,:,:]
        elif inttask0 and not inttask1:
            return kmat[...,0,:,:,:]
        elif not inttask0 and inttask1:
            return kmat[...,:,0,:,:]
        else: # not inttask0 and not inttask1
            return kmat
    def post_ci(self, x, task:Union[int,torch.Tensor]=None, confidence:float=0.99, eval:bool=True):
        """
        Posterior credible interval.

        Args:
            x (torch.Tensor[N,d]): sampling locations
            task (Union[int,torch.Tensor[T]]): task indices
            confidence (float): confidence level in $(0,1)$ for the credible interval
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`

        Returns:
            pmean (torch.Tensor[...,T,N]): posterior mean
            pvar (torch.Tensor[T,N]): posterior variance 
            quantile (np.float64):
                ```python
                scipy.stats.norm.ppf(1-(1-confidence)/2)
                ```
            ci_low (torch.Tensor[...,T,N]): credible interval lower bound
            ci_high (torch.Tensor[...,T,N]): credible interval upper bound
        """
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pmean = self.post_mean(x,task=task,eval=eval) 
        pvar = self.post_var(x,task=task,eval=eval)
        pstd = torch.sqrt(pvar)
        ci_low = pmean-q*pstd 
        ci_high = pmean+q*pstd
        return pmean,pvar,q,ci_low,ci_high
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
    def post_cubature_ci(self, task:Union[int,torch.Tensor]=None, confidence:float=0.99, eval:bool=True):
        """
        Posterior cubature credible.

        Args:
            task (Union[int,torch.Tensor[T]]): task indices
            confidence (float): confidence level in $(0,1)$ for the credible interval
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`
        
        Returns:
            pcmean (torch.Tensor[...,T]): posterior cubature mean
            pcvar (torch.Tensor[T]): posterior cubature variance
            quantile (np.float64):
                ```python
                scipy.stats.norm.ppf(1-(1-confidence)/2)
                ```
            cci_low (torch.Tensor[...,T]): scalar credible interval lower bound
            cci_high (torch.Tensor[...,T]): scalar credible interval upper bound
        """
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pmean = self.post_cubature_mean(task=task,eval=eval) 
        pvar = self.post_cubature_var(task=task,eval=eval)
        pstd = torch.sqrt(pvar)
        ci_low = pmean-q*pstd 
        ci_high = pmean+q*pstd 
        return pmean,pvar,q,ci_low,ci_high
    @property
    def scale(self):
        """
        Kernel scale parameter.
        """
        return self.tf_scale(self.raw_scale)
    @property
    def lengthscales(self):
        """
        Kernel lengthscale parameter.
        """
        return self.tf_lengthscales(self.raw_lengthscales)
    @property
    def noise(self):
        """
        Noise parameter.
        """
        return self.tf_noise(self.raw_noise)
    @property
    def factor_task_kernel(self):
        """
        Factor for the task kernel parameter.
        """
        return self.tf_factor_task_kernel(self.raw_factor_task_kernel)
    @property
    def noise_task_kernel(self):
        """
        Noise for the task kernel parameter.
        """
        return self.tf_noise_task_kernel(self.raw_noise_task_kernel)
    @property 
    def gram_matrix_tasks(self):
        """
        Gram matrix for the task kernel.
        """
        return self.task_cov_cache()
    @property 
    def coeffs(self):
        r"""
        Coefficients $\mathsf{K}^{-1} \boldsymbol{y}$.
        """
        return self.coeffs_cache()
    @property
    def x(self):
        """
        Current sampling locations. 
        A `torch.Tensor` for single task problems.
        A `list` for multitask problems.
        """
        xs = [self.get_x(l) for l in range(self.num_tasks)]
        return xs[0] if self.solo_task else xs
    @property
    def y(self):
        """
        Current sampling values. 
        A `torch.Tensor` for single task problems.
        A `list` for multitask problems.
        """
        return self._y[0] if self.solo_task else self._y 
    def get_x(self, task, n=None):
        assert 0<=task<self.num_tasks
        if n is None: n = self.n[task]
        assert n>=0
        x,xb = self.xxb_seqs[task][:n]
        return x
    def get_xb(self, task, n=None):
        assert 0<=task<self.num_tasks
        if n is None: n = self.n[task]
        assert n>=0
        x,xb = self.xxb_seqs[task][:n]
        return xb
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
            self.inv_log_det_cache_dict[ntup] = _InverseLogDetCache(self,n)
        for key in list(self.inv_log_det_cache_dict.keys()):
            if (torch.tensor(key)<self.n.cpu()).any():
                del self.inv_log_det_cache_dict[key]
        return self.inv_log_det_cache_dict[ntup]
    def get_inv_log_det(self, n=None):
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        return inv_log_det_cache()
    def _kernel_parts(self, x, z):
        return self._kernel_parts_from_delta(self._ominus(x,z))
    def _kernel_from_parts(self, parts):
        ndim = parts.ndim
        scale = self.scale.reshape(self.scale.shape+torch.Size([1]*(ndim-2))) 
        lengthscales = self.lengthscales.reshape(self.lengthscales.shape[:-1]+torch.Size([1]*(ndim-1)+[self.lengthscales.size(-1)]))
        return scale*(1+lengthscales*parts).prod(-1)
    def _kernel_from_delta(self, delta):
        return self._kernel_from_parts(self._kernel_parts_from_delta(delta))
    def _kernel(self, x:torch.Tensor, z:torch.Tensor):
        return self._kernel_from_parts(self._kernel_parts(x,z))
