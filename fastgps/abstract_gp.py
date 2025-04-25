from .util import (
    _XXbSeq,
    _TaskCovCache,
    _CoeffsCache,
)
import torch
import numpy as np 
import scipy.stats 
import os
from typing import Union,List

class AbstractGP(torch.nn.Module):
    def __init__(self,
            seqs,
            num_tasks,
            default_task,
            solo_task,
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
            derivatives,
            derivatives_coeffs,
            adaptive_nugget,
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
        # derivatives
        if derivatives is not None or derivatives_coeffs is not None:
            rank_factor_task_kernel = 1
            tfs_noise_task_kernel = (lambda x: x, lambda x: x)
            noise_task_kernel = 0.
        if derivatives is None: derivatives = [torch.zeros((1,self.d),dtype=torch.int64,device=self.device) for i in range(self.num_tasks)]
        if isinstance(derivatives,torch.Tensor): derivatives = [derivatives]
        assert isinstance(derivatives,list) and len(derivatives)==self.num_tasks
        derivatives = [deriv[None,:] if deriv.ndim==1 else deriv for deriv in derivatives]
        assert all((derivatives[i].ndim==2 and derivatives[i].size(1)==self.d) for i in range(self.num_tasks))
        self.derivatives = derivatives
        if derivatives_coeffs is None: derivatives_coeffs = [torch.ones(len(self.derivatives[i]),device=self.device) for i in range(self.num_tasks)]
        assert isinstance(derivatives_coeffs,list) and len(derivatives_coeffs)==self.num_tasks
        assert all((derivatives_coeffs[i].ndim==1 and len(derivatives_coeffs[i]))==len(self.derivatives[i]) for i in range(self.num_tasks))
        self.derivatives_coeffs = derivatives_coeffs
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
        assert (noise_task_kernel>=0).all(), "noise_task_kernel must be positive"
        assert len(tfs_noise_task_kernel)==2 and callable(tfs_noise_task_kernel[0]) and callable(tfs_noise_task_kernel[1]), "tfs_noise_task_kernel should be a tuple of two callables, the transform and inverse transform"
        self.tf_noise_task_kernel = tfs_noise_task_kernel[1]
        if requires_grad_noise_task_kernel is None: requires_grad_noise_task_kernel = self.num_tasks>1
        self.raw_noise_task_kernel = torch.nn.Parameter(tfs_noise_task_kernel[0](noise_task_kernel),requires_grad=requires_grad_noise_task_kernel)
        # storage and dynamic caches
        self._y = [torch.empty(0,device=self.device) for l in range(self.num_tasks)]
        self.xxb_seqs = np.array([_XXbSeq(self,self.seqs[i]) for i in range(self.num_tasks)],dtype=object)
        self.coeffs_cache = _CoeffsCache(self)
        self.task_cov_cache = _TaskCovCache(self)
        self.inv_log_det_cache_dict = {}
        # derivative multitask setting checks 
        if any((self.derivatives[i]>0).any() or (self.derivatives_coeffs[i]!=1).any() for i in range(self.num_tasks)):
            self.raw_noise_task_kernel.requires_grad_(False)
            self.raw_factor_task_kernel.requires_grad_(False)
            assert (self.gram_matrix_tasks==1).all()
        self.adaptive_nugget = adaptive_nugget
    def fit(self,
        loss_metric:str = "MLL",
        iterations:int = 5000,
        lr:float = None,
        optimizer:torch.optim.Optimizer = None,
        stop_crit_improvement_threshold:float = 5e-2,
        stop_crit_wait_iterations:int = 10,
        store_hists:bool = False,
        store_loss_hist:bool = False, 
        store_scale_hist:bool = False, 
        store_lengthscales_hist:bool = False,
        store_noise_hist:bool = False,
        store_task_kernel_hist:bool = False,
        verbose:int = 5,
        verbose_indent:int = 4,
        masks:torch.Tensor = None,
        cv_weights:torch.Tensor = 1,
        ):
        """
        Args:
            loss_metric (str): either "MLL" (Marginal Log Likelihood) or "CV" (Cross Validation) or "GCV" (Generalized CV)
            iterations (int): number of optimization iterations
            lr (float): learning rate for default optimizer
            optimizer (torch.optim.Optimizer): optimizer defaulted to `torch.optim.Rprop(self.parameters(),lr=lr)`
            stop_crit_improvement_threshold (float): stop fitting when the maximum number of iterations is reached or the best loss is note reduced by `stop_crit_improvement_threshold` for `stop_crit_wait_iterations` iterations 
            stop_crit_wait_iterations (int): number of iterations to wait for improved loss before early stopping, see the argument description for `stop_crit_improvement_threshold`
            store_hists (bool): if True then store all hists, otherwise specify individually with the following arguments 
            store_loss_hist (bool): if `True`, store and return iteration data for loss
            store_scale_hist (bool): if `True`, store and return iteration data for the kernel scale parameter
            store_lengthscales_hist (bool): if `True`, store and return iteration data for the kernel lengthscale parameters
            store_noise_hist (bool): if `True`, store and return iteration data for noise
            store_task_kernel_hist (bool): if `True`, store and return iteration data for the task kernel
            verbose (int): log every `verbose` iterations, set to `0` for silent mode
            verbose_indent (int): size of the indent to be applied when logging, helpful for logging multiple models
            masks (torch.Tensor): only optimize outputs corresponding to `y[...,*masks]`
            cv_weights (Union[str,torch.Tensor]): weights for cross validation
            
        Returns:
            data (dict): iteration data which, dependeing on storage arguments, may include keys in 
                ```python
                ["loss_hist","scale_hist","lengthscales_hist","noise_hist","task_kernel_hist"]
                ```
        """
        assert isinstance(loss_metric,str) and loss_metric.upper() in ["MLL","GCV","CV"] 
        assert (self.n>0).any(), "cannot fit without data"
        assert isinstance(iterations,int) and iterations>=0
        if optimizer is None:
            optimizer = self.get_default_optimizer(lr)
        assert isinstance(optimizer,torch.optim.Optimizer)
        assert isinstance(store_hists,bool), "require bool store_mll_hist" 
        assert isinstance(store_loss_hist,bool), "require bool store_loss_hist" 
        assert isinstance(store_scale_hist,bool), "require bool store_scale_hist" 
        assert isinstance(store_lengthscales_hist,bool), "require bool store_lengthscales_hist" 
        assert isinstance(store_noise_hist,bool), "require bool store_noise_hist"
        assert isinstance(store_task_kernel_hist,bool), "require bool store_task_kernel_hist"
        assert (isinstance(verbose,int) or isinstance(verbose,bool)) and verbose>=0, "require verbose is a non-negative int"
        assert isinstance(verbose_indent,int) and verbose_indent>=0, "require verbose_indent is a non-negative int"
        assert np.isscalar(stop_crit_improvement_threshold) and 0<stop_crit_improvement_threshold, "require stop_crit_improvement_threshold is a positive float"
        assert isinstance(stop_crit_wait_iterations,int) and stop_crit_wait_iterations>0
        assert masks is None or (isinstance(masks,torch.Tensor))
        loss_metric = loss_metric.upper()
        logtol = np.log(1+stop_crit_improvement_threshold)
        store_loss_hist = store_hists or store_loss_hist
        store_scale_hist = store_hists or (store_scale_hist and self.raw_scale.requires_grad)
        store_lengthscales_hist = store_hists or (store_lengthscales_hist and self.raw_lengthscales.requires_grad)
        store_noise_hist = store_hists or (store_noise_hist and self.raw_noise.requires_grad)
        store_task_kernel_hist = store_hists or (store_task_kernel_hist and (self.raw_factor_task_kernel.requires_grad or self.raw_noise_task_kernel.requires_grad))
        if store_loss_hist: loss_hist = torch.empty(iterations+1)
        if store_scale_hist: scale_hist = torch.empty(torch.Size([iterations+1])+self.raw_scale.shape)
        if store_lengthscales_hist: lengthscales_hist = torch.empty(torch.Size([iterations+1])+self.raw_lengthscales.shape)
        if store_noise_hist: noise_hist = torch.empty(torch.Size([iterations+1])+self.raw_noise.shape)
        if store_task_kernel_hist: task_kernel_hist = torch.empty(torch.Size([iterations+1])+self.gram_matrix_tasks.shape)
        if masks is not None:
            masks = torch.atleast_2d(masks)
            assert masks.ndim==2
            assert len(masks)<=len(self.shape_batch)
            d_out = torch.empty(self.shape_batch)[...,*masks].numel()
        else:
            d_out = int(torch.tensor(self.shape_batch).prod())
        if verbose:
            _s = "%16s | %-10s | %-10s | %-10s"%("iter of %.1e"%iterations,"loss","term1","term2")
            print(" "*verbose_indent+_s)
            print(" "*verbose_indent+"~"*len(_s))
        mll_const = d_out*self.n.sum()*np.log(2*np.pi)
        stop_crit_best_loss = torch.inf 
        stop_crit_save_loss = torch.inf 
        stop_crit_iterations_without_improvement_loss = 0
        os.environ["FASTGP_FORCE_RECOMPILE"] = "True"
        inv_log_det_cache = self.get_inv_log_det_cache()
        for i in range(iterations+1):
            if loss_metric=="GCV":
                numer,denom = inv_log_det_cache.get_gcv_numer_denom()
                if masks is None:
                    term1 = numer 
                    term2 = denom
                else:
                    term1 = numer[...,*masks,:]
                    term2 = denom.expand(list(self.shape_batch)+[1])[...,*masks,:]
                loss = (term1/term2).sum()
                metric_val = loss
            elif loss_metric=="MLL":
                norm_term,logdet = inv_log_det_cache.get_norm_term_logdet_term()
                if masks is None:
                    term1 = norm_term.sum()
                    term2 = d_out/torch.tensor(logdet.shape).prod()*logdet.sum()
                else:
                    term1 = norm_term[...,*masks,0].sum()
                    term2 = logdet.expand(list(self.shape_batch)+[1])[...,*masks,0].sum()
                loss = 1/2*(term1+term2+mll_const)
                metric_val = -loss
            elif loss_metric=="CV":
                inv_diag = inv_log_det_cache.get_inv_diag()
                term1 = term2 = torch.nan*torch.ones(1)
                del os.environ["FASTGP_FORCE_RECOMPILE"]
                coeffs = self.coeffs # avoid repeated Gram matrix inverse computations
                os.environ["FASTGP_FORCE_RECOMPILE"] = "True"
                squared_sums = ((coeffs/inv_diag)**2*cv_weights).sum(-1,keepdim=True)
                if masks is None:
                    loss = squared_sums.sum()
                else:
                    loss = squared_sums[...,*masks,0].sum()
            else:
                assert False, "loss_metric parsing implementation error"
            if loss.item()<stop_crit_best_loss:
                stop_crit_best_loss = loss.item()
                best_params = {param[0]:param[1].data.clone() for param in self.named_parameters()}
            if (stop_crit_save_loss-loss.item())>logtol:
                stop_crit_iterations_without_improvement_loss = 0
                stop_crit_save_loss = stop_crit_best_loss
            else:
                stop_crit_iterations_without_improvement_loss += 1
            break_condition = i==iterations or stop_crit_iterations_without_improvement_loss==stop_crit_wait_iterations
            if store_loss_hist: loss_hist[i] = metric_val.item()
            if store_scale_hist: scale_hist[i] = self.scale.detach().to(scale_hist.device)
            if store_lengthscales_hist: lengthscales_hist[i] = self.lengthscales.detach().to(lengthscales_hist.device)
            if store_noise_hist: noise_hist[i] = self.noise.detach().to(noise_hist.device)
            if store_task_kernel_hist: task_kernel_hist[i] = self.gram_matrix_tasks.detach().to(task_kernel_hist.device)
            if verbose and (i%verbose==0 or break_condition):
                _s = "%16.2e | %-10.2e | %-10.2e | %-10.2e"%(i,loss.item(),term1.item() if term1.numel()==1 else torch.nan,term2.item() if term2.numel()==1 else torch.nan)
                print(" "*verbose_indent+_s)
            if break_condition: break
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        for pname,pdata in best_params.items():
            setattr(self,pname,torch.nn.Parameter(pdata,requires_grad=getattr(self,pname).requires_grad))
        del os.environ["FASTGP_FORCE_RECOMPILE"]
        data = {"iterations":i}
        if store_loss_hist: data["loss_hist"] = loss_hist[:(i+1)]
        if store_scale_hist: data["scale_hist"] = scale_hist[:(i+1)]
        if store_lengthscales_hist: data["lengthscales_hist"] = lengthscales_hist[:(i+1)]
        if store_noise_hist: data["noise_hist"] = noise_hist[:(i+1)]
        if store_task_kernel_hist: data["task_kernel_hist"] = task_kernel_hist[:(i+1)]
        return data
    def _sample(self, seq, n_min, n_max):
        x = torch.from_numpy(seq.gen_samples(n_min=int(n_min),n_max=int(n_max))).to(self.device)
        return x,x
    def get_x_next(self, n:Union[int,torch.Tensor], task:Union[int,torch.Tensor]=None):
        """
        Get the next sampling locations. 

        Args:
            n (Union[int,torch.Tensor]): maximum sample index per task
            task (Union[int,torch.Tensor]): task index
        
        Returns:
            x_next (Union[torch.Tensor,List]): next samples in the sequence
        """
        if isinstance(n,(int,np.int64)): n = torch.tensor([n],dtype=int,device=self.device) 
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
        for key in list(self.inv_log_det_cache_dict.keys()):
            if (torch.tensor(key)<self.n.cpu()).any():
                del self.inv_log_det_cache_dict[key]
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
        kmat = torch.cat([torch.cat([kmat_tasks[...,task[l0],l1,None,None]*self._kernel(x[:,None,:],self.get_xb(l1)[None,:,:],self.derivatives[task[l0]],self.derivatives[l1],self.derivatives_coeffs[task[l0]],self.derivatives_coeffs[l1]) for l1 in range(self.num_tasks)],dim=-1)[...,None,:,:] for l0 in range(len(task))],dim=-3)
        #pmean = (kmat*coeffs[...,None,None,:]).sum(-1)
        pmean = torch.einsum("...i,...i->...",kmat,coeffs[...,None,None,:])
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
        kmat_new = torch.cat([kmat_tasks[...,task[l0],task[l0],None,None]*self._kernel(x,x,self.derivatives[task[l0]],self.derivatives[task[l0]],self.derivatives_coeffs[task[l0]],self.derivatives_coeffs[task[l0]])[...,None,:] for l0 in range(len(task))],dim=-2)
        kmat = torch.cat([torch.cat([kmat_tasks[...,task[l0],l1,None,None]*self._kernel(x[:,None,:],self.get_xb(l1,n=n[l1])[None,:,:],self.derivatives[task[l0]],self.derivatives[l1],self.derivatives_coeffs[task[l0]],self.derivatives_coeffs[l1]) for l1 in range(self.num_tasks)],dim=-1)[...,None,:,:] for l0 in range(len(task))],dim=-3)
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
        kmat_new = torch.cat([torch.cat([kmat_tasks[...,task0[l0],task1[l1],None,None,None,None]*self._kernel(x0[:,None,:],x1[None,:,:],self.derivatives[task0[l0]],self.derivatives[task1[l1]],self.derivatives_coeffs[task0[l0]],self.derivatives_coeffs[task1[l1]])[...,None,None,:,:] for l1 in range(len(task1))],dim=-3) for l0 in range(len(task0))],dim=-4)
        kmat1 = torch.cat([torch.cat([kmat_tasks[...,task0[l0],l1,None,None]*self._kernel(x0[:,None,:],self.get_xb(l1,n=n[l1])[None,:,:],self.derivatives[task0[l0]],self.derivatives[l1],self.derivatives_coeffs[task0[l0]],self.derivatives_coeffs[l1]) for l1 in range(self.num_tasks)],dim=-1)[...,None,:,:] for l0 in range(len(task0))],dim=-3)
        kmat2 = kmat1 if equal else torch.cat([torch.cat([kmat_tasks[...,task1[l0],l1,None,None]*self._kernel(x1[:,None,:],self.get_xb(l1,n=n[l1])[None,:,:],self.derivatives[task1[l0]],self.derivatives[l1],self.derivatives_coeffs[task1[l0]],self.derivatives_coeffs[l1]) for l1 in range(self.num_tasks)],dim=-1)[...,None,:,:] for l0 in range(len(task1))],dim=-3)
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
    def post_error(self, x:torch.Tensor, task:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, confidence:float=0.99, eval:bool=True):
        """
        Posterior error. 

        Args:
            x (torch.Tensor[N,d]): sampling locations
            task (Union[int,torch.Tensor[T]]): task indices
            n (Union[int,torch.Tensor[num_tasks]]): number of points at which to evaluate the posterior cubature variance.
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`
            confidence (float): confidence level in $(0,1)$ for the credible interval

        Returns:
            cvar (torch.Tensor[T]): posterior variance
            quantile (np.float64):
                ```python
                scipy.stats.norm.ppf(1-(1-confidence)/2)
                ```
            perror (torch.Tensor[T]): posterior error
        """
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pvar = self.post_var(x,task=task,n=n,eval=eval,)
        pstd = torch.sqrt(pvar)
        perror = q*pstd
        return pvar,q,perror
    def post_ci(self, x:torch.Tensor, task:Union[int,torch.Tensor]=None, confidence:float=0.99, eval:bool=True):
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
            pci_low (torch.Tensor[...,T,N]): posterior credible interval lower bound
            pci_high (torch.Tensor[...,T,N]): posterior credible interval upper bound
        """
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pmean = self.post_mean(x,task=task,eval=eval)
        pvar,q,perror = self.post_error(x,task=task,confidence=confidence)
        pci_low = pmean-q*perror 
        pci_high = pmean+q*perror
        return pmean,pvar,q,pci_low,pci_high
    def post_cubature_mean(self, task:Union[int,torch.Tensor]=None, eval:bool=True):
        """
        Posterior cubature mean. 

        Args:
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`
            task (Union[int,torch.Tensor[T]]): task indices

        Returns:
            pcmean (torch.Tensor[...,T]): posterior cubature mean
        """
        raise NotImplementedError()
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
        raise NotImplementedError()
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
        raise NotImplementedError()
    def post_cubature_error(self, task:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, confidence:float=0.99, eval:bool=True):
        """
        Posterior cubature error. 

        Args:
            task (Union[int,torch.Tensor[T]]): task indices
            n (Union[int,torch.Tensor[num_tasks]]): number of points at which to evaluate the posterior cubature variance.
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`
            confidence (float): confidence level in $(0,1)$ for the credible interval

        Returns:
            pcvar (torch.Tensor[T]): posterior cubature variance
            quantile (np.float64):
                ```python
                scipy.stats.norm.ppf(1-(1-confidence)/2)
                ```
            pcerror (torch.Tensor[T]): posterior cubature error
        """
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pcvar = self.post_cubature_var(task=task,n=n,eval=eval)
        pcstd = torch.sqrt(pcvar)
        pcerror = q*pcstd
        return pcvar,q,pcerror
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
            pcci_low (torch.Tensor[...,T]): posterior cubature credible interval lower bound
            pcci_high (torch.Tensor[...,T]): posterior cubature credible interval upper bound
        """
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pcmean = self.post_cubature_mean(task=task,eval=eval) 
        pcvar,q,pcerror = self.post_cubature_error(task=task,confidence=confidence,eval=eval)
        pcci_low = pcmean-pcerror
        pcci_high = pcmean+pcerror
        return pcmean,pcvar,q,pcci_low,pcci_high
    @property
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    @property 
    def total_tuneable_parameters(self):
        return sum((p.numel() if p.requires_grad else 0) for p in self.parameters())
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
    def kernel(self, x:torch.Tensor, z:torch.Tensor, beta0:torch.Tensor=None, beta1:torch.Tensor=None, c0:torch.Tensor=None, c1:torch.Tensor=None):
        assert isinstance(x,torch.Tensor) and x.size(-1)==self.d
        assert isinstance(z,torch.Tensor) and z.size(-1)==self.d
        if beta0 is None: beta0 = torch.zeros((1,self.d),dtype=int,device=self.device)
        if beta0.shape==(len(beta0),): beta0 = beta0[None,:]
        assert isinstance(beta0,torch.Tensor) and beta0.ndim==2 and beta0.size(1)==self.d 
        if beta1 is None: beta1 = torch.zeros((1,self.d),dtype=int,device=self.device)
        if beta1.shape==(len(beta1),): beta1 = beta1[None,:]
        assert isinstance(beta1,torch.Tensor) and beta1.ndim==2 and beta1.size(1)==self.d 
        if c0 is None: c0 = torch.ones(len(beta0),device=self.device)
        assert isinstance(c0,torch.Tensor) and c0.shape==(beta0.size(0),)
        if c1 is None: c1 = torch.ones(len(beta1),device=self.device)
        assert isinstance(c1,torch.Tensor) and c1.shape==(beta1.size(0),)
        return self._kernel(x,z,beta0,beta1,c0,c1)
