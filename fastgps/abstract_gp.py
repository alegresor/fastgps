from .util import (
    _XXbSeq,
    _CoeffsCache,
)
import torch
import numpy as np 
import qmcpy 
import scipy.stats 
import os
from typing import Union,List
from collections import OrderedDict
import warnings


class AbstractGP(torch.nn.Module):
    def __init__(
            self,
            kernel,
            seqs,
            num_tasks,
            default_task,
            solo_task,
            noise,
            tfs_noise,
            requires_grad_noise,
            shape_noise,
            derivatives,
            derivatives_coeffs,
            adaptive_nugget,
        ):
        super().__init__()
        if not torch.get_default_dtype()==torch.float64:
            warnings.warn('''
                Using torch.float32 precision may significantly hurt FastGPs accuracy. 
                This is especailly evident when computing posterior variance and covariance values. 
                If possible, please use
                    torch.set_default_dtype(torch.float64)''')
        # copy kernel parameters 
        self.kernel = kernel
        assert self.kernel.torchify, "requires torchify=True for the kernel"
        self.device = self.kernel.device
        self.d = self.kernel.d
        # multi-task kernel abstraction
        assert isinstance(num_tasks,int) and num_tasks>0
        self.num_tasks = num_tasks
        self.default_task = default_task
        self.solo_task = solo_task
        self.task_range = torch.arange(num_tasks,device=self.device)
        if solo_task:
            self.kernel = qmcpy.KernelMultiTask(
                base_kernel = self.kernel,
                num_tasks = 1, 
                factor = 1.,
                diag =  0.,
                requires_grad_factor = False, 
                requires_grad_diag = False,
                tfs_diag = (qmcpy.util.transforms.tf_identity,qmcpy.util.transforms.tf_identity),
                rank_factor = 1)
        assert isinstance(self.kernel,qmcpy.KernelMultiTask)
        # seqs setup 
        assert isinstance(seqs,np.ndarray) and seqs.shape==(self.num_tasks,)
        assert all(seqs[i].d==self.d for i in range(self.num_tasks))
        self.seqs = seqs
        self.n = torch.zeros(self.num_tasks,dtype=int,device=self.device)
        self.m = -1*torch.ones(self.num_tasks,dtype=int,device=self.device)
        # derivatives setup 
        if derivatives is None: derivatives = [torch.zeros((1,self.d),dtype=torch.int64,device=self.device) for i in range(self.num_tasks)]
        if isinstance(derivatives,torch.Tensor): derivatives = [derivatives]
        assert isinstance(derivatives,list) and len(derivatives)==self.num_tasks
        derivatives = [deriv[None,:] if deriv.ndim==1 else deriv for deriv in derivatives]
        assert all((derivatives[i].ndim==2 and derivatives[i].size(1)==self.d) for i in range(self.num_tasks))
        if derivatives_coeffs is None: derivatives_coeffs = [torch.ones(len(derivatives[i]),device=self.device) for i in range(self.num_tasks)]
        assert isinstance(derivatives_coeffs,list) and len(derivatives_coeffs)==self.num_tasks
        assert all((derivatives_coeffs[i].ndim==1 and len(derivatives_coeffs[i]))==len(derivatives[i]) for i in range(self.num_tasks))
        self.derivatives_cross = [[[None,None] for l1 in range(self.num_tasks)] for l0 in range(self.num_tasks)]
        self.derivatives_coeffs_cross = [[None for l1 in range(self.num_tasks)] for l0 in range(self.num_tasks)]
        for l0 in range(self.num_tasks):
            p0r = torch.arange(len(derivatives_coeffs[l0]),device=self.device)
            for l1 in range(l0+1):
                p1r = torch.arange(len(derivatives_coeffs[l1]),device=self.device)
                i0m,i1m = torch.meshgrid(p0r,p1r,indexing="ij")
                i0,i1 = i0m.flatten(),i1m.flatten()
                self.derivatives_cross[l0][l1][0] = derivatives[l0][i0]
                self.derivatives_cross[l0][l1][1] = derivatives[l1][i1]
                self.derivatives_coeffs_cross[l0][l1] = derivatives_coeffs[l0][i0]*derivatives_coeffs[l1][i1]
                if l0!=l1:
                    self.derivatives_cross[l1][l0][0] = self.derivatives_cross[l0][l1][1]
                    self.derivatives_cross[l1][l0][1] = self.derivatives_cross[l0][l1][0]
                    self.derivatives_coeffs_cross[l1][l0] = self.derivatives_coeffs_cross[l0][l1]
        # noise
        self.raw_noise,self.tf_noise = self.kernel.parse_assign_param(
            pname = "noise",
            param = noise,
            shape_param = shape_noise,
            requires_grad_param = requires_grad_noise,
            tfs_param = tfs_noise,
            endsize_ops = [1],
            constraints = ["POSITIVE"])
        # storage and dynamic caches
        self._y = [torch.empty(0,device=self.device) for l in range(self.num_tasks)]
        self.xxb_seqs = np.array([_XXbSeq(self,self.seqs[i]) for i in range(self.num_tasks)],dtype=object)
        self.coeffs_cache = _CoeffsCache(self)
        self.inv_log_det_cache_dict = {}
        # derivative multitask setting checks 
        if any((derivatives[i]>0).any() or (derivatives_coeffs[i]!=1).any() for i in range(self.num_tasks)):
            self.kernel.raw_factor.requires_grad_(False)
            self.kernel.raw_diag.requires_grad_(False)
            assert (self.kernel.taskmat==1).all()
        self.adaptive_nugget = adaptive_nugget
        self.batch_param_names = ["noise"]
    def save_params(self, path):
        """ Save the state dict to path 
        
        Arg:
            path (str): the path. 
        """
        torch.save(self.state_dict(),path)
    def load_params(self, path):
        """ Load the state dict from path 
        
        Arg:
            path (str): the path. 
        """
        self.load_state_dict(torch.load(path,weights_only=True))
    def get_default_optimizer(self, lr):
        # return torch.optim.Adam(self.parameters(),lr=lr,amsgrad=True)
        if lr is None: lr = 1e-1
        return torch.optim.Rprop(self.parameters(),lr=lr,etas=(0.5,1.2),step_sizes=(0,10))
    def fit(
            self,
            loss_metric:str = "MLL",
            iterations:int = 5000,
            lr:float = None,
            optimizer:torch.optim.Optimizer = None,
            stop_crit_improvement_threshold:float = 5e-2,
            stop_crit_wait_iterations:int = 10,
            store_hists:bool = False,
            verbose:int = 5,
            verbose_indent:int = 4,
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
            store_hists (Union[bool,int]): store parameter data every `store_hists` iterations.
            verbose (int): log every `verbose` iterations, set to `0` for silent mode
            verbose_indent (int): size of the indent to be applied when logging, helpful for logging multiple models
            cv_weights (Union[str,torch.Tensor]): weights for cross validation
            
        Returns:
            hist_data (dict): iteration history data.
        """
        assert isinstance(loss_metric,str) and loss_metric.upper() in ["MLL","GCV","CV"] 
        assert (self.n>0).any(), "cannot fit without data"
        assert isinstance(iterations,int) and iterations>=0
        if optimizer is None:
            optimizer = self.get_default_optimizer(lr)
        assert isinstance(optimizer,torch.optim.Optimizer)
        assert isinstance(store_hists,int), "require int store_mll_hist" 
        assert (isinstance(verbose,int) or isinstance(verbose,bool)) and verbose>=0, "require verbose is a non-negative int"
        assert isinstance(verbose_indent,int) and verbose_indent>=0, "require verbose_indent is a non-negative int"
        assert np.isscalar(stop_crit_improvement_threshold) and 0<stop_crit_improvement_threshold, "require stop_crit_improvement_threshold is a positive float"
        assert (isinstance(stop_crit_wait_iterations,int) or stop_crit_wait_iterations==np.inf) and stop_crit_wait_iterations>0
        loss_metric = loss_metric.upper()
        logtol = np.log(1+stop_crit_improvement_threshold)
        if isinstance(cv_weights,str) and cv_weights.upper()=="L2R":
            _y = self.y 
            if _y.ndim==1:
                cv_weights = 1/torch.abs(_y) 
            else:
                cv_weights = 1/torch.linalg.norm(_y,2,dim=[i for i in range(_y.ndim-1)])
        if store_hists:
            hist_data = {}
            hist_data["iteration"] = []
            hist_data["loss"] = []
            hist_data["best_loss"] = []
            for pname in self.batch_param_names:
                hist_data[pname] = []
            for pname in self.kernel.batch_param_names:
                hist_data[pname] = []
            for pname in self.kernel.base_kernel.batch_param_names:
                hist_data[pname] = []
        else:
            hist_data = {}
        if verbose:
            _s = "%16s | %-10s | %-10s | %-10s | %-10s"%("iter of %.1e"%iterations,"best loss","loss","term1","term2")
            print(" "*verbose_indent+_s)
            print(" "*verbose_indent+"~"*len(_s))
        stop_crit_best_loss = torch.inf 
        stop_crit_save_loss = torch.inf 
        stop_crit_iterations_without_improvement_loss = 0
        os.environ["FASTGP_FORCE_RECOMPILE"] = "True"
        inv_log_det_cache = self.get_inv_log_det_cache()
        for i in range(iterations+1):
            if loss_metric=="GCV":
                numer,denom = inv_log_det_cache.get_gcv_numer_denom()
                term1 = numer 
                term2 = denom
                loss = (term1/term2).sum()
            elif loss_metric=="MLL":
                norm_term,logdet = inv_log_det_cache.get_norm_term_logdet_term()
                d_out = norm_term.numel()
                term1 = norm_term.sum()
                mll_const = d_out*self.n.sum()*np.log(2*np.pi)
                term2 = d_out/torch.tensor(logdet.shape).prod()*logdet.sum()
                loss = 1/2*(term1+term2+mll_const)
            elif loss_metric=="CV":
                coeffs = self.coeffs
                del os.environ["FASTGP_FORCE_RECOMPILE"]
                inv_diag = inv_log_det_cache.get_inv_diag()
                os.environ["FASTGP_FORCE_RECOMPILE"] = "True"
                term1 = term2 = torch.nan*torch.ones(1)
                squared_sums = ((coeffs/inv_diag)**2*cv_weights).sum(-1,keepdim=True)
                loss = squared_sums.sum().real
            else:
                assert False, "loss_metric parsing implementation error"
            if loss.item()<stop_crit_best_loss:
                stop_crit_best_loss = loss.item()
                best_params = OrderedDict([(pname,pval.clone()) for pname,pval in self.state_dict().items()])
            if (stop_crit_save_loss-loss.item())>logtol:
                stop_crit_iterations_without_improvement_loss = 0
                stop_crit_save_loss = stop_crit_best_loss
            else:
                stop_crit_iterations_without_improvement_loss += 1
            break_condition = i==iterations or stop_crit_iterations_without_improvement_loss==stop_crit_wait_iterations
            if store_hists and (break_condition or i%store_hists==0):
                hist_data["iteration"].append(i)
                hist_data["loss"].append(loss.item())
                hist_data["best_loss"].append(stop_crit_best_loss)
                for pname in self.batch_param_names:
                    hist_data[pname].append(getattr(self,pname).data.detach().clone().cpu())
                for pname in self.kernel.batch_param_names:
                    hist_data[pname].append(getattr(self.kernel,pname).data.detach().clone().cpu())
                for pname in self.kernel.base_kernel.batch_param_names:
                    hist_data[pname].append(getattr(self.kernel.base_kernel,pname).data.detach().clone().cpu())
            if verbose and (i%verbose==0 or break_condition):
                _s = "%16.2e | %-10.2e | %-10.2e | %-10.2e | %-10.2e"%(i,stop_crit_best_loss,loss.item(),term1.item() if term1.numel()==1 else torch.nan,term2.item() if term2.numel()==1 else torch.nan)
                print(" "*verbose_indent+_s)
            if break_condition: break
            # with torch.autograd.set_detect_anomaly(True):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.load_state_dict(best_params)
        del os.environ["FASTGP_FORCE_RECOMPILE"]
        if store_hists:
            hist_data["iteration"] = torch.tensor(hist_data["iteration"])
            hist_data["loss"] = torch.tensor(hist_data["loss"])
            hist_data["best_loss"] = torch.tensor(hist_data["best_loss"])
            for pname in self.batch_param_names:
                hist_data[pname] = torch.stack(hist_data[pname],dim=0)
            for pname in self.kernel.batch_param_names:
                hist_data[pname] = torch.stack(hist_data[pname],dim=0)
            for pname in self.kernel.base_kernel.batch_param_names:
                hist_data[pname] = torch.stack(hist_data[pname],dim=0)
        return hist_data
    def _sample(self, seq, n_min, n_max):
        x = torch.from_numpy(seq(n_min=int(n_min),n_max=int(n_max))).to(torch.get_default_dtype()).to(self.device)
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
        if isinstance(n,list): n = torch.tensor(n,dtype=int,device=self.device)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int,device=self.device)
        if isinstance(task,list): task = torch.tensor(task,dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and isinstance(task,torch.Tensor) and n.ndim==task.ndim==1 and len(n)==len(task)
        assert (n>=self.n[task]).all(), "maximum sequence index must be greater than the current number of samples"
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
        if isinstance(task,int): task = torch.tensor([task],dtype=int,device=self.device)
        if isinstance(task,list): task = torch.tensor(task,dtype=int,device=self.device)
        assert isinstance(y_next,list) and isinstance(task,torch.Tensor) and task.ndim==1 and len(y_next)==len(task)
        for i,l in enumerate(task):
            self._y[l] = torch.cat([self._y[l],y_next[i]],-1)
        self.n = torch.tensor([self._y[i].size(-1) for i in range(self.num_tasks)],dtype=int,device=self.device)
        self.m = torch.where(self.n==0,-1,torch.log2(self.n).round()).to(int) # round to avoid things like torch.log2(torch.tensor([2**3],dtype=torch.int64,device="cuda")).item() = 2.9999999999999996
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
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int,device=self.device)
        if isinstance(task,list): task = torch.tensor(task,dtype=int,device=self.device)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        kmat = torch.cat([torch.cat([self.kernel(task[l0],l1,x[:,None,:],self.get_xb(l1)[None,:,:],*self.derivatives_cross[task[l0]][l1],self.derivatives_coeffs_cross[task[l0]][l1]) for l1 in range(self.num_tasks)],dim=-1)[...,None,:,:] for l0 in range(len(task))],dim=-3)
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
        assert isinstance(n,torch.Tensor)
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int,device=self.device)
        if isinstance(task,list): task = torch.tensor(task,dtype=int,device=self.device)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        kmat_new = torch.cat([self.kernel(task[l0],task[l0],x,x,*self.derivatives_cross[task[l0]][task[l0]],self.derivatives_coeffs_cross[task[l0]][task[l0]])[...,None,:] for l0 in range(len(task))],dim=-2)
        kmat = torch.cat([torch.cat([self.kernel(task[l0],l1,x[:,None,:],self.get_xb(l1,n=n[l1])[None,:,:],*self.derivatives_cross[task[l0]][l1],self.derivatives_coeffs_cross[task[l0]][l1]) for l1 in range(self.num_tasks)],dim=-1)[...,None,:,:] for l0 in range(len(task))],dim=-3)
        kmat_perm = torch.permute(kmat,[-3,-2]+[i for i in range(kmat.ndim-3)]+[-1])
        t_perm = inv_log_det_cache.gram_matrix_solve(kmat_perm)
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
        assert isinstance(n,torch.Tensor)
        assert x0.ndim==2 and x0.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        assert x1.ndim==2 and x1.size(1)==self.d, "z must a torch.Tensor with shape (-1,d)"
        inv_log_det_cache = self.get_inv_log_det_cache(n)
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
        equal = torch.equal(x0,x1) and torch.equal(task0,task1)
        kmat_new = torch.cat([torch.cat([self.kernel(task0[l0],task1[l1],x0[:,None,:],x1[None,:,:],*self.derivatives_cross[task0[l0]][task1[l1]],self.derivatives_coeffs_cross[task0[l0]][task1[l1]])[...,None,None,:,:] for l1 in range(len(task1))],dim=-3) for l0 in range(len(task0))],dim=-4)
        kmat1 = torch.cat([torch.cat([self.kernel(task0[l0],l1,x0[:,None,:],self.get_xb(l1,n=n[l1])[None,:,:],*self.derivatives_cross[task0[l0]][l1],self.derivatives_coeffs_cross[task0[l0]][l1]) for l1 in range(self.num_tasks)],dim=-1)[...,None,:,:] for l0 in range(len(task0))],dim=-3)
        kmat2 = kmat1 if equal else torch.cat([torch.cat([self.kernel(task1[l0],l1,x1[:,None,:],self.get_xb(l1,n=n[l1])[None,:,:],*self.derivatives_cross[task1[l0]][l1],self.derivatives_coeffs_cross[task1[l0]][l1]) for l1 in range(self.num_tasks)],dim=-1)[...,None,:,:] for l0 in range(len(task1))],dim=-3)
        kmat2_perm = torch.permute(kmat2,[-3,-2]+[i for i in range(kmat2.ndim-3)]+[-1])
        t_perm = inv_log_det_cache.gram_matrix_solve(kmat2_perm)
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
    def noise(self):
        """
        Noise parameter.
        """
        return self.tf_noise(self.raw_noise)
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
    def kernel_abstracted(self, x:torch.Tensor, z:torch.Tensor, beta0:torch.Tensor=None, beta1:torch.Tensor=None, c0:torch.Tensor=None, c1:torch.Tensor=None):
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
