import torch
import numpy as np
from typing import Union,List
from .abstract_gp import AbstractGP
from .util import _freeze,_frozen_equal,_force_recompile
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
        self.k1parts = [[None for l1 in range(self.num_tasks)] for l0 in range(self.num_tasks)]
        self.n_k1parts = np.array([[0 if l1>=l0 else -1 for l1 in range(self.num_tasks)] for l0 in range(self.num_tasks)],dtype=int)
        self.lam_caches = np.array([[self._LamCaches(self,l0,l1,*self.derivatives_cross[l0][l1],self.derivatives_coeffs_cross[l0][l1]) if l1>=l0 else None for l1 in range(self.num_tasks)] for l0 in range(self.num_tasks)],dtype=object)
        self.ytilde = [None]*self.num_tasks
        self.n_ytilde = np.zeros(self.num_tasks,dtype=int)
    class _LamCaches(object):
        def __init__(self, fgp, l0, l1, beta0, beta1, c):
            self.fgp = fgp
            self.l0 = l0
            self.l1 = l1
            assert c.ndim==1
            assert beta0.shape==(len(c),self.fgp.d) and beta1.shape==(len(c),self.fgp.d)
            self.c = c 
            self.beta0 = beta0 
            self.beta1 = beta1
            self.m_min,self.m_max = -1,-1
            self.raw_scale_freeze_list = [None]
            self.raw_lengthscales_freeze_list = [None]
            self.raw_alpha_freeze_list = [None]
            self.raw_noise_freeze_list = [None]
            self._freeze(0)
            self.lam_list = [torch.empty(0,dtype=self.fgp._FTOUTDTYPE,device=self.fgp.device)]
        def _frozen_equal(self, i):
            return (
                (self.fgp.kernel.base_kernel.raw_scale==self.raw_scale_freeze_list[i]).all() and 
                (self.fgp.kernel.base_kernel.raw_lengthscales==self.raw_lengthscales_freeze_list[i]).all() and 
                (self.fgp.kernel.base_kernel.raw_alpha==self.raw_alpha_freeze_list[i]).all() and 
                (self.fgp.raw_noise==self.raw_noise_freeze_list[i]).all())
        def _force_recompile(self):
            return os.environ.get("FASTGP_FORCE_RECOMPILE")=="True" and (
                self.fgp.kernel.base_kernel.raw_scale.requires_grad or 
                self.fgp.kernel.base_kernel.raw_lengthscales.requires_grad or 
                self.fgp.kernel.base_kernel.raw_alpha.requires_grad or 
                self.fgp.raw_noise.requires_grad)
        def _freeze(self, i):
            self.raw_scale_freeze_list[i] = self.fgp.kernel.base_kernel.raw_scale.clone()
            self.raw_lengthscales_freeze_list[i] = self.fgp.kernel.base_kernel.raw_lengthscales.clone()
            self.raw_alpha_freeze_list[i] = self.fgp.kernel.base_kernel.raw_alpha.clone()
            self.raw_noise_freeze_list[i] = self.fgp.raw_noise.clone()
        def __getitem__no_delete(self, m):
            if isinstance(m,torch.Tensor):
                assert m.numel()==1 and isinstance(m,torch.int64)
                m = m.item()
            assert isinstance(m,int)
            assert m>=self.m_min, "old lambda are not retained after updating"
            if self.m_min==-1 and m>=0:
                batch_params = self.fgp.kernel.base_kernel.get_batch_params(1)
                k1 = self.fgp.kernel.base_kernel.combine_per_dim_components(self.fgp.get_k1parts(self.l0,self.l1,2**m),self.beta0,self.beta1,self.c,batch_params)
                self.lam_list = [self.fgp.ft(k1)]
                self._freeze(0)
                self.m_min = self.m_max = m
                return self.lam_list[0]
            if m==self.m_min:
                if not self._frozen_equal(0) or self._force_recompile():
                    batch_params = self.fgp.kernel.base_kernel.get_batch_params(1)
                    k1 = self.fgp.kernel.base_kernel.combine_per_dim_components(self.fgp.get_k1parts(self.l0,self.l1,2**self.m_min),self.beta0,self.beta1,self.c,batch_params)
                    self.lam_list[0] = self.fgp.ft(k1)
                    self._freeze(0)
                return self.lam_list[0]
            if m>self.m_max:
                self.lam_list += [torch.empty(2**mm,dtype=self.fgp._FTOUTDTYPE,device=self.fgp.device) for mm in range(self.m_max+1,m+1)]
                self.raw_scale_freeze_list += [torch.empty_like(self.raw_scale_freeze_list[0])]*(m-self.m_max)
                self.raw_lengthscales_freeze_list += [torch.empty_like(self.raw_lengthscales_freeze_list[0])]*(m-self.m_max)
                self.raw_alpha_freeze_list += [torch.empty_like(self.raw_alpha_freeze_list[0])]*(m-self.m_max)
                self.raw_noise_freeze_list += [torch.empty_like(self.raw_noise_freeze_list[0])]*(m-self.m_max)
                self.m_max = m
            midx = m-self.m_min
            if not self._frozen_equal(midx) or self._force_recompile():
                omega_m = self.fgp.omega(m-1).to(self.fgp.device)
                batch_params = self.fgp.kernel.base_kernel.get_batch_params(1)
                k1_m = self.fgp.kernel.base_kernel.combine_per_dim_components(self.fgp.get_k1parts(self.l0,self.l1,slice(2**(m-1),2**m)),self.beta0,self.beta1,self.c,batch_params)
                lam_m = self.fgp.ft(k1_m)
                omega_lam_m = omega_m*lam_m
                lam_m_prev = self.__getitem__no_delete(m-1)
                self.lam_list[midx] = torch.cat([lam_m_prev+omega_lam_m,lam_m_prev-omega_lam_m],-1)/np.sqrt(2)
                self._freeze(midx)
            return self.lam_list[midx]
        def __getitem__(self, m):
            lam = self.__getitem__no_delete(m)
            while self.m_min<max(self.fgp.m[self.l0],self.fgp.m[self.l1]):
                del self.lam_list[0]
                del self.raw_scale_freeze_list[0]
                del self.raw_lengthscales_freeze_list[0]
                del self.raw_alpha_freeze_list[0]
                del self.raw_noise_freeze_list[0]
                self.m_min += 1
            return lam
    class _FastInverseLogDetCache:
        def __init__(self, fgp, n):
            self.fgp = fgp
            self.n = n
            self.task_order = self.n.argsort(descending=True)
            self.inv_task_order = self.task_order.argsort()
        def __call__(self):
            if not hasattr(self,"inv") or not _frozen_equal(self.fgp,self.state_dict) or _force_recompile(self.fgp):
                n = self.n[self.task_order]
                kmat_tasks = self.fgp.kernel.taskmat
                lams = np.empty((self.fgp.num_tasks,self.fgp.num_tasks),dtype=object)
                for l0 in range(self.fgp.num_tasks):
                    to0 = self.task_order[l0]
                    for l1 in range(l0,self.fgp.num_tasks):
                        to1 = self.task_order[l1]
                        lam = self.fgp.get_lam(to0,to1,n[l0]) if to0<=to1 else self.fgp.get_lam(to1,to0,n[l0]).conj()
                        lams[l0,l1] = kmat_tasks[...,to0,to1,None]*torch.sqrt(n[l1])*lam
                if self.fgp.adaptive_nugget:
                    tr00 = lams[self.inv_task_order[0],self.inv_task_order[0]].sum(-1)
                    for l in range(self.fgp.num_tasks):
                        trll = lams[l,l].sum(-1)
                        lams[l,l] = lams[l,l]+self.fgp.noise*(trll/tr00).abs()
                else:
                    for l in range(self.fgp.num_tasks):
                        lams[l,l] = lams[l,l]+self.fgp.noise
                self.logdet = torch.log(torch.abs(lams[0,0])).sum(-1)
                A = (1/lams[0,0])[...,None,None,:]
                for l in range(1,self.fgp.num_tasks):
                    if n[l]==0: break
                    _B = torch.cat([lams[k,l] for k in range(l)],dim=-1)
                    B = _B.reshape(_B.shape[:-1]+torch.Size([-1,n[l]]))
                    Bvec = B.reshape(B.shape[:-2]+(1,A.size(-2),-1))
                    _T = (Bvec*A).sum(-2)
                    T = _T.reshape(_T.shape[:-2]+torch.Size([-1,n[l]]))
                    M = (B.conj()*T).sum(-2)
                    S = lams[l,l]-M
                    self.logdet += torch.log(torch.abs(S)).sum(-1)
                    P = T/S[...,None,:]
                    C = P[...,:,None,:]*(T[...,None,:,:].conj())
                    r = A.size(-1)//C.size(-1)
                    ii = torch.arange(A.size(-2))
                    jj = torch.arange(A.size(-1))
                    ii0,ii1,ii2 = torch.meshgrid(ii,ii,jj,indexing="ij")
                    ii0,ii1,ii2 = ii0.ravel(),ii1.ravel(),ii2.ravel()
                    jj0 = ii2%C.size(-1)
                    jj1 = ii2//C.size(-1)
                    C[...,ii0*r+jj1,ii1*r+jj1,jj0] += A[...,ii0,ii1,ii2]
                    ur = torch.cat([C,-P[...,:,None,:]],dim=-2)
                    br = torch.cat([-P.conj()[...,None,:,:],1/S[...,None,None,:]],dim=-2)
                    A = torch.cat([ur,br],dim=-3)
                if os.environ.get("FASTGP_DEBUG")=="True":
                    lammats = np.empty((self.fgp.num_tasks,self.fgp.num_tasks),dtype=object)
                    for l0 in range(self.fgp.num_tasks):
                        for l1 in range(l0,self.fgp.num_tasks):
                            lammats[l0,l1] = (lams[l0,l1].reshape((-1,n[l1],1))*torch.eye(n[l1])).reshape((-1,n[l1]))
                            if l0==l1: continue 
                            lammats[l1,l0] = lammats[l0,l1].conj().transpose(dim0=-2,dim1=-1)
                    lammat = torch.vstack([torch.hstack(lammats[i].tolist()) for i in range(self.fgp.num_tasks)])
                    assert torch.allclose(torch.logdet(lammat).real,self.logdet)
                    Afull = torch.vstack([torch.hstack([A[l0,l1]*torch.eye(A.size(-1)) for l1 in range(A.size(1))]) for l0 in range(A.size(0))])
                    assert torch.allclose(torch.linalg.inv(lammat),Afull,rtol=1e-4)
                self.state_dict = _freeze(self.fgp)
                self.inv = A
            return self.inv,self.logdet
        def gram_matrix_solve(self, y):
            inv,logdet = self()
            return self._gram_matrix_solve(y,inv)
        def _gram_matrix_solve(self, y, inv):
            assert y.size(-1)==self.n.sum() 
            ys = y.split(self.n.tolist(),dim=-1)
            yst = [self.fgp.ft(ys[i]) for i in range(self.fgp.num_tasks)]
            yst = self._gram_matrix_solve_tilde_to_tilde(yst,inv)
            ys = [self.fgp.ift(yst[i]).real for i in range(self.fgp.num_tasks)]
            y = torch.cat(ys,dim=-1)
            return y
        def _gram_matrix_solve_tilde_to_tilde(self, zst, inv):
            zsto = [zst[o] for o in self.task_order]
            z = torch.cat(zsto,dim=-1)
            z = z.reshape(list(zsto[0].shape[:-1])+[1,-1,self.n[self.n>0].min()])
            z = (z*inv).sum(-2)
            z = z.reshape(list(z.shape[:-2])+[-1])
            zsto = z.split(self.n[self.task_order].tolist(),dim=-1)
            zst = [zsto[o] for o in self.inv_task_order]
            return zst
        def compute_mll_loss(self, update_prior_mean):
            inv,logdet = self()
            ytildes = [self.fgp.get_ytilde(i) for i in range(self.fgp.num_tasks)]
            sqrtn = torch.sqrt(self.fgp.n)
            if update_prior_mean:
                rhs = self._gram_matrix_solve_tilde_to_tilde(ytildes,inv)
                rhs = torch.cat([rhs_i[...,0,None] for rhs_i in rhs],dim=-1).real
                to = self.task_order
                ito = self.inv_task_order
                nord = self.fgp.n[to]
                mvec = torch.hstack([torch.zeros(1,device=self.fgp.device),(nord/nord[-1]).cumsum(0)]).to(int)[:-1]
                tasksums = sqrtn*inv[...,0][...,mvec,:][...,:,mvec][...,ito,:][...,:,ito].real
                self.fgp.prior_mean = torch.linalg.solve(tasksums,rhs[...,None])[...,0]
            deltatildescat = torch.cat(ytildes,dim=-1)
            deltatildescat[...,self.fgp.n_cumsum] = deltatildescat[...,self.fgp.n_cumsum]-sqrtn*self.fgp.prior_mean
            ztildes = self._gram_matrix_solve_tilde_to_tilde(deltatildescat.split(self.n.tolist(),dim=-1),inv)
            ztildescat = torch.cat(ztildes,dim=-1)
            norm_term = (deltatildescat.conj()*ztildescat).real.sum(-1,keepdim=True)
            logdet = logdet[...,None]
            d_out = norm_term.numel()
            term1 = norm_term.sum()
            mll_const = d_out*self.fgp.n.sum()*np.log(2*np.pi)
            term2 = d_out/torch.tensor(logdet.shape).prod()*logdet.sum()
            mll_loss = 1/2*(term1+term2+mll_const)
            return mll_loss
        def gcv_loss(self, update_prior_mean):
            inv,logdet = self()
            ytildes = [self.fgp.get_ytilde(i) for i in range(self.fgp.num_tasks)]
            sqrtn = torch.sqrt(self.fgp.n)
            if update_prior_mean:
                rhs = self._gram_matrix_solve_tilde_to_tilde(ytildes,inv)
                rhs = self._gram_matrix_solve_tilde_to_tilde(rhs,inv)
                rhs = torch.cat([rhs_i[...,0,None] for rhs_i in rhs],dim=-1).real
                to = self.task_order
                ito = self.inv_task_order
                nord = self.fgp.n[to]
                mvec = torch.hstack([torch.zeros(1,device=self.fgp.device),(nord/nord[-1]).cumsum(0)]).to(int)[:-1]
                inv2 = torch.einsum("...ij,...jk->...ik",inv[...,0],inv[...,0])
                tasksums = sqrtn*inv2[...,mvec,:][...,:,mvec][...,ito,:][...,:,ito].real
                self.fgp.prior_mean = torch.linalg.solve(tasksums,rhs[...,None])[...,0]
            deltatildescat = torch.cat(ytildes,dim=-1)
            deltatildescat[...,self.fgp.n_cumsum] = deltatildescat[...,self.fgp.n_cumsum]-torch.sqrt(self.fgp.n)*self.fgp.prior_mean
            ztildes = self._gram_matrix_solve_tilde_to_tilde(deltatildescat.split(self.n.tolist(),dim=-1),inv)
            ztildescat = torch.cat(ztildes,dim=-1)
            numer = (ztildescat.conj()*ztildescat).real.sum(-1,keepdim=True)
            n = inv.size(-2)
            nrange = torch.arange(n,device=self.fgp.device)
            tr_k_inv = inv[...,nrange,nrange,:].real.sum(-1).sum(-1,keepdim=True)
            denom = ((tr_k_inv/self.n.sum())**2).real
            gcv_loss = (numer/denom).sum()
            return gcv_loss
        def cv_loss(self, cv_weights, update_prior_mean):
            assert not update_prior_mean, "fast GP updates to prior mean with CV loss not yet worked out"
            if self.fgp.num_tasks==1:
                inv,logdet = self()
                coeffs = self._gram_matrix_solve(torch.cat([self.fgp._y[i]-self.fgp.prior_mean[...,i,None] for i in range(self.fgp.num_tasks)],dim=-1),inv)
                inv_diag = inv[0,0].sum()/self.fgp.n
                squared_sums = ((coeffs/inv_diag)**2*cv_weights).sum(-1,keepdim=True)
                cv_loss = squared_sums.sum().real
            else:
                assert False, "fast multitask GPs do not yet support efficient CV loss computation"
            return cv_loss
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
            self.inv_log_det_cache_dict[ntup] = self._FastInverseLogDetCache(self,n)
        return self.inv_log_det_cache_dict[ntup]
    def post_cubature_mean(self, task:Union[int,torch.Tensor]=None, eval:bool=True):
        kmat_tasks = self.kernel.taskmat
        coeffs = self.coeffs
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int,device=self.device)
        if isinstance(task,list): task = torch.tensor(task,dtype=int,device=self.device)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
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
        if inttask: task = torch.tensor([task],dtype=int,device=self.device)
        if isinstance(task,list): task = torch.tensor(task,dtype=int,device=self.device)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        inv_cut = inv[...,mvec,:,:][...,:,mvec,:][...,0]
        kmat_tasks_left = kmat_tasks[...,task,:][...,:,to].to(self._FTOUTDTYPE)
        kmat_tasks_right = kmat_tasks[...,to,:][...,:,task].to(self._FTOUTDTYPE)
        term = torch.einsum("...ij,...jk,...ki->...i",kmat_tasks_left,nsqrts*inv_cut,kmat_tasks_right).real
        s = self.kernel.base_kernel.scale
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
        return self.lam_caches[task0,task1][m]
    def get_k1parts(self, task0, task1, i=None):
        assert 0<=task0<self.num_tasks
        assert 0<=task1<self.num_tasks
        assert task0<=task1
        if i is None: i = self.n[task0]
        if isinstance(i,int): i = slice(None,i,None)
        if isinstance(i,torch.Tensor):
            assert i.numel()==1 and isinstance(i,torch.int64)
            i = slice(None,i.item(),None)
        assert isinstance(i,slice)
        if i.stop>self.n_k1parts[task0,task1]:
            xb_next = self.get_xb(task0,slice(self.n_k1parts[task0,task1],i.stop))
            xb0 = self.get_xb(task1,slice(0,1))
            k1parts_next = self.kernel.base_kernel.get_per_dim_components(xb_next,xb0,*self.derivatives_cross[task0][task1])
            if self.k1parts[task0][task1] is None:
                self.k1parts[task0][task1] = k1parts_next 
            else:
                self.k1parts[task0][task1] = torch.cat([self.k1parts[task0][task1],k1parts_next],dim=0)
            self.n_k1parts[task0,task1] = i.stop
        return self.k1parts[task0][task1][i]
    def get_ytilde(self, task):
        assert 0<=task<self.num_tasks
        if self.ytilde[task] is None or self.n_ytilde[task]<=1:
            self.ytilde[task] = self.ft(self._y[task]) if self.n[task]>1 else self._y[task].clone().to(self._FTOUTDTYPE)
            self.n_ytilde[task] = self.n[task].item()
            return self.ytilde[task]
        while self.n_ytilde[task]!=self.n[task]:
            n_double = 2*self.n_ytilde[task]
            ytilde_next = self.ft(self._y[task][...,self.n_ytilde[task]:n_double])
            omega_m = self.omega(int(np.log2(self.n_ytilde[task]))).to(self.device)
            omega_ytilde_next = omega_m*ytilde_next
            self.ytilde[task] = torch.cat([self.ytilde[task]+omega_ytilde_next,self.ytilde[task]-omega_ytilde_next],-1)/np.sqrt(2)
            self.n_ytilde[task] = n_double
        return self.ytilde[task]
    def get_inv_log_det(self, n=None):
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        return inv_log_det_cache()
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

