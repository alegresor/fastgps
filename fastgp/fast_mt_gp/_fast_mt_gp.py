import torch 
import numpy as np 
import scipy.stats 
import os
import qmcpy
import itertools 

class _XXbSeq(object):
    def __init__(self, fgp, seq):
        self.fgp = fgp
        self.seq = seq
        self.n = 0
        self.x = torch.empty((0,seq.d))
        self.xb = torch.empty((0,seq.d),dtype=self.fgp._XBDTYPE)
    def __getitem__(self, i):
        if isinstance(i,int): i = slice(None,i,None) 
        assert isinstance(i,slice)
        if i.stop>self.n:
            x_next,xb_next = self.fgp._sample(self.seq,self.n,i.stop)
            if x_next.data_ptr()==xb_next.data_ptr():
                self.x = self.xb = torch.vstack([self.x,x_next])
            else:
                self.x = torch.vstack([self.x,x_next])
                self.xb = torch.vstack([self.xb,xb_next])
            self.n = i.stop
        return self.x[i],self.xb[i]

class _K1PartsSeq(object):
    def __init__(self, fgp, xxb_seq_first, xxb_seq_second, _kernel_parts):
        self.xxb_seq_first = xxb_seq_first
        self.xxb_seq_second = xxb_seq_second
        self._kernel_parts = _kernel_parts
        self.k1parts = torch.empty((0,fgp.d))
        self.n = 0
    def __getitem__(self, i):
        if isinstance(i,int): i = slice(None,i,None) 
        assert isinstance(i,slice)
        if i.stop>self.n:
            _,xb_next = self.xxb_seq_first[self.n:i.stop]
            _,xb0 = self.xxb_seq_second[:1]
            k1parts_next = self._kernel_parts(xb_next,xb0)
            self.k1parts = torch.vstack([self.k1parts,k1parts_next])
            self.n = i.stop
        return self.k1parts[i]

class _LamCaches(object):
    def __init__(self, fgp, i0, i1):
        self.fgp = fgp
        self.i0 = i0
        self.i1 = i1
        self.m_min,self.m_max = -1,-1
        self.raw_scale_freeze_list = [None]
        self.raw_lengthscales_freeze_list = [None]
        self.raw_noise_freeze_list = [None]
        self._freeze(0)
        self.lam_list = [torch.empty(0,dtype=self.fgp._FTOUTDTYPE)]
    def _frozen_equal(self, i):
        return (
            (self.fgp.raw_scale==self.raw_scale_freeze_list[i]).all() and 
            (self.fgp.raw_lengthscales==self.raw_lengthscales_freeze_list[i]).all() and 
            (self.fgp.raw_noise==self.raw_noise_freeze_list[i]).all())
    def _freeze(self, i):
        self.raw_scale_freeze_list[i] = self.fgp.raw_scale.clone()
        self.raw_lengthscales_freeze_list[i] = self.fgp.raw_lengthscales.clone()
        self.raw_noise_freeze_list[i] = self.fgp.raw_noise.clone()
    def __getitem__no_delete(self, m):
        assert isinstance(m,int)
        assert m>=self.m_min, "old lambda are not retained after updating"
        if self.m_min==-1 and m>=0:
            k1 = self.fgp._kernel_from_parts(self.fgp.get_k1parts(self.i0,self.i1))
            if self.i0==self.i1:
                k1[0] += self.fgp.noise
            self.lam_list = [np.sqrt(self.fgp.n[self.i1])*self.fgp.ft(k1)]
            FASTGP_DEBUG = os.environ.get("FASTGP_DEBUG")
            if FASTGP_DEBUG=="True":
                kfull = self.fgp.kernel(self.fgp.get_x(self.i0)[:,None,:],self.fgp.get_x(self.i1)[None,:,:])
                if self.i0==self.i1:
                    kfull += self.fgp.noise*torch.eye(kfull.size(0))
                lamfullhat = self.fgp.ft(self.fgp.ft(kfull.T).conj().T).conj()
                lamfull = (self.lam_list[0].reshape((-1,self.fgp.n[self.i1],1))*torch.eye(self.fgp.n[self.i1])).reshape((self.fgp.n[self.i0],self.fgp.n[self.i1]))
                assert torch.allclose(lamfullhat,lamfull)
            self._freeze(0)
            self.m_min = self.m_max = self.fgp.m(self.i0)
            return self.lam_list[0]
        if m==self.m_min:
            assert False, r"account for sqrt(2^{m_\ell'})"
            if not self._frozen_equal(0):
                k1 = self.fgp._kernel_from_parts(self.fgp.k1parts_seq[self.i0,self.i1][:2**self.m_min])
                k1[0] += self.fgp.noise
                self.lam_list[0] = np.sqrt(2**self.m_min)*self.fgp.ft(k1)
                self._freeze(0)
            return self.lam_list[0]
        if m>self.m_max:
            assert False, r"account for sqrt(2^{m_\ell'})"
            self.lam_list += [torch.empty(2**mm,dtype=self.fgp._FTOUTDTYPE) for mm in range(self.m_max+1,m+1)]
            self.raw_scale_freeze_list += [torch.empty_like(self.raw_scale_freeze_list[0])]*(m-self.m_max)
            self.raw_lengthscales_freeze_list += [torch.empty_like(self.raw_lengthscales_freeze_list[0])]*(m-self.m_max)
            self.raw_noise_freeze_list += [torch.empty_like(self.raw_noise_freeze_list[0])]*(m-self.m_max)
            self.m_max = m
        midx = m-self.m_min
        if not self._frozen_equal(midx):
            omega_m = self.fgp.get_omega(m-1)
            k1_m = self.fgp._kernel_from_parts(self.fgp.k1parts_seq[self.i0,self.i1][2**(m-1):2**m])
            lam_m = np.sqrt(2**(m-1))*self.fgp.ft(k1_m)
            omega_lam_m = omega_m*lam_m
            lam_m_prev = self.__getitem__no_delete(m-1)
            self.lam_list[midx] = torch.hstack([lam_m_prev+omega_lam_m,lam_m_prev-omega_lam_m])
            FASTGP_DEBUG = os.environ.get("FASTGP_DEBUG")
            if FASTGP_DEBUG=="True":
                k1_full = self.fgp._kernel_from_parts(self.fgp.k1parts_seq[self.i0,self.i1][:2**m])
                lam_full = np.sqrt(2**m)*self.fgp.ft(k1_full)
                assert torch.allclose(self.lam_list[midx],lam_full,atol=1e-7,rtol=0)
            self._freeze(midx)
        return self.lam_list[midx]
    def __getitem__(self, m):
        lam = self.__getitem__no_delete(m)
        while self.m_min<self.fgp.m(self.i0):
            del self.lam_list[0]
            del self.raw_scale_freeze_list[0]
            del self.raw_lengthscales_freeze_list[0]
            del self.raw_noise_freeze_list[0]
            self.m_min += 1
        return lam

class _YtildeCache(object):
    def __init__(self, fgp, i):
        self.fgp = fgp
        self.i = i
    def __call__(self):
        if not hasattr(self,"ytilde"):
            self.ytilde = self.fgp.ft(self.fgp.y[self.i]) if self.fgp.n[self.i]>0 else self.fgp.y[self.i].clone()
            return self.ytilde
        while self.fgp.y.shape!=self.ytilde.shape:
            n_curr = self.ytilde.size(-1)
            ytilde_next = self.fgp.ft(self.fgp.y[...,n_curr:(2*n_curr)])
            omega_m = self.fgp.get_omega(int(np.log2(n_curr)))
            omega_ytilde_next = omega_m*ytilde_next
            self.ytilde = 1/np.sqrt(2)*torch.hstack([
                self.ytilde+omega_ytilde_next,
                self.ytilde-omega_ytilde_next])
            FASTGP_DEBUG = os.environ.get("FASTGP_DEBUG")
            if FASTGP_DEBUG=="True":
                ytilde_ref = self.fgp.ft(self.fgp.y[:(2*n_curr)])
                assert torch.allclose(self.ytilde,ytilde_ref,atol=1e-7,rtol=0)
        return self.ytilde 

class _InverseLogDetCache(object):
    def __init__(self, fgp):
        self.fgp = fgp 
        self._freeze()
    def _frozen_equal(self):
        return (
            (self.fgp.raw_scale==self.raw_scale_freeze).all() and 
            (self.fgp.raw_lengthscales==self.raw_lengthscales_freeze).all() and 
            (self.fgp.raw_noise==self.raw_noise_freeze).all())
    def _freeze(self):
        self.raw_scale_freeze = self.fgp.raw_scale.clone()
        self.raw_lengthscales_freeze = self.fgp.raw_lengthscales.clone()
        self.raw_noise_freeze = self.fgp.raw_noise.clone()
    def __call__(self):
        # TODO: permute self.inv based on level ordering
        # TODO: Account for index kernel
        if not hasattr(self,"inv") or (self.n!=self.fgp.n).any() or (not self._frozen_equal()):
            o = self.fgp.task_order
            n = self.fgp.n[o]
            lams = np.empty((self.fgp.num_tasks,self.fgp.num_tasks),dtype=object)
            for i0 in range(self.fgp.num_tasks):
                for i1 in range(i0,self.fgp.num_tasks):
                    lams[i0,i1] = self.fgp.get_lam(o[i0],o[i1])
            self.logdet = torch.log(torch.abs(lams[0,0])).sum()
            A = (1/lams[0,0])[None,None,:]
            for l in range(1,self.fgp.num_tasks):
                B = torch.cat([lams[k,l] for k in range(l)],dim=0).reshape((-1,n[l]))
                Bvec = B.reshape((A.size(1),-1))
                T = (Bvec*A).sum(1).reshape((-1,n[l]))
                M = (B.conj()*T).sum(0)
                S = lams[l,l]-M
                self.logdet += torch.log(torch.abs(S)).sum()
                Sinv = 1/S
                P = T*Sinv
                C = P[:,None,:]*(T[None,:,:].conj())
                r = A.size(-1)//C.size(-1)
                ii = torch.arange(A.size(0))
                jj = torch.arange(A.size(-1))
                ii0,ii1,ii2 = torch.meshgrid(ii,ii,jj,indexing="ij")
                ii0,ii1,ii2 = ii0.ravel(),ii1.ravel(),ii2.ravel()
                jj0 = ii2%C.size(-1)
                jj1 = ii2//C.size(-1)
                C[ii0*r+jj1,ii1*r+jj1,jj0] += A[ii0,ii1,ii2]
                ur = torch.cat([C,-P[:,None,:]],dim=1)
                br = torch.cat([-P.conj()[None,:,:],Sinv[None,None,:]],dim=1)
                A = torch.cat([ur,br],dim=0)
            FASTGP_DEBUG = os.environ.get("FASTGP_DEBUG")
            if FASTGP_DEBUG=="True":
                lammats = np.empty((self.fgp.num_tasks,self.fgp.num_tasks),dtype=object)
                for i0 in range(self.fgp.num_tasks):
                    for i1 in range(i0,self.fgp.num_tasks):
                        lammats[i0,i1] = (lams[i0,i1].reshape((-1,n[i1],1))*torch.eye(n[i1])).reshape((-1,n[i1]))
                        if i0==i1: continue 
                        lammats[i1,i0] = lammats[i0,i1].conj().T
                lammat = torch.vstack([torch.hstack(lammats[i].tolist()) for i in range(self.fgp.num_tasks)])
                assert torch.allclose(torch.abs(torch.logdet(lammat)),self.logdet)
                Afull = torch.vstack([torch.hstack([A[i0,i1]*torch.eye(A.size(-1)) for i1 in range(A.size(1))]) for i0 in range(A.size(0))])
                assert torch.allclose(torch.linalg.inv(lammat),Afull)
            self._freeze()
            self.n = self.fgp.n.clone()
            self.inv = A
        return self.inv,self.logdet

class _FastMultiTaskGP(torch.nn.Module):
    def __init__(self,
        seqs,
        num_tasks, 
        alpha,
        scale,
        lengthscales,
        noise,
        device,
        tfs_scale,
        tfs_lengthscales,
        tfs_noise,
        requires_grad_scale, 
        requires_grad_lengthscales, 
        requires_grad_noise, 
        ft,
        ift,
        ):
        super().__init__()
        assert torch.get_default_dtype()==torch.float64, "fast transforms do not work without torch.float64 precision" 
        assert isinstance(num_tasks,int) and num_tasks>0
        self.num_tasks = num_tasks
        self.device = torch.device(device)
        assert isinstance(seqs,np.ndarray) and seqs.shape==(self.num_tasks,)
        self.d = seqs[0].d
        assert all(seqs[i].d==self.d for i in range(self.num_tasks))
        self.seqs = seqs
        self.n = torch.zeros(self.num_tasks,dtype=int)
        assert (np.isscalar(alpha) and alpha%1==0) or (isinstance(alpha,torch.Tensor) and alpha.shape==(self,d,)), "alpha should be an int or a torch.Tensor of length d"
        if np.isscalar(alpha):
            alpha = int(alpha)*torch.ones(self.d,dtype=int,device=self.device)
        self.alpha = alpha
        assert np.isscalar(scale) and scale>0, "scale should be a positive float"
        scale = torch.tensor(scale,device=self.device)
        assert len(tfs_scale)==2 and callable(tfs_scale[0]) and callable(tfs_scale[1]), "tfs_scale should be a tuple of two callables, the transform and inverse transform"
        self.tf_scale = tfs_scale[1]
        self.raw_scale = torch.nn.Parameter(tfs_scale[0](scale),requires_grad=requires_grad_scale)
        assert (np.isscalar(lengthscales) and lengthscales>0) or (isinstance(lengthscales,torch.Tensor) and lengthscales.shape==(self,d) and (lengthscales>0).all()), "lengthscales should be a float or torch.Tensor of length d and must be postivie"
        if np.isscalar(lengthscales): 
            lengthscales = lengthscales*torch.ones(self.d,device=self.device)
        assert len(tfs_lengthscales)==2 and callable(tfs_lengthscales[0]) and callable(tfs_lengthscales[1]), "tfs_lengthscales should be a tuple of two callables, the transform and inverse transform"
        self.tf_lengthscales = tfs_lengthscales[1]
        self.raw_lengthscales = torch.nn.Parameter(tfs_lengthscales[0](lengthscales),requires_grad=requires_grad_lengthscales)
        assert np.isscalar(noise) and noise>0, "noise should be a positive float"
        noise = torch.tensor(noise,device=self.device)
        assert len(tfs_noise)==2 and callable(tfs_noise[0]) and callable(tfs_noise[1]), "tfs_scale should be a tuple of two callables, the transform and inverse transform"
        self.tf_noise = tfs_noise[1]
        self.raw_noise = torch.nn.Parameter(tfs_noise[0](noise),requires_grad=requires_grad_noise)
        self.ft = ft
        self.ift = ift
        self.xxb_seqs = np.array([_XXbSeq(self,self.seqs[i]) for i in range(self.num_tasks)],dtype=object)
        self.k1parts_seq = np.array([[_K1PartsSeq(self,self.xxb_seqs[i0],self.xxb_seqs[i1],self._kernel_parts) for i1 in range(self.num_tasks)] for i0 in range(self.num_tasks)],dtype=object)
        self.lam_caches = np.array([[_LamCaches(self,i0,i1) for i1 in range(self.num_tasks)] for i0 in range(self.num_tasks)],dtype=object)
        self.ytilde_cache = np.array([_YtildeCache(self,i) for i in range(self.num_tasks)],dtype=object)
        self.inv_log_det_cache = _InverseLogDetCache(self)
    def get_x(self, task):
        assert 0<=task<self.num_tasks
        x,xb = self.xxb_seqs[task][:self.n[task]]
        return x
    def get_xb(self, task):
        assert 0<=task<self.num_tasks
        x,xb = self.xxb_seq[task][:self.n[task]]
        return xb
    def get_lam(self, task0, task1):
        assert 0<=task0<self.num_tasks
        assert 0<=task1<self.num_tasks
        return self.lam_caches[task0,task1][self.m(task0)]
    def get_k1parts(self, task0, task1):
        assert 0<=task0<self.num_tasks
        assert 0<=task1<self.num_tasks
        return self.k1parts_seq[task0,task1][:self.n[task0]]
    def get_ytilde(self, task):
        assert 0<=task<self.num_tasks
        return self.ytilde_cache[task]()
    def m(self, task):
        return -1 if self.n[task]==0 else int(np.log2(self.n[task].item()))
    def get_inv_log_det(self):
        return self.inv_log_det_cache()
    def gram_matrix_solve(self, y):
        inv,logdet = self.get_inv_log_det()
        yogdim = y.ndim 
        if yogdim==1:
            y = y[:,None] 
        assert y.size(-2)==self.n.sum() 
        z = y.transpose(dim0=-2,dim1=-1)
        zs = z.split(self.n.tolist(),dim=-1)
        zsp = [self.ft(zs[o]) for o in self.task_order]
        z = torch.cat(zsp,dim=-1).reshape(list(z.shape[:-1])+[1,-1,self.n.min()])
        z = (z*inv).sum(-2)
        z = z.reshape(list(z.shape[:-2])+[-1])
        zs = z.split(self.n[self.task_order].tolist(),dim=-1)
        zsp = [self.ift(zs[o]).real for o in self.inv_task_order]
        z = torch.cat(zsp,dim=-1).transpose(dim0=-2,dim1=-1)
        FASTGP_DEBUG = os.environ.get("FASTGP_DEBUG")
        if FASTGP_DEBUG=="True":
            kmat = torch.vstack([
                torch.hstack([self.kernel(self.get_x(ell0)[:,None,:],self.get_x(ell1)[None,:,:]) for ell1 in range(self.num_tasks)])
                for ell0 in range(self.num_tasks)])
            kmat += self.noise*torch.eye(kmat.size(0))
            assert torch.allclose(logdet,torch.logdet(kmat))
            ztrue = torch.linalg.solve(kmat,y)
            assert torch.allclose(ztrue,z)
        if yogdim==1:
            z = z[:,0]
        return z
    @property
    def scale(self):
        return self.tf_scale(self.raw_scale)
    @property
    def lengthscales(self):
        return self.tf_lengthscales(self.raw_lengthscales)
    @property
    def noise(self):
        return self.tf_noise(self.raw_noise)
    @property 
    def task_order(self):
        return self.n.argsort(descending=True)
    @property
    def inv_task_order(self):
        return self.task_order.argsort()
    def get_x_next(self, n):
        """
        Get next sampling locations. 

        Args:
            n (torch.Tensor[num_tasks]): maximum sample index for each task
        
        Returns:
            x_next (np.ndarray[num_tasks] of torch.Tensor[n[i]-self.n[i],d]): next samples in the sequence
        """
        assert isinstance(n,list) or isinstance(n,torch.Tensor)
        if isinstance(n,list):
            n = torch.tensor(n,dtype=int)
        assert isinstance(n,torch.Tensor) and n.shape==(self.num_tasks,) and (n>=self.n).all() and torch.logical_or(n==0,n&(n-1)==0).all(), "maximum sequence index must be a power of 2 greater than the current number of samples"
        return [self.xxb_seqs[i][self.n[i]:n[i]][0] for i in range(self.num_tasks)]
    def add_y_next(self, y_next):
        """
        Increase the sample size to `n`. 

        Args:
            y_next (np.ndarray[num_tasks] of torch.Tensor[...,n[i]-self.n[i]]): new function evaluations for each task
        """
        if not hasattr(self,"y"):
            self.y = y_next
        else:
            self.y = np.array([torch.cat([self.y[i],y_next[i]],-1) for i in range(self.num_tasks)],dtype=object)
        self.n = torch.tensor([self.y[i].size(-1) for i in range(self.num_tasks)],dtype=int)
        assert torch.logical_or(self.n==0,(self.n&(self.n-1)==0)).all(), "total samples must be power of 2"
    def _kernel_parts(self, x, z):
        return self._kernel_parts_from_delta(self._ominus(x,z))
    def _kernel_from_parts(self, parts):
        return self.scale*(1+self.lengthscales*parts).prod(-1)
    def _kernel_from_delta(self, delta):
        return self._kernel_from_parts(self._kernel_parts_from_delta(delta))
    def kernel(self, x:torch.Tensor, z:torch.Tensor):
        """
        Evaluate kernel

        Args:
            x (torch.Tensor[N,d]): first argument to kernel  
            z (torch.Tensor[M,d]): second argument to kernel 
        
        Returns:
            kmat (torch.Tensor[N,M]): matrix of kernel evaluations
        """
        return self._kernel_from_parts(self._kernel_parts(x,z))
    def post_mean(self, x:torch.Tensor, eval:bool=True):
        """
        Posterior mean. 

        Args:
            x (torch.Tensor[N,d]): sampling locations
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`
        
        Returns:
            pmean (torch.Tensor[*batch_shape,N]): posterior mean where `batch_shape` is inferred from `y=f(x)`
        """
        if eval:
            coeffs = self.coeffs
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        k = self.kernel(x[:,None,:],self.xb[None,:self.n,:])
        pmean = torch.einsum("il,...l->...i",k,coeffs)
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pmean
    def post_cov(self, x:torch.Tensor, z:torch.Tensor, n:int=None, eval:bool=True):
        """
        Posterior covariance. 
        If `torch.equal(x,z)` then the diagonal of the covariance matrix is forced to be non-negative. 

        Args:
            x (torch.Tensor[N,d]): sampling locations
            z (torch.Tensor[M,d]): sampling locations
            n (int): Number of points at which to evaluate the posterior cubature variance. Defaults to `n=self.n`. Must be `n=2^m` for some `m>=int(np.log2(self.n))`.  
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`
        
        Returns:
            pcov (torch.Tensor[N,M]): posterior covariance matrix
        """
        if n is None: n = self.n
        assert isinstance(n,int) and n&(n-1)==0 and n>=self.n, "require n is an int power of two greater than or equal to self.n"
        m = int(np.log2(n))
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        assert z.ndim==2 and z.size(1)==self.d, "z must a torch.Tensor with shape (-1,d)"
        equal = torch.equal(x,z)
        _,self__x = self.xxb_seq[:2**m]
        lam = self.lam_caches[m]
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        k = self.kernel(x[:,None,:],z[None,:,:])
        k1t = self.ft(self.kernel(x[:,None,:],self__x[None,:,:]))
        k2t = k1t if equal else self.ft(self.kernel(z[:,None,:],self__x[None,:,:])) 
        kmat = k-torch.einsum("il,rl->ir",k1t.conj(),k2t/lam).real
        if equal:
            nrange = torch.arange(x.size(0),device=x.device)
            diag = kmat[nrange,nrange]
            diag[diag<0] = 0 
            kmat[nrange,nrange] = diag 
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return kmat
    def post_var(self, x:torch.Tensor, n:int=None, eval:bool=True):
        """
        Posterior variance. Forced to be non-negative.  

        Args:
            x (torch.Tensor[N,d]): sampling locations
            n (int): Number of points at which to evaluate the posterior cubature variance. Defaults to `n=self.n`. Must be `n=2^m` for some `m>=int(np.log2(self.n))`.  
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`

        Returns:
            pvar (torch.Tensor[N]): posterior variance vector
        """
        if n is None: n = self.n
        assert isinstance(n,int) and n&(n-1)==0 and n>=self.n, "require n is an int power of two greater than or equal to self.n"
        m = int(np.log2(n))
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        _,self__x = self.xxb_seq[:2**m]
        lam = self.lam_caches[m]
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        k = self.kernel(x,x)
        k1t = self.ft(self.kernel(x[:,None,:],self__x[None,:,:]))
        diag = k-torch.einsum("il,il->i",k1t.conj(),k1t/lam).real
        diag[diag<0] = 0 
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return diag
    def post_ci(self, x, confidence:float=0.99, eval:bool=True):
        """
        Posterior credible interval.

        Args:
            x (torch.Tensor[N,d]): sampling locations
            confidence (float): confidence level in $(0,1)$ for the credible interval
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`

        Returns:
            pmean (torch.Tensor[*batch_shape,N]): posterior mean where `batch_shape` is inferred from `y=f(x)`
            pvar (torch.Tensor[N]): posterior variance vector
            quantile (np.float64):
                ```python
                scipy.stats.norm.ppf(1-(1-confidence)/2)
                ```
            ci_low (torch.Tensor[*batch_shape,N]): credible interval lower bound
            ci_high (torch.Tensor[*batch_shape,N]): credible interval upper bound
        """
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pmean = self.post_mean(x) 
        pvar = self.post_var(x)
        pstd = torch.sqrt(pvar)
        ci_low = pmean-q*pstd 
        ci_high = pmean+q*pstd
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pmean,pvar,q,ci_low,ci_high
    def post_cubature_mean(self, eval:bool=True):
        """
        Posterior cubature mean. 

        Args:
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`

        Returns:
            pcmean (torch.Tensor[*batch_shape]): posterior cubature mean where `batch_shape` is inferred from `y=f(x)`
        """
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        pcmean = self.scale*self.coeffs.sum()
        FASTGP_DEBUG = os.environ.get("FASTGP_DEBUG")
        if FASTGP_DEBUG=="True":
            assert torch.allclose(pcmean,self.y.mean(),atol=1e-3,rtol=0), "pcmean-self.y.mean()"
            assert torch.allclose(pcmean,self.ytilde[0].real/np.sqrt(self.n),atol=1e-3,rtol=0)
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pcmean
    def post_cubature_var(self, n:int=None, eval:bool=True):
        """
        Posterior cubature variance. 

        Args:
            n (int): Number of points at which to evaluate the posterior cubature variance. Defaults to `n=self.n`. Must be `n=2^m` for some `m>=int(np.log2(self.n))`.  
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`

        Returns:
            pcvar (torch.Tensor[*batch_shape]): posterior cubature variance where `batch_shape` is inferred from `y=f(x)`
        """
        if n is None: n = self.n
        assert isinstance(n,int) and n&(n-1)==0 and n>=self.n, "require n is an int power of two greater than or equal to self.n"
        m = int(np.log2(n))
        lam = self.lam_caches[m]
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        pcvar = self.scale-self.scale**2*n/lam[0].real
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pcvar
    def post_cubature_ci(self, confidence:float=0.99, eval:bool=True):
        """
        Posterior cubature credible.

        Args:
            confidence (float): confidence level in $(0,1)$ for the credible interval
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`
        
        Returns:
            pcmean (torch.Tensor[*batch_shape]): posterior cubature mean where `batch_shape` is inferred from `y=f(x)`
            pcvar (torch.Tensor[*batch_shape]): posterior cubature variance
            quantile (np.float64):
                ```python
                scipy.stats.norm.ppf(1-(1-confidence)/2)
                ```
            cci_low (torch.Tensor[*batch_shape]): scalar credible interval lower bound
            cci_high (torch.Tensor[*batch_shape]): scalar credible interval upper bound
        """
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pmean = self.post_cubature_mean() 
        pvar = self.post_cubature_var()
        pstd = torch.sqrt(pvar)
        ci_low = pmean-q*pstd 
        ci_high = pmean+q*pstd 
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pmean,pvar,q,ci_low,ci_high
    def fit(self,
        iterations:int = 5000,
        optimizer:torch.optim.Optimizer = None,
        lr:float = 1e-1,
        store_mll_hist:bool = True, 
        store_scale_hist:bool = True, 
        store_lengthscales_hist:bool = True,
        store_noise_hist:bool = True,
        verbose:int = 5,
        verbose_indent:int = 4,
        stop_crit_improvement_threshold:float = 1e-5,
        stop_crit_wait_iterations:int = 10,
        ):
        """
        Args:
            iterations (int): number of optimization iterations
            optimizer (torch.optim.Optimizer): optimizer defaulted to `torch.optim.Rprop(self.parameters(),lr=lr)`
            lr (float): learning rate for default optimizer
            store_mll_hist (bool): if `True`, store and return iteration data for mll
            store_scale_hist (bool): if `True`, store and return iteration data for the kernel scale parameter
            store_lengthscales_hist (bool): if `True`, store and return iteration data for the kernel lengthscale parameters
            store_noise_hist (bool): if `True`, store and return iteration data for noise
            verbose (int): log every `verbose` iterations, set to `0` for silent mode
            verbose_indent (int): size of the indent to be applied when logging, helpful for logging multiple models
            stop_crit_improvement_threshold (float): stop fitting when the maximum number of iterations is reached or the best mll is note reduced by `stop_crit_improvement_threshold` for `stop_crit_wait_iterations` iterations 
            stop_crit_wait_iterations (int): number of iterations to wait for improved mll before early stopping, see the argument description for `stop_crit_improvement_threshold`
        
        Returns:
            data (dict): iteration data which, dependeing on storage arguments, may include keys in 
                ```python
                ["mll_hist","scale_hist","lengthscales_hist","noise_hist"]
                ```
        """
        assert self.n>0, "cannot fit without data"
        assert isinstance(iterations,int) and iterations>=0
        if optimizer is None:
            assert np.isscalar(lr) and lr>0, "require lr is a positive float"
            optimizer = torch.optim.Rprop(self.parameters(),lr=lr)
        assert isinstance(optimizer,torch.optim.Optimizer)
        assert isinstance(store_mll_hist,bool), "require bool store_mll_hist" 
        assert isinstance(store_scale_hist,bool), "require bool store_scale_hist" 
        assert isinstance(store_lengthscales_hist,bool), "require bool store_lengthscales_hist" 
        assert isinstance(store_noise_hist,bool), "require bool store_noise_hist"
        assert (isinstance(verbose,int) or isinstance(verbose,bool)) and verbose>=0, "require verbose is a non-negative int"
        assert isinstance(verbose_indent,int) and verbose_indent>=0, "require verbose_indent is a non-negative int"
        assert isinstance(stop_crit_improvement_threshold,float) and 0<=stop_crit_improvement_threshold<1, "require stop_crit_improvement_threshold is a float in [0,1)"
        assert isinstance(stop_crit_wait_iterations,int) and stop_crit_wait_iterations>0
        if store_mll_hist:
            mll_hist = torch.empty(iterations+1)
        store_scale_hist = store_scale_hist and self.raw_scale.requires_grad
        store_lengthscales_hist = store_lengthscales_hist and self.raw_lengthscales.requires_grad
        store_noise_hist = store_noise_hist and self.raw_noise.requires_grad
        if store_scale_hist:
            scale_hist = torch.empty(iterations+1)
        if store_lengthscales_hist:
            lengthscales_hist = torch.empty((iterations+1,self.d))
        if store_noise_hist:
            noise_hist = torch.empty(iterations+1)
        if verbose:
            _s = "%16s | %-10s | %-10s | %-10s | %-s"%("iter of %.1e"%iterations,"NMLL","noise","scale","lengthscales")
            print(" "*verbose_indent+_s)
            print(" "*verbose_indent+"~"*len(_s))
        mll_const = self.d_out*self.n*np.log(2*np.pi)
        stop_crit_best_mll = torch.inf 
        stop_crit_save_mll = torch.inf 
        stop_crit_iterations_without_improvement_mll = 0
        absytilde2 = torch.abs(self.ytilde)**2
        for i in range(iterations+1):
            mll = (absytilde2/self.lam.real).sum()+self.d_out*torch.log(torch.abs(self.lam)).sum()+mll_const
            if mll.item()<stop_crit_best_mll:
                stop_crit_best_mll = mll.item()
            if mll.item()<stop_crit_save_mll*(1-stop_crit_improvement_threshold):
                stop_crit_iterations_without_improvement_mll = 0
                stop_crit_save_mll = stop_crit_best_mll
            else:
                stop_crit_iterations_without_improvement_mll += 1
            break_condition = i==iterations or stop_crit_iterations_without_improvement_mll==stop_crit_wait_iterations
            if store_mll_hist:
                mll_hist[i] = mll.item()
            if store_scale_hist:
                scale_hist[i] = self.scale.item()
            if store_lengthscales_hist:
                lengthscales_hist[i] = self.lengthscales.detach().to(lengthscales_hist.device)
            if store_noise_hist:
                noise_hist[i] = self.noise.item()
            if verbose and (i%verbose==0 or break_condition):
                with np.printoptions(formatter={"float":lambda x: "%.2e"%x},threshold=6,edgeitems=3):
                    _s = "%16.2e | %-10.2e | %-10.2e | %-10.2e | %-s"%\
                        (i,mll.item(),self.noise.item(),self.scale.item(),str(self.lengthscales.detach().cpu().numpy()))
                print(" "*verbose_indent+_s)
            if break_condition: break
            mll.backward()
            optimizer.step()
            optimizer.zero_grad()
        data = {}
        if store_mll_hist:
            data["mll_hist"] = mll_hist
        if store_scale_hist:
            data["scale_hist"] = scale_hist
        if store_lengthscales_hist:
            data["lengthscales_hist"] = lengthscales_hist
        if store_noise_hist:
            data["noise_hist"] = noise_hist
        return data


