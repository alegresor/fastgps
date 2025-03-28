import torch 
import os 
import numpy as np 

class _XXbSeq(object):
    def __init__(self, fgp, seq):
        self.fgp = fgp
        self.seq = seq
        self.n = 0
        self.x = torch.empty((0,seq.d))
        self.xb = torch.empty((0,seq.d),dtype=self.fgp._XBDTYPE)
    def __getitem__(self, i):
        if isinstance(i,int): i = slice(None,i,None)
        if isinstance(i,torch.Tensor):
            assert i.numel()==1 and isinstance(i,torch.int64)
            i = slice(None,i.item(),None)
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
    def __init__(self, fgp, xxb_seq_first, xxb_seq_second):
        self.fgp = fgp
        self.xxb_seq_first = xxb_seq_first
        self.xxb_seq_second = xxb_seq_second
        self.k1parts = torch.empty((0,fgp.d))
        self.n = 0
    def __getitem__(self, i):
        if isinstance(i,int): i = slice(None,i,None)
        if isinstance(i,torch.Tensor):
            assert i.numel()==1 and isinstance(i,torch.int64)
            i = slice(None,i.item(),None)
        assert isinstance(i,slice)
        if i.stop>self.n:
            _,xb_next = self.xxb_seq_first[self.n:i.stop]
            _,xb0 = self.xxb_seq_second[:1]
            k1parts_next = self.fgp._kernel_parts(xb_next,xb0)
            self.k1parts = torch.vstack([self.k1parts,k1parts_next])
            self.n = i.stop
        return self.k1parts[i]

class _LamCaches(object):
    def __init__(self, fgp, l0, l1):
        self.fgp = fgp
        self.l0 = l0
        self.l1 = l1
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
    def _force_recompile(self):
        return os.environ.get("FASTGP_FORCE_RECOMPILE")=="True" and (
            self.fgp.raw_scale.requires_grad or 
            self.fgp.raw_lengthscales.requires_grad or 
            self.fgp.raw_noise.requires_grad)
    def _freeze(self, i):
        self.raw_scale_freeze_list[i] = self.fgp.raw_scale.clone()
        self.raw_lengthscales_freeze_list[i] = self.fgp.raw_lengthscales.clone()
        self.raw_noise_freeze_list[i] = self.fgp.raw_noise.clone()
    def __getitem__no_delete(self, m):
        if isinstance(m,torch.Tensor):
            assert m.numel()==1 and isinstance(m,torch.int64)
            m = m.item()
        assert isinstance(m,int)
        assert m>=self.m_min, "old lambda are not retained after updating"
        if self.m_min==-1 and m>=0:
            k1 = self.fgp._kernel_from_parts(self.fgp.get_k1parts(self.l0,self.l1,n=2**m))
            if self.l0==self.l1:
                k1[0] += self.fgp.noise
            self.lam_list = [self.fgp.ft(k1)]
            self._freeze(0)
            self.m_min = self.m_max = m
            return self.lam_list[0]
        if m==self.m_min:
            if not self._frozen_equal(0) or self._force_recompile():
                k1 = self.fgp._kernel_from_parts(self.fgp.k1parts_seq[self.l0,self.l1][:2**self.m_min])
                k1[0] += self.fgp.noise
                self.lam_list[0] = self.fgp.ft(k1)
                self._freeze(0)
            return self.lam_list[0]
        if m>self.m_max:
            self.lam_list += [torch.empty(2**mm,dtype=self.fgp._FTOUTDTYPE) for mm in range(self.m_max+1,m+1)]
            self.raw_scale_freeze_list += [torch.empty_like(self.raw_scale_freeze_list[0])]*(m-self.m_max)
            self.raw_lengthscales_freeze_list += [torch.empty_like(self.raw_lengthscales_freeze_list[0])]*(m-self.m_max)
            self.raw_noise_freeze_list += [torch.empty_like(self.raw_noise_freeze_list[0])]*(m-self.m_max)
            self.m_max = m
        midx = m-self.m_min
        if not self._frozen_equal(midx) or self._force_recompile():
            omega_m = self.fgp.get_omega(m-1)
            k1_m = self.fgp._kernel_from_parts(self.fgp.k1parts_seq[self.l0,self.l1][2**(m-1):2**m])
            lam_m = self.fgp.ft(k1_m)
            omega_lam_m = omega_m*lam_m
            lam_m_prev = self.__getitem__no_delete(m-1)
            self.lam_list[midx] = torch.hstack([lam_m_prev+omega_lam_m,lam_m_prev-omega_lam_m])/np.sqrt(2)
            if os.environ.get("FASTGP_DEBUG")=="True":
                k1_full = self.fgp._kernel_from_parts(self.fgp.k1parts_seq[self.l0,self.l1][:2**m])
                lam_full = self.fgp.ft(k1_full)
                assert torch.allclose(self.lam_list[midx],lam_full,atol=1e-7,rtol=0)
            self._freeze(midx)
        return self.lam_list[midx]
    def __getitem__(self, m):
        lam = self.__getitem__no_delete(m)
        while self.m_min<self.fgp.m[self.l0]:
            del self.lam_list[0]
            del self.raw_scale_freeze_list[0]
            del self.raw_lengthscales_freeze_list[0]
            del self.raw_noise_freeze_list[0]
            self.m_min += 1
        return lam

class _TaskCovCache(object):
    def __init__(self, fgp):
        self.fgp = fgp 
        self.num_tasks_range = torch.arange(self.fgp.num_tasks,device=self.fgp.device)
    def _frozen_equal(self):
        return (
            (self.fgp.raw_factor_task_kernel==self.raw_factor_task_kernel_freeze).all() and 
            (self.fgp.raw_noise_task_kernel==self.raw_noise_task_kernel_freeze).all())
    def _force_recompile(self):
        return os.environ.get("FASTGP_FORCE_RECOMPILE")=="True" and (
            self.fgp.raw_factor_task_kernel.requires_grad or 
            self.fgp.raw_noise_task_kernel.requires_grad)
    def _freeze(self):
        self.raw_factor_task_kernel_freeze = self.fgp.raw_factor_task_kernel.clone()
        self.raw_noise_task_kernel_freeze = self.fgp.raw_noise_task_kernel.clone()
    def __call__(self):
        if not hasattr(self,"kmat") or not self._frozen_equal() or self._force_recompile():
            self.kmat = self.fgp.factor_task_kernel@self.fgp.factor_task_kernel.T
            self.kmat[self.num_tasks_range,self.num_tasks_range] += self.fgp.noise_task_kernel
            self._freeze()
        return self.kmat

class _YtildeCache(object):
    def __init__(self, fgp, l):
        self.fgp = fgp
        self.l = l
    def __call__(self):
        if not hasattr(self,"ytilde") or self.fgp.n[self.l]<=1:
            self.ytilde = self.fgp.ft(self.fgp._y[self.l]) if self.fgp.n[self.l]>1 else self.fgp._y[self.l].clone().to(self.fgp._FTOUTDTYPE)
            self.n = self.fgp.n[self.l].item()
            return self.ytilde
        while self.n!=self.fgp.n[self.l]:
            n_double = 2*self.n
            ytilde_next = self.fgp.ft(self.fgp._y[self.l][...,self.n:n_double])
            omega_m = self.fgp.get_omega(int(np.log2(self.n)))
            omega_ytilde_next = omega_m*ytilde_next
            self.ytilde = torch.hstack([self.ytilde+omega_ytilde_next,self.ytilde-omega_ytilde_next])/np.sqrt(2)
            if os.environ.get("FASTGP_DEBUG")=="True":
                ytilde_ref = self.fgp.ft(self.fgp._y[self.l][:n_double])
                assert torch.allclose(self.ytilde,ytilde_ref,atol=1e-7,rtol=0)
            self.n = n_double
        return self.ytilde

class _InverseLogDetCache(object):
    def __init__(self, fgp, n):
        self.fgp = fgp
        self.n = n
        self.task_order = self.n.argsort(descending=True)
        self.inv_task_order = self.task_order.argsort()
    def _frozen_equal(self):
        return (
            (self.fgp.raw_scale==self.raw_scale_freeze).all() and 
            (self.fgp.raw_lengthscales==self.raw_lengthscales_freeze).all() and 
            (self.fgp.raw_noise==self.raw_noise_freeze).all() and 
            (self.fgp.raw_factor_task_kernel==self.raw_factor_task_kernel_freeze).all() and 
            (self.fgp.raw_noise_task_kernel==self.raw_noise_task_kernel_freeze).all())
    def _force_recompile(self):
        return os.environ.get("FASTGP_FORCE_RECOMPILE")=="True" and (
            self.fgp.raw_scale.requires_grad or 
            self.fgp.raw_lengthscales.requires_grad or 
            self.fgp.raw_noise.requires_grad or 
            self.fgp.raw_factor_task_kernel.requires_grad or 
            self.fgp.raw_noise_task_kernel.requires_grad)
    def _freeze(self):
        self.raw_scale_freeze = self.fgp.raw_scale.clone()
        self.raw_lengthscales_freeze = self.fgp.raw_lengthscales.clone()
        self.raw_noise_freeze = self.fgp.raw_noise.clone()
        self.raw_factor_task_kernel_freeze = self.fgp.raw_factor_task_kernel.clone()
        self.raw_noise_task_kernel_freeze = self.fgp.raw_noise_task_kernel.clone()
    def __call__(self):
        if not hasattr(self,"inv") or not self._frozen_equal() or self._force_recompile():
            n = self.n[self.task_order]
            kmat_tasks = self.fgp.gram_matrix_tasks
            lams = np.empty((self.fgp.num_tasks,self.fgp.num_tasks),dtype=object)
            for l0 in range(self.fgp.num_tasks):
                to0 = self.task_order[l0]
                for l1 in range(l0,self.fgp.num_tasks):
                    to1 = self.task_order[l1]
                    lams[l0,l1] = torch.sqrt(n[l1])*kmat_tasks[to0,to1]*self.fgp.get_lam(to0,to1,n[l0])
            self.logdet = torch.log(torch.abs(lams[0,0])).sum()
            A = (1/lams[0,0])[None,None,:]
            for l in range(1,self.fgp.num_tasks):
                if n[l]==0: break
                B = torch.cat([lams[k,l] for k in range(l)],dim=0).reshape((-1,n[l]))
                Bvec = B.reshape((A.size(1),-1))
                T = (Bvec*A).sum(1).reshape((-1,n[l]))
                M = (B.conj()*T).sum(0)
                S = lams[l,l]-M
                self.logdet += torch.log(torch.abs(S)).sum()
                P = T/S
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
                br = torch.cat([-P.conj()[None,:,:],1/S[None,None,:]],dim=1)
                A = torch.cat([ur,br],dim=0)
            if os.environ.get("FASTGP_DEBUG")=="True":
                lammats = np.empty((self.fgp.num_tasks,self.fgp.num_tasks),dtype=object)
                for l0 in range(self.fgp.num_tasks):
                    for l1 in range(l0,self.fgp.num_tasks):
                        lammats[l0,l1] = (lams[l0,l1].reshape((-1,n[l1],1))*torch.eye(n[l1])).reshape((-1,n[l1]))
                        if l0==l1: continue 
                        lammats[l1,l0] = lammats[l0,l1].conj().T
                lammat = torch.vstack([torch.hstack(lammats[i].tolist()) for i in range(self.fgp.num_tasks)])
                assert torch.allclose(torch.logdet(lammat).real,self.logdet)
                Afull = torch.vstack([torch.hstack([A[l0,l1]*torch.eye(A.size(-1)) for l1 in range(A.size(1))]) for l0 in range(A.size(0))])
                assert torch.allclose(torch.linalg.inv(lammat),Afull,rtol=1e-4)
            self._freeze()
            self.inv = A
        return self.inv,self.logdet
    def gram_matrix_solve(self, y):
        yogdim = y.ndim 
        if yogdim==1:
            y = y[:,None] 
        assert y.size(-2)==self.n.sum() 
        z = y.transpose(dim0=-2,dim1=-1)
        zs = z.split(self.n.tolist(),dim=-1)
        zst = [self.fgp.ft(zs[i]) for i in range(self.fgp.num_tasks)]
        zst,_ = self._gram_matrix_solve_tilde_to_tilde(zst)
        zs = [self.fgp.ift(zst[i]).real for i in range(self.fgp.num_tasks)]
        z = torch.cat(zs,dim=-1).transpose(dim0=-2,dim1=-1)
        if os.environ.get("FASTGP_DEBUG")=="True":
            _,logdet = self()
            kmat_tasks = self.fgp.gram_matrix_tasks
            kmat = torch.vstack([torch.hstack([kmat_tasks[ell0,ell1]*self.fgp._kernel(self.fgp.get_x(ell0,self.n[ell0])[:,None,:],self.fgp.get_x(ell1,self.n[ell1])[None,:,:]) for ell1 in range(self.fgp.num_tasks)]) for ell0 in range(self.fgp.num_tasks)])
            kmat += self.fgp.noise*torch.eye(kmat.size(0))
            assert torch.allclose(logdet,torch.logdet(kmat),rtol=1e-3)
            ztrue = torch.linalg.solve(kmat,y)
            assert torch.allclose(ztrue,z,atol=1e-3)
        if yogdim==1:
            z = z[:,0]
        return z
    def _gram_matrix_solve_tilde_to_tilde(self, zst):
        inv,logdet = self()
        zsto = [zst[o] for o in self.task_order]
        z = torch.cat(zsto,dim=-1).reshape(list(zsto[0].shape[:-1])+[1,-1,self.n[self.n>0].min()])
        z = (z*inv).sum(-2)
        z = z.reshape(list(z.shape[:-2])+[-1])
        zsto = z.split(self.n[self.task_order].tolist(),dim=-1)
        zst = [zsto[o] for o in self.inv_task_order]
        return zst,logdet

class _CoeffsCache(object):
    def __init__(self, fgp):
        self.fgp = fgp
    def _frozen_equal(self):
        return (
            (self.fgp.raw_scale==self.raw_scale_freeze).all() and 
            (self.fgp.raw_lengthscales==self.raw_lengthscales_freeze).all() and 
            (self.fgp.raw_noise==self.raw_noise_freeze).all() and 
            (self.fgp.raw_factor_task_kernel==self.raw_factor_task_kernel_freeze).all() and 
            (self.fgp.raw_noise_task_kernel==self.raw_noise_task_kernel_freeze).all())
    def _force_recompile(self):
        return os.environ.get("FASTGP_FORCE_RECOMPILE")=="True" and (
            self.fgp.raw_scale.requires_grad or 
            self.fgp.raw_lengthscales.requires_grad or 
            self.fgp.raw_noise.requires_grad or 
            self.fgp.raw_factor_task_kernel.requires_grad or 
            self.fgp.raw_noise_task_kernel.requires_grad)
    def _freeze(self):
        self.raw_scale_freeze = self.fgp.raw_scale.clone()
        self.raw_lengthscales_freeze = self.fgp.raw_lengthscales.clone()
        self.raw_noise_freeze = self.fgp.raw_noise.clone()
        self.raw_factor_task_kernel_freeze = self.fgp.raw_factor_task_kernel.clone()
        self.raw_noise_task_kernel_freeze = self.fgp.raw_noise_task_kernel.clone()
    def __call__(self):
        if not hasattr(self,"coeffs") or (self.n!=self.fgp.n).any() or not self._frozen_equal() or self._force_recompile():
            self.coeffs = self.fgp.get_inv_log_det_cache().gram_matrix_solve(torch.cat(self.fgp._y,dim=-1)[...,None])[...,0]
            self._freeze()
            self.n = self.fgp.n.clone()
        return self.coeffs 