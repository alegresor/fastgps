import torch 
import os 
import numpy as np 
import qmcpy as qp 


class DummyDiscreteDistrib(qp.discrete_distribution.AbstractDiscreteDistribution):
    def __init__(self, x):
        assert isinstance(x,np.ndarray)
        self.x = x
        assert self.x.ndim==2 
        self.n,self.d = x.shape
        super(DummyDiscreteDistrib,self).__init__(dimension=x.shape[1],replications=None,seed=None,d_limit=np.inf,n_limit=np.inf)
    def _gen_samples(self, n_min, n_max, return_binary, warn):
        assert return_binary is False
        assert n_min==0 and n_max==self.n, "trying to generate samples other than the one provided is invalid"
        return self.x[None]

class _XXbSeq(object):
    def __init__(self, fgp, seq):
        self.fgp = fgp
        self.seq = seq
        self.n = 0
        self.x = torch.empty((0,seq.d),device=self.fgp.device)
        self.xb = torch.empty((0,seq.d),dtype=self.fgp._XBDTYPE,device=self.fgp.device)
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
    def __init__(self, fgp, xxb_seq_first, xxb_seq_second, beta, kappa):
        self.fgp = fgp
        self.xxb_seq_first = xxb_seq_first
        self.xxb_seq_second = xxb_seq_second
        assert beta.ndim==2 and beta.size(-1)==self.fgp.d and kappa.ndim==2 and kappa.size(-1)==self.fgp.d
        assert beta.shape==kappa.shape
        self.beta = beta 
        self.kappa = kappa
        self.k1parts = torch.empty((0,len(self.kappa),self.fgp.d),device=self.fgp.device)
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
            k1parts_next = self.fgp.kernel.base_kernel.get_per_dim_components(xb_next,xb0,self.beta,self.kappa)
            self.k1parts = torch.cat([self.k1parts,k1parts_next],dim=0)
            self.n = i.stop
        return self.k1parts[i]

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
        self.raw_noise_freeze_list = [None]
        self._freeze(0)
        self.lam_list = [torch.empty(0,dtype=self.fgp._FTOUTDTYPE,device=self.fgp.device)]
    def _frozen_equal(self, i):
        return (
            (self.fgp.kernel.base_kernel.raw_scale==self.raw_scale_freeze_list[i]).all() and 
            (self.fgp.kernel.base_kernel.raw_lengthscales==self.raw_lengthscales_freeze_list[i]).all() and 
            (self.fgp.raw_noise==self.raw_noise_freeze_list[i]).all())
    def _force_recompile(self):
        return os.environ.get("FASTGP_FORCE_RECOMPILE")=="True" and (
            self.fgp.kernel.base_kernel.raw_scale.requires_grad or 
            self.fgp.kernel.base_kernel.raw_lengthscales.requires_grad or 
            self.fgp.raw_noise.requires_grad)
    def _freeze(self, i):
        self.raw_scale_freeze_list[i] = self.fgp.kernel.base_kernel.raw_scale.clone()
        self.raw_lengthscales_freeze_list[i] = self.fgp.kernel.base_kernel.raw_lengthscales.clone()
        self.raw_noise_freeze_list[i] = self.fgp.raw_noise.clone()
    def __getitem__no_delete(self, m):
        if isinstance(m,torch.Tensor):
            assert m.numel()==1 and isinstance(m,torch.int64)
            m = m.item()
        assert isinstance(m,int)
        assert m>=self.m_min, "old lambda are not retained after updating"
        if self.m_min==-1 and m>=0:
            batch_params = self.fgp.kernel.base_kernel.get_batch_params(1)
            k1 = self.fgp.kernel.base_kernel.combine_per_dim_components(self.fgp.get_k1parts(self.l0,self.l1,n=2**m),self.beta0,self.beta1,self.c,batch_params)
            self.lam_list = [self.fgp.ft(k1)]
            self._freeze(0)
            self.m_min = self.m_max = m
            return self.lam_list[0]
        if m==self.m_min:
            if not self._frozen_equal(0) or self._force_recompile():
                batch_params = self.fgp.kernel.base_kernel.get_batch_params(1)
                k1 = self.fgp.kernel.base_kernel.combine_per_dim_components(self.fgp.k1parts_seq[self.l0,self.l1][:2**self.m_min],self.beta0,self.beta1,self.c,batch_params)
                self.lam_list[0] = self.fgp.ft(k1)
                self._freeze(0)
            return self.lam_list[0]
        if m>self.m_max:
            self.lam_list += [torch.empty(2**mm,dtype=self.fgp._FTOUTDTYPE,device=self.fgp.device) for mm in range(self.m_max+1,m+1)]
            self.raw_scale_freeze_list += [torch.empty_like(self.raw_scale_freeze_list[0])]*(m-self.m_max)
            self.raw_lengthscales_freeze_list += [torch.empty_like(self.raw_lengthscales_freeze_list[0])]*(m-self.m_max)
            self.raw_noise_freeze_list += [torch.empty_like(self.raw_noise_freeze_list[0])]*(m-self.m_max)
            self.m_max = m
        midx = m-self.m_min
        if not self._frozen_equal(midx) or self._force_recompile():
            omega_m = self.fgp.omega(m-1).to(self.fgp.device)
            batch_params = self.fgp.kernel.base_kernel.get_batch_params(1)
            k1_m = self.fgp.kernel.base_kernel.combine_per_dim_components(self.fgp.k1parts_seq[self.l0,self.l1][2**(m-1):2**m],self.beta0,self.beta1,self.c,batch_params)
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
            del self.raw_noise_freeze_list[0]
            self.m_min += 1
        return lam

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
            omega_m = self.fgp.omega(int(np.log2(self.n))).to(self.fgp.device)
            omega_ytilde_next = omega_m*ytilde_next
            self.ytilde = torch.cat([self.ytilde+omega_ytilde_next,self.ytilde-omega_ytilde_next],-1)/np.sqrt(2)
            if os.environ.get("FASTGP_DEBUG")=="True":
                ytilde_ref = self.fgp.ft(self.fgp._y[self.l][:n_double])
                assert torch.allclose(self.ytilde,ytilde_ref,atol=1e-7,rtol=0)
            self.n = n_double
        return self.ytilde

class _AbstractCache(object):
    def _frozen_equal(self):
        return not any((self.state_dict[pname]!=pval).any() for pname,pval in self.fgp.named_parameters())
    def _force_recompile(self):
        return os.environ.get("FASTGP_FORCE_RECOMPILE")=="True" and any(pval.requires_grad for pname,pval in self.fgp.named_parameters())
    def _freeze(self):
        self.state_dict = {pname:pval.data.detach().clone() for pname,pval in self.fgp.state_dict().items()}

class _StandardInverseLogDetCache(_AbstractCache):
    def __init__(self, fgp, n):
        self.fgp = fgp
        self.n = n
    def __call__(self):
        if not hasattr(self,"thetainv") or not self._frozen_equal() or self._force_recompile():
            kmat_tasks = self.fgp.kernel.taskmat
            kmat_lower_tri = [[self.fgp.kernel.base_kernel(self.fgp.get_x(l0,self.n[l0])[:,None,:],self.fgp.get_x(l1,self.n[l1])[None,:,:],*self.fgp.derivatives_cross[l0][l1],self.fgp.derivatives_coeffs_cross[l0][l1]) for l1 in range(l0+1)] for l0 in range(self.fgp.num_tasks)]
            if self.fgp.adaptive_nugget:
                assert self.fgp.noise.size(-1)==1
                n0range = torch.arange(self.n[0],device=self.fgp.device)
                tr00 = kmat_lower_tri[0][0][...,n0range,n0range].sum(-1)
            spd_factor = 1.
            while True:
                for l in range(self.fgp.num_tasks):
                    if self.fgp.adaptive_nugget:
                        nlrange = torch.arange(self.n[l],device=self.fgp.device)
                        trll = kmat_lower_tri[l][l][...,nlrange,nlrange].sum(-1)
                        noise_l = self.fgp.noise[...,0]*trll/tr00
                    else:
                        noise_l = self.fgp.noise[...,0]
                    kmat_lower_tri[l][l] = kmat_lower_tri[l][l]+spd_factor*noise_l[...,None,None]*torch.eye(self.n[l],device=self.fgp.device)
                kmat_full = [[kmat_tasks[...,l0,l1,None,None]*(kmat_lower_tri[l0][l1] if l1<=l0 else kmat_lower_tri[l1][l0].transpose(dim0=-2,dim1=-1)) for l1 in range(self.fgp.num_tasks)] for l0 in range(self.fgp.num_tasks)]
                kmat = torch.cat([torch.cat(kmat_full[l0],dim=-1) for l0 in range(self.fgp.num_tasks)],dim=-2)
                try:
                    l_chol = torch.linalg.cholesky(kmat,upper=False)
                    break
                except torch._C._LinAlgError as e:
                    expected_str = "linalg.cholesky: The factorization could not be completed because the input is not positive-definite"
                    if str(e)[:len(expected_str)]!=expected_str: raise
                    spd_factor *= 2#raise Exception("Cholesky factor not SPD, try increasing noise")
            nfrange = torch.arange(self.n.sum(),device=self.fgp.device)
            self.logdet = 2*torch.log(l_chol[...,nfrange,nfrange]).sum(-1)
            try:
                self.thetainv = torch.cholesky_inverse(l_chol,upper=False)
            except NotImplementedError as e:
                expected_str = "The operator 'aten::cholesky_inverse' is not currently implemented for the MPS device."
                if str(e)[:len(expected_str)]!=expected_str: raise
                eye = torch.eye(l_chol.size(-1),device=l_chol.device)
                l_chol_inv = torch.linalg.solve_triangular(l_chol,eye,upper=False)
                self.thetainv = torch.einsum("...ji,...jk->...ik",l_chol_inv,l_chol_inv)
            self._freeze()
        return self.thetainv,self.logdet
    def gram_matrix_solve(self, y):
        assert y.size(-1)==self.n.sum()
        thetainv,logdet = self()
        v = torch.einsum("...ij,...j->...i",thetainv,y)
        return v
    def get_norm_term_logdet_term(self):
        y = torch.cat(self.fgp._y,dim=-1)
        thetainv,logdet = self()
        v = torch.einsum("...ij,...j->...i",thetainv,y)
        norm_term = (y*v).sum(-1,keepdim=True)
        return norm_term,logdet[...,None]
    def get_gcv_numer_denom(self):
        thetainv,logdet = self()
        y = torch.cat(self.fgp._y,dim=-1)
        v = torch.einsum("...ij,...j->...i",thetainv,y)
        numer = (v**2).sum(-1,keepdim=True)
        tr_k_inv = torch.einsum("...ii",thetainv)[...,None]
        denom = (tr_k_inv/thetainv.size(-1))**2
        return numer,denom
    def get_inv_diag(self):
        # right now this costs costs O(n^3), maybe it can be done faster?
        thetainv,logdet = self()
        nrange = torch.arange(thetainv.size(-1),device=self.fgp.device)
        inv_diag = thetainv[...,nrange,nrange]
        return inv_diag
    
class _FastInverseLogDetCache(_AbstractCache):
    def __init__(self, fgp, n):
        self.fgp = fgp
        self.n = n
        self.task_order = self.n.argsort(descending=True)
        self.inv_task_order = self.task_order.argsort()
    def __call__(self):
        if not hasattr(self,"inv") or not self._frozen_equal() or self._force_recompile():
            n = self.n[self.task_order]
            kmat_tasks = self.fgp.kernel.taskmat
            lams = np.empty((self.fgp.num_tasks,self.fgp.num_tasks),dtype=object)
            for l0 in range(self.fgp.num_tasks):
                to0 = self.task_order[l0]
                for l1 in range(l0,self.fgp.num_tasks):
                    to1 = self.task_order[l1]
                    lam = self.fgp.get_lam(to0,to1,n[l0]) if to0<=to1 else self.fgp.get_lam(to1,to0,n[l0]).conj()
                    lams[l0,l1] = torch.sqrt(n[l1])*lam
            if self.fgp.adaptive_nugget:
                tr00 = lams[self.inv_task_order[0],self.inv_task_order[0]].sum(-1)
                for l in range(self.fgp.num_tasks):
                    trll = lams[l,l].sum(-1)
                    lams[l,l] = lams[l,l]+self.fgp.noise*(trll/tr00).abs()
            else:
                for l in range(self.fgp.num_tasks):
                    lams[l,l] = lams[l,l]+self.fgp.noise
            for l0 in range(self.fgp.num_tasks):
                to0 = self.task_order[l0]
                for l1 in range(l0,self.fgp.num_tasks):
                    to1 = self.task_order[l1]
                    lams[l0,l1] = lams[l0,l1]*kmat_tasks[...,to0,to1,None]
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
            self._freeze()
            self.inv = A
        return self.inv,self.logdet
    def gram_matrix_solve(self, y):
        assert y.size(-1)==self.n.sum() 
        ys = y.split(self.n.tolist(),dim=-1)
        yst = [self.fgp.ft(ys[i]) for i in range(self.fgp.num_tasks)]
        yst,_,_ = self._gram_matrix_solve_tilde_to_tilde(yst)
        ys = [self.fgp.ift(yst[i]).real for i in range(self.fgp.num_tasks)]
        y = torch.cat(ys,dim=-1)
        if os.environ.get("FASTGP_DEBUG")=="True":
            _,logdet = self()
            kmat_tasks = self.fgp.gram_matrix_tasks
            kmat = torch.vstack([torch.hstack([kmat_tasks[ell0,ell1]*self.fgp._kernel(self.fgp.get_x(ell0,self.n[ell0])[:,None,:],self.fgp.get_x(ell1,self.n[ell1])[None,:,:]) for ell1 in range(self.fgp.num_tasks)]) for ell0 in range(self.fgp.num_tasks)])
            kmat += self.fgp.noise*torch.eye(kmat.size(0))
            assert torch.allclose(logdet,torch.logdet(kmat),rtol=1e-3)
            ytrue = torch.linalg.solve(kmat,y)
            assert torch.allclose(ytrue,y,atol=1e-3)
        return y
    def _gram_matrix_solve_tilde_to_tilde(self, zst):
        inv,logdet = self()
        zsto = [zst[o] for o in self.task_order]
        z = torch.cat(zsto,dim=-1)
        z = z.reshape(list(zsto[0].shape[:-1])+[1,-1,self.n[self.n>0].min()])
        z = (z*inv).sum(-2)
        z = z.reshape(list(z.shape[:-2])+[-1])
        zsto = z.split(self.n[self.task_order].tolist(),dim=-1)
        zst = [zsto[o] for o in self.inv_task_order]
        return zst,inv,logdet
    def get_norm_term_logdet_term(self):
        ytildes = [self.fgp.get_ytilde(i) for i in range(self.fgp.num_tasks)]
        ytildescat = torch.cat(ytildes,dim=-1)
        ztildes,inv,logdet = self._gram_matrix_solve_tilde_to_tilde(ytildes)
        ztildescat = torch.cat(ztildes,dim=-1)
        norm_term = (ytildescat.conj()*ztildescat).real.sum(-1,keepdim=True)
        return norm_term,logdet[...,None]
    def get_gcv_numer_denom(self):
        ytildes = [self.fgp.get_ytilde(i) for i in range(self.fgp.num_tasks)]
        ztildes,inv,logdet = self._gram_matrix_solve_tilde_to_tilde(ytildes)
        ztildescat = torch.cat(ztildes,dim=-1)
        numer = (ztildescat.conj()*ztildescat).real.sum(-1,keepdim=True)
        n = inv.size(-2)
        nrange = torch.arange(n,device=self.fgp.device)
        tr_k_inv = inv[...,nrange,nrange,:].real.sum(-1).sum(-1,keepdim=True)
        denom = ((tr_k_inv/self.n.sum())**2).real
        return numer,denom
    def get_inv_diag(self):
        if self.fgp.num_tasks==1:
            lam = self.fgp.get_lam(0,0)
            rootn = np.sqrt(lam.size(-1))
            inv_diag = (1/(lam*rootn)).mean(-1,keepdim=True)
        else:
            # there should be a more efficient way than this current O(n^2 log n) approach 
            inv,logdet = self()
            nsum = self.fgp.n.sum()
            eye = torch.eye(nsum,device=self.fgp.device).reshape([nsum]+[1]*(inv.ndim-3)+[nsum])
            kmatinv = self.gram_matrix_solve(eye).permute([1+i for i in range(inv.ndim-3)]+[0,-1])
            nrange = torch.arange(kmatinv.size(-1),device=self.fgp.device)
            inv_diag = kmatinv[...,nrange,nrange]
        return inv_diag

class _CoeffsCache(_AbstractCache):
    def __init__(self, fgp):
        self.fgp = fgp
    def __call__(self):
        if not hasattr(self,"coeffs") or (self.n!=self.fgp.n).any() or not self._frozen_equal() or self._force_recompile():
            inv_log_det_cache = self.fgp.get_inv_log_det_cache()
            self.coeffs = inv_log_det_cache.gram_matrix_solve(torch.cat(self.fgp._y,dim=-1))
            self._freeze()
            self.n = self.fgp.n.clone()
        return self.coeffs  