import torch 
import os 
import numpy as np 
import qmcpy as qp

class DummyDiscreteDistrib(qp.discrete_distribution.abstract_discrete_distribution.AbstractDiscreteDistribution):
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
        self.seq = seq
        self.n = 0
        self.xb = torch.empty((0,seq.d),dtype=fgp._XBDTYPE,device=fgp.device)
    def getitem_xb(self, fgp, i):
        if isinstance(i,int): i = slice(None,i,None)
        if isinstance(i,torch.Tensor):
            assert i.numel()==1 and isinstance(i,torch.int64)
            i = slice(None,i.item(),None)
        assert isinstance(i,slice)
        if i.stop>self.n:
            xb_next = fgp._sample(self.seq,self.n,i.stop)
            self.xb = torch.vstack([self.xb,xb_next]).contiguous()
            self.n = i.stop
            del xb_next
        return self.xb[i]
    def getitem_x(self, fgp, i):
        xb = self.getitem_xb(fgp,i)
        x = fgp._convert_xb_to_x(xb)
        if fgp.ptransform=="NONE":
            pass
        elif fgp.ptransform=="BAKER":
            x = 1-2*torch.abs(x-1/2)
        else:
            raise Exception("invalid ptransform = %s"%fgp.ptransform)
        return x

class _K1PartsSeq(object):
    def __init__(self, fgp, l0, l1, beta, kappa):
        self.l0,self.l1 = l0,l1
        assert beta.ndim==2 and beta.size(-1)==fgp.d and kappa.ndim==2 and kappa.size(-1)==fgp.d
        assert beta.shape==kappa.shape
        self.beta = beta 
        self.kappa = kappa
        self.n = 0
    def getitem(self, fgp, i):
        if isinstance(i,int): i = slice(None,i,None)
        if isinstance(i,torch.Tensor):
            assert i.numel()==1 and isinstance(i,torch.int64)
            i = slice(None,i.item(),None)
        assert isinstance(i,slice)
        if i.stop>self.n:
            xb_next = fgp.xxb_seqs[self.l0].getitem_xb(fgp,slice(self.n,i.stop))
            xb0 = fgp.xxb_seqs[self.l1].getitem_xb(fgp,slice(0,1))
            k1parts_next = fgp.kernel.base_kernel.get_per_dim_components(xb_next,xb0,self.beta,self.kappa)
            if not hasattr(self,"k1parts"):
                self.k1parts = k1parts_next 
            else:
                self.k1parts = torch.cat([self.k1parts,k1parts_next],dim=0)
            self.n = i.stop
        return self.k1parts[i]

class _YtildeCache(object):
    def __init__(self, fgp, l):
        self.l = l
    def __call__(self, fgp):
        if not hasattr(self,"ytilde") or fgp.n[self.l]<=1:
            self.ytilde = fgp.ft(fgp._y[self.l]) if fgp.n[self.l]>1 else fgp._y[self.l].clone().to(fgp._FTOUTDTYPE)
            self.n = fgp.n[self.l].item()
            return self.ytilde
        while self.n!=fgp.n[self.l]:
            n_double = 2*self.n
            ytilde_next = fgp.ft(fgp._y[self.l][...,self.n:n_double])
            omega_m = fgp.omega(int(np.log2(self.n))).to(fgp.device)
            omega_ytilde_next = omega_m*ytilde_next
            self.ytilde = torch.cat([self.ytilde+omega_ytilde_next,self.ytilde-omega_ytilde_next],-1)/np.sqrt(2)
            if os.environ.get("FASTGP_DEBUG")=="True":
                ytilde_ref = fgp.ft(fgp._y[self.l][:n_double])
                assert torch.allclose(self.ytilde,ytilde_ref,atol=1e-7,rtol=0)
            self.n = n_double
        return self.ytilde

def _freeze(fgp):
    return {pname:pval.data.detach().clone() for pname,pval in fgp.state_dict().items()}

def _frozen_equal(fgp, state_dict):
    return not any((state_dict[pname]!=pval).any() for pname,pval in fgp.named_parameters())

def _force_recompile(fgp):
    return os.environ.get("FASTGP_FORCE_RECOMPILE")=="True" and any(pval.requires_grad for pname,pval in fgp.named_parameters())

class _LamCaches:
    def __init__(self, fgp, l0, l1, beta0, beta1, c):
        self.l0 = l0
        self.l1 = l1
        assert c.ndim==1
        assert beta0.shape==(len(c),fgp.d) and beta1.shape==(len(c),fgp.d)
        self.c = c 
        self.beta0 = beta0 
        self.beta1 = beta1
        self.m_min,self.m_max = -1,-1
        self.raw_scale_freeze_list = [None]
        self.raw_lengthscales_freeze_list = [None]
        self.raw_alpha_freeze_list = [None]
        self.raw_noise_freeze_list = [None]
        self._freeze(fgp, 0)
        self.lam_list = [torch.empty(0,dtype=fgp._FTOUTDTYPE,device=fgp.device)]
    def _frozen_equal(self, fgp, i):
        return (
            (fgp.kernel.base_kernel.raw_scale==self.raw_scale_freeze_list[i]).all() and 
            (fgp.kernel.base_kernel.raw_lengthscales==self.raw_lengthscales_freeze_list[i]).all() and 
            (fgp.kernel.base_kernel.raw_alpha==self.raw_alpha_freeze_list[i]).all() and 
            (fgp.raw_noise==self.raw_noise_freeze_list[i]).all())
    def _force_recompile(self, fgp):
        return os.environ.get("FASTGP_FORCE_RECOMPILE")=="True" and (
            fgp.kernel.base_kernel.raw_scale.requires_grad or 
            fgp.kernel.base_kernel.raw_lengthscales.requires_grad or 
            fgp.kernel.base_kernel.raw_alpha.requires_grad or 
            fgp.raw_noise.requires_grad)
    def _freeze(self, fgp, i):
        self.raw_scale_freeze_list[i] = fgp.kernel.base_kernel.raw_scale.clone()
        self.raw_lengthscales_freeze_list[i] = fgp.kernel.base_kernel.raw_lengthscales.clone()
        self.raw_alpha_freeze_list[i] = fgp.kernel.base_kernel.raw_alpha.clone()
        self.raw_noise_freeze_list[i] = fgp.raw_noise.clone()
    def __getitem__no_delete(self, fgp, m):
        if isinstance(m,torch.Tensor):
            assert m.numel()==1 and isinstance(m,torch.int64)
            m = m.item()
        assert isinstance(m,int)
        assert m>=self.m_min, "old lambda are not retained after updating"
        if self.m_min==-1 and m>=0:
            batch_params = fgp.kernel.base_kernel.get_batch_params(1)
            _,k1m1 = fgp.kernel.base_kernel.combine_per_dim_components_raw_m1(fgp.get_k1parts(self.l0,self.l1,n=2**m),self.beta0,self.beta1,self.c,batch_params,fgp.stable)
            self.lam_list = [fgp.ft(k1m1)]
            self._freeze(fgp,0)
            self.m_min = self.m_max = m
            return self.lam_list[0]
        if m==self.m_min:
            if not self._frozen_equal(fgp,0) or self._force_recompile(fgp):
                batch_params = fgp.kernel.base_kernel.get_batch_params(1)
                _,k1m1 = fgp.kernel.base_kernel.combine_per_dim_components_raw_m1(fgp.get_k1parts(self.l0,self.l1,n=2**self.m_min),self.beta0,self.beta1,self.c,batch_params,fgp.stable)
                self.lam_list[0] = fgp.ft(k1m1)
                self._freeze(fgp,0)
            return self.lam_list[0]
        if m>self.m_max:
            self.lam_list += [torch.empty(2**mm,dtype=fgp._FTOUTDTYPE,device=fgp.device) for mm in range(self.m_max+1,m+1)]
            self.raw_scale_freeze_list += [torch.empty_like(self.raw_scale_freeze_list[0])]*(m-self.m_max)
            self.raw_lengthscales_freeze_list += [torch.empty_like(self.raw_lengthscales_freeze_list[0])]*(m-self.m_max)
            self.raw_alpha_freeze_list += [torch.empty_like(self.raw_alpha_freeze_list[0])]*(m-self.m_max)
            self.raw_noise_freeze_list += [torch.empty_like(self.raw_noise_freeze_list[0])]*(m-self.m_max)
            self.m_max = m
        midx = m-self.m_min
        if not self._frozen_equal(fgp,midx) or self._force_recompile(fgp):
            omega_m = fgp.omega(m-1).to(fgp.device)
            batch_params = fgp.kernel.base_kernel.get_batch_params(1)
            _,k1m1_m = fgp.kernel.base_kernel.combine_per_dim_components_raw_m1(fgp.k1parts_seq[self.l0,self.l1].getitem(fgp,slice(2**(m-1),2**m)),self.beta0,self.beta1,self.c,batch_params,fgp.stable)
            lam_m = fgp.ft(k1m1_m)
            omega_lam_m = omega_m*lam_m
            lam_m_prev = self.__getitem__no_delete(fgp,m-1)
            self.lam_list[midx] = torch.cat([lam_m_prev+omega_lam_m,lam_m_prev-omega_lam_m],-1)/np.sqrt(2)
            self._freeze(fgp,midx)
        return self.lam_list[midx]
    def getitem_m1(self, fgp, m):
        lam = self.__getitem__no_delete(fgp,m)
        while self.m_min<max(fgp.m[self.l0],fgp.m[self.l1]):
            del self.lam_list[0]
            del self.raw_scale_freeze_list[0]
            del self.raw_lengthscales_freeze_list[0]
            del self.raw_alpha_freeze_list[0]
            del self.raw_noise_freeze_list[0]
            self.m_min += 1
        return lam
    def getitem(self, fgp, m):
        lam = self.getitem_m1(fgp, m)
        e = 1*(torch.arange(lam.size(-1),device=lam.device)==0)
        return lam+self.c.sum()*fgp.kernel.base_kernel.scale*np.sqrt(2**m)*e
    
class _FastInverseLogDetCache:
    def __init__(self, n):
        self.n = n.clone()
        self.task_order = self.n.argsort(descending=True)
        self.inv_task_order = self.task_order.argsort()
    def __call__(self, fgp):
        if not hasattr(self,"inv") or not _frozen_equal(fgp,self.state_dict) or _force_recompile(fgp):
            n = self.n[self.task_order]
            kmat_tasks = fgp.kernel.taskmat
            lams = np.empty((fgp.num_tasks,fgp.num_tasks),dtype=object)
            for l0 in range(fgp.num_tasks):
                to0 = self.task_order[l0]
                for l1 in range(l0,fgp.num_tasks):
                    to1 = self.task_order[l1]
                    lam = fgp.get_lam(to0,to1,n[l0]) if to0<=to1 else fgp.get_lam(to1,to0,n[l0]).conj()
                    lams[l0,l1] = kmat_tasks[...,to0,to1,None]*torch.sqrt(n[l1])*lam
            if fgp.adaptive_nugget:
                tr00 = lams[self.inv_task_order[0],self.inv_task_order[0]].sum(-1)
                for l in range(fgp.num_tasks):
                    trll = lams[l,l].sum(-1)
                    lams[l,l] = lams[l,l]+fgp.noise*(trll/tr00).abs()
            else:
                for l in range(fgp.num_tasks):
                    lams[l,l] = lams[l,l]+fgp.noise
            self.logdet = torch.log(torch.abs(lams[0,0])).sum(-1)
            A = (1/lams[0,0])[...,None,None,:]
            for l in range(1,fgp.num_tasks):
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
                lammats = np.empty((fgp.num_tasks,fgp.num_tasks),dtype=object)
                for l0 in range(fgp.num_tasks):
                    for l1 in range(l0,fgp.num_tasks):
                        lammats[l0,l1] = (lams[l0,l1].reshape((-1,n[l1],1))*torch.eye(n[l1])).reshape((-1,n[l1]))
                        if l0==l1: continue 
                        lammats[l1,l0] = lammats[l0,l1].conj().transpose(dim0=-2,dim1=-1)
                lammat = torch.vstack([torch.hstack(lammats[i].tolist()) for i in range(fgp.num_tasks)])
                assert torch.allclose(torch.logdet(lammat).real,self.logdet)
                Afull = torch.vstack([torch.hstack([A[l0,l1]*torch.eye(A.size(-1)) for l1 in range(A.size(1))]) for l0 in range(A.size(0))])
                assert torch.allclose(torch.linalg.inv(lammat),Afull,rtol=1e-4)
            self.state_dict = _freeze(fgp)
            self.inv = A
        return self.inv,self.logdet
    def gram_matrix_solve(self, fgp, y):
        inv,logdet = self(fgp)
        return self._gram_matrix_solve(fgp, y,inv)
    def _gram_matrix_solve(self, fgp, y, inv):
        assert y.size(-1)==self.n.sum() 
        ys = y.split(self.n.tolist(),dim=-1)
        yst = [fgp.ft(ys[i]) for i in range(fgp.num_tasks)]
        yst = self._gram_matrix_solve_tilde_to_tilde(yst,inv)
        ys = [fgp.ift(yst[i]).real for i in range(fgp.num_tasks)]
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
    def mll_loss(self, fgp, update_prior_mean):
        inv,logdet = self(fgp)
        ytildes = [fgp.get_ytilde(i) for i in range(fgp.num_tasks)]
        sqrtn = torch.sqrt(fgp.n)
        if update_prior_mean:
            if fgp.solo_task:
                fgp.prior_mean = fgp._y[0].mean(dim=-1,keepdim=True)
            else:
                rhs = self._gram_matrix_solve_tilde_to_tilde(ytildes,inv)
                rhs = torch.cat([rhs_i[...,0,None] for rhs_i in rhs],dim=-1).real
                to = self.task_order
                ito = self.inv_task_order
                nord = fgp.n[to]
                mvec = torch.hstack([torch.zeros(1,device=fgp.device),(nord/nord[-1]).cumsum(0)]).to(int)[:-1]
                tasksums = sqrtn*inv[...,0][...,mvec,:][...,:,mvec][...,ito,:][...,:,ito].real
                fgp.prior_mean = torch.linalg.solve_ex(tasksums,rhs[...,None])[0][...,0]
        deltatildescat = torch.cat(ytildes,dim=-1)
        deltatildescat[...,fgp.n_cumsum] = deltatildescat[...,fgp.n_cumsum]-sqrtn*fgp.prior_mean
        ztildes = self._gram_matrix_solve_tilde_to_tilde(deltatildescat.split(self.n.tolist(),dim=-1),inv)
        ztildescat = torch.cat(ztildes,dim=-1)
        norm_term = (deltatildescat.conj()*ztildescat).real.sum(-1,keepdim=True)
        logdet = logdet[...,None]
        d_out = norm_term.numel()
        term1 = norm_term.sum()
        mll_const = d_out*fgp.n.sum()*np.log(2*np.pi)
        term2 = d_out/torch.tensor(logdet.shape).prod()*logdet.sum()
        mll_loss = 1/2*(term1+term2+mll_const)
        return mll_loss
    def gcv_loss(self, fgp, update_prior_mean):
        inv,logdet = self(fgp)
        ytildes = [fgp.get_ytilde(i) for i in range(fgp.num_tasks)]
        sqrtn = torch.sqrt(fgp.n)
        if update_prior_mean:
            if fgp.solo_task: # stable computation
                fgp.prior_mean = fgp._y[0].mean(dim=-1,keepdim=True)
            else:
                rhs = self._gram_matrix_solve_tilde_to_tilde(ytildes,inv)
                rhs = self._gram_matrix_solve_tilde_to_tilde(rhs,inv)
                rhs = torch.cat([rhs_i[...,0,None] for rhs_i in rhs],dim=-1).real
                to = self.task_order
                ito = self.inv_task_order
                nord = fgp.n[to]
                mvec = torch.hstack([torch.zeros(1,device=fgp.device),(nord/nord[-1]).cumsum(0)]).to(int)[:-1]
                inv2 = torch.einsum("...ij,...jk->...ik",inv[...,0],inv[...,0])
                tasksums = sqrtn*inv2[...,mvec,:][...,:,mvec][...,ito,:][...,:,ito].real            
                fgp.prior_mean = torch.linalg.solve_ex(tasksums,rhs[...,None])[0][...,0]
        deltatildescat = torch.cat(ytildes,dim=-1)
        deltatildescat[...,fgp.n_cumsum] = deltatildescat[...,fgp.n_cumsum]-torch.sqrt(fgp.n)*fgp.prior_mean
        ztildes = self._gram_matrix_solve_tilde_to_tilde(deltatildescat.split(self.n.tolist(),dim=-1),inv)
        ztildescat = torch.cat(ztildes,dim=-1)
        numer = (ztildescat.conj()*ztildescat).real.sum(-1,keepdim=True)
        n = inv.size(-2)
        nrange = torch.arange(n,device=fgp.device)
        tr_k_inv = inv[...,nrange,nrange,:].real.sum(-1).sum(-1,keepdim=True)
        denom = ((tr_k_inv/self.n.sum())**2).real
        gcv_loss = (numer/denom).sum()
        return gcv_loss
    def cv_loss(self, fgp, cv_weights, update_prior_mean):
        assert not update_prior_mean, "fast GP updates to prior mean with CV loss not yet worked out"
        if fgp.num_tasks==1:
            inv,logdet = self(fgp)
            coeffs = self._gram_matrix_solve(fgp,torch.cat([fgp._y[i]-fgp.prior_mean[...,i,None] for i in range(fgp.num_tasks)],dim=-1),inv)
            inv_diag = inv[0,0].sum()/fgp.n
            squared_sums = ((coeffs/inv_diag)**2*cv_weights).sum(-1,keepdim=True)
            cv_loss = squared_sums.sum().real
        else:
            assert False, "fast multitask GPs do not yet support efficient CV loss computation"
        return cv_loss

class _StandardInverseLogDetCache:
    def __init__(self, n):
        self.n = n
    def __call__(self, fgp):
        if not hasattr(self,"thetainv") or not _frozen_equal(fgp,self.state_dict) or _force_recompile(fgp):
            kmat_tasks = fgp.kernel.taskmat
            kmat_lower_tri = [[kmat_tasks[...,l0,l1,None,None]*fgp.kernel.base_kernel(fgp.get_xb(l0,self.n[l0])[:,None,:],fgp.get_xb(l1,self.n[l1])[None,:,:],*fgp.derivatives_cross[l0][l1],fgp.derivatives_coeffs_cross[l0][l1]) for l1 in range(l0+1)] for l0 in range(fgp.num_tasks)]
            if fgp.adaptive_nugget:
                assert fgp.noise.size(-1)==1
                n0range = torch.arange(self.n[0],device=fgp.device)
                tr00 = kmat_lower_tri[0][0][...,n0range,n0range].sum(-1)
            spd_factor = 1.
            while True:
                noise_ls = [None]*fgp.num_tasks
                for l in range(fgp.num_tasks):
                    if fgp.adaptive_nugget:
                        nlrange = torch.arange(self.n[l],device=fgp.device)
                        trll = kmat_lower_tri[l][l][...,nlrange,nlrange].sum(-1)
                        noise_ls[l] = fgp.noise[...,0]*trll/tr00
                    else:
                        noise_ls[l] = fgp.noise[...,0]
                kmat_full = [[(kmat_lower_tri[l0][l1] if l1<=l0 else kmat_lower_tri[l1][l0].transpose(dim0=-2,dim1=-1))+(0 if l1!=l0 else (spd_factor*noise_ls[l0][...,None,None]*torch.eye(self.n[l0],device=fgp.device))) for l1 in range(fgp.num_tasks)] for l0 in range(fgp.num_tasks)]
                kmat = torch.cat([torch.cat(kmat_full[l0],dim=-1) for l0 in range(fgp.num_tasks)],dim=-2)
                try:
                    l_chol = torch.linalg.cholesky(kmat,upper=False)
                    break
                except torch._C._LinAlgError as e:
                    expected_str = "linalg.cholesky: The factorization could not be completed because the input is not positive-definite"
                    if str(e)[:len(expected_str)]!=expected_str: raise
                    spd_factor *= 2#raise Exception("Cholesky factor not SPD, try increasing noise")
            nfrange = torch.arange(self.n.sum(),device=fgp.device)
            self.logdet = 2*torch.log(l_chol[...,nfrange,nfrange]).sum(-1)
            try:
                self.thetainv = torch.cholesky_inverse(l_chol,upper=False)
            except NotImplementedError as e:
                expected_str = "The operator 'aten::cholesky_inverse' is not currently implemented for the MPS device."
                if str(e)[:len(expected_str)]!=expected_str: raise
                eye = torch.eye(l_chol.size(-1),device=l_chol.device)
                l_chol_inv = torch.linalg.solve_triangular(l_chol,eye,upper=False)
                self.thetainv = torch.einsum("...ji,...jk->...ik",l_chol_inv,l_chol_inv)
            self.state_dict = _freeze(fgp)
        return self.thetainv,self.logdet
    def gram_matrix_solve(self, fgp, y):
        assert y.size(-1)==self.n.sum()
        thetainv,logdet = self(fgp)
        v = torch.einsum("...ij,...j->...i",thetainv,y)
        return v
    def mll_loss(self, fgp, update_prior_mean):
        thetainv,logdet = self(fgp)
        y = torch.cat(fgp._y,dim=-1) 
        if update_prior_mean:
            rhs = torch.einsum("...ij,...j->...i",thetainv,y).split(self.n.tolist(),dim=-1)
            rhs = torch.cat([rhs_i.sum(-1,keepdim=True) for rhs_i in rhs],dim=-1)
            thetainv_split = [thetinv_i.split(self.n.tolist(),dim=-1) for thetinv_i in thetainv.split(self.n.tolist(),dim=-2)]
            tasksums = torch.cat([torch.cat([thetainv_split[i][j].sum((-2,-1),keepdim=True) for j in range(fgp.num_tasks)],dim=-1) for i in range(fgp.num_tasks)],dim=-2)
            fgp.prior_mean = torch.linalg.solve_ex(tasksums,rhs[...,None])[0][...,0]
        delta = y.clone()
        for i in range(fgp.num_tasks):
            delta[...,fgp.n_cumsum[i]:(fgp.n_cumsum[i]+fgp.n[i])] -= fgp.prior_mean[...,i,None]
        v = torch.einsum("...ij,...j->...i",thetainv,delta)
        norm_term = (delta*v).sum(-1,keepdim=True)
        logdet = logdet[...,None]
        d_out = norm_term.numel()
        term1 = norm_term.sum()
        mll_const = d_out*fgp.n.sum()*np.log(2*np.pi)
        term2 = d_out/torch.tensor(logdet.shape).prod()*logdet.sum()
        mll_loss = 1/2*(term1+term2+mll_const)
        return mll_loss
    def gcv_loss(self, fgp, update_prior_mean):
        thetainv,logdet = self(fgp)
        y = torch.cat(fgp._y,dim=-1) 
        thetainv2 = torch.einsum("...ij,...jk->...ik",thetainv,thetainv)
        if update_prior_mean:
            rhs = torch.einsum("...ij,...j->...i",thetainv2,y).split(self.n.tolist(),dim=-1)
            rhs = torch.cat([rhs_i.sum(-1,keepdim=True) for rhs_i in rhs],dim=-1)
            thetainv2_split = [thetinv2_i.split(self.n.tolist(),dim=-1) for thetinv2_i in thetainv2.split(self.n.tolist(),dim=-2)]
            tasksums = torch.cat([torch.cat([thetainv2_split[i][j].sum((-2,-1),keepdim=True) for j in range(fgp.num_tasks)],dim=-1) for i in range(fgp.num_tasks)],dim=-2)
            fgp.prior_mean = torch.linalg.solve_ex(tasksums,rhs[...,None])[0][...,0]
        delta = y.clone()
        for i in range(fgp.num_tasks):
            delta[...,fgp.n_cumsum[i]:(fgp.n_cumsum[i]+fgp.n[i])] -= fgp.prior_mean[...,i,None]
        v = torch.einsum("...ij,...j->...i",thetainv2,delta)
        numer = (v*delta).sum(-1,keepdim=True)
        tr_k_inv = torch.einsum("...ii",thetainv)[...,None]
        denom = (tr_k_inv/thetainv.size(-1))**2
        gcv_loss = (numer/denom).sum()
        return gcv_loss
    def cv_loss(self, fgp, cv_weights, update_prior_mean):
        thetainv,logdet = self(fgp)
        y = torch.cat(fgp._y,dim=-1) 
        nrange = torch.arange(thetainv.size(-1),device=fgp.device)
        diag = cv_weights/thetainv[...,nrange,nrange]**2
        cmat = torch.einsum("...ij,...jk->...ik",thetainv,diag[...,None]*thetainv)
        if update_prior_mean:
            rhs = torch.einsum("...ij,...j->...i",cmat,y).split(self.n.tolist(),dim=-1)
            rhs = torch.cat([rhs_i.sum(-1,keepdim=True) for rhs_i in rhs],dim=-1)
            cmat_split = [cmat_i.split(self.n.tolist(),dim=-1) for cmat_i in cmat.split(self.n.tolist(),dim=-2)]
            tasksums = torch.cat([torch.cat([cmat_split[i][j].sum((-2,-1),keepdim=True) for j in range(fgp.num_tasks)],dim=-1) for i in range(fgp.num_tasks)],dim=-2)
            fgp.prior_mean = torch.linalg.solve_ex(tasksums,rhs[...,None])[0][...,0]
        delta = y.clone()
        for i in range(fgp.num_tasks):
            delta[...,fgp.n_cumsum[i]:(fgp.n_cumsum[i]+fgp.n[i])] -= fgp.prior_mean[...,i,None]
        v = torch.einsum("...ij,...j->...i",cmat,delta)
        cv_losses = (v*delta).sum(-1)
        cv_loss = cv_losses.sum()
        return cv_loss