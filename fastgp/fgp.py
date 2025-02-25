import torch 
import qmcpy as qp
import numpy as np 
import scipy.special 

class _FastGP(torch.nn.Module):
    def __init__(self,
        f,
        dd_obj,
        n,
        alpha,
        global_scale,
        lengthscales,
        noise,
        device,
        save_x,
        save_y,
        tfs_global_scale,
        tfs_lengthscales,
        tfs_noise,
        requires_grad_global_scale, 
        requires_grad_lengthscales, 
        requires_grad_noise, 
        ft,
        ift,
    ):
        super().__init__()
        assert torch.get_default_dtype()==torch.float64, "fast transforms do not work without torch.float64 precision" 
        self.device = torch.device(device)
        assert callable(f), "f must be a callable"
        self.f = f 
        self.dd_obj = dd_obj
        self.d = self.dd_obj.d
        assert np.isscalar(n) and n%1==0 and n>0 and np.log2(n)%1==0, "require n=2^m for some m>=0" 
        self.n_min = 0
        self.n_max = int(n)
        assert (np.isscalar(alpha) and alpha%1==0) or (isinstance(alpha,torch.Tensor) and alpha.shape==(self,d,)), "alpha should be an int or a torch.Tensor of length d"
        if np.isscalar(alpha):
            alpha = int(alpha)*torch.ones(self.d,dtype=int,device=self.device)
        self.alpha = alpha
        assert np.isscalar(global_scale) and global_scale>0, "global_scale should be a positive float"
        global_scale = torch.tensor(global_scale,device=self.device)
        assert len(tfs_global_scale)==2 and callable(tfs_global_scale[0]) and callable(tfs_global_scale[1]), "tfs_global_scale should be a tuple of two callables, the transform and inverse transform"
        self.tf_global_scale = tfs_global_scale[1]
        self.raw_global_scale = torch.nn.Parameter(tfs_global_scale[0](global_scale),requires_grad=requires_grad_global_scale)
        assert (np.isscalar(lengthscales) and lengthscales>0) or (isinstance(lengthscales,torch.Tensor) and lengthscales.shape==(self,d) and (lengthscales>0).all()), "lengthscales should be a float or torch.Tensor of length d and must be postivie"
        if np.isscalar(lengthscales): 
            lengthscales = lengthscales*torch.ones(self.d,device=self.device)
        assert len(tfs_lengthscales)==2 and callable(tfs_lengthscales[0]) and callable(tfs_lengthscales[1]), "tfs_lengthscales should be a tuple of two callables, the transform and inverse transform"
        self.tf_lengthscales = tfs_lengthscales[1]
        self.raw_lengthscales = torch.nn.Parameter(tfs_lengthscales[0](lengthscales),requires_grad=requires_grad_lengthscales)
        assert np.isscalar(noise) and noise>0, "noise should be a positive float"
        noise = torch.tensor(noise,device=self.device)
        assert len(tfs_noise)==2 and callable(tfs_noise[0]) and callable(tfs_noise[1]), "tfs_global_scale should be a tuple of two callables, the transform and inverse transform"
        self.tf_noise = tfs_noise[1]
        self.raw_noise = torch.nn.Parameter(tfs_noise[0](noise),requires_grad=requires_grad_noise)
        self.ft = ft 
        self.ift = ift
        self.save_x = save_x 
        self.save_y = save_y
    def _finish_init(self):
        x,y,k1full = self._setup()
        assert y.size(-1)==(self.n_max-self.n_min)
        self.k1full = k1full
        self.ytilde = self.ft(y)
        if self.save_x:
            self.x = x
        if self.save_y: 
            self.y = y
        self.d_out = y[...,-1].numel()
    @property
    def global_scale(self):
        return self.tf_global_scale(self.raw_global_scale)
    @property
    def lengthscales(self):
        return self.tf_lengthscales(self.raw_lengthscales)
    @property
    def noise(self):
        return self.tf_noise(self.raw_noise)
    def fit(self, steps:int=10, optimizer:torch.optim.Optimizer=None, lr:float=1e-1):
        assert isinstance(steps,int)
        if optimizer is None: 
            optimizer = torch.optim.Rprop(self.parameters(),lr=lr)
        assert isinstance(optimizer,torch.optim.Optimizer)
        for i in range(steps+1):
            optimizer.zero_grad()
            k1 = self.global_scale*(1+self.lengthscales*self.k1full).prod(1)
            k1[0] += self.noise
            self.lam = np.sqrt(self.n_max)*self.ft(k1)
            self.coeffs = self.ift(self.ytilde/self.lam)
            mll = (self.ytilde*self.lam*self.ytilde).real.sum()-self.d_out*torch.log(torch.abs(self.lam)).sum()-self.d_out*self.n_max*np.log(2*np.pi)
            print(mll)
            if i==steps: break
            mll.backward()
            optimizer.step()

class FastGPRLattice(_FastGP):
    def __init__(self,
            f:callable = lambda x: 1/2*((10*x-5)**4-16*(10*x-5)**2+5*(10*x-5)).sum(1), # https://www.sfu.ca/~ssurjano/stybtang.html
            lattice:qp.Lattice = qp.Lattice(2,seed=7),
            n:int = 2**16,
            alpha:int = 3,
            global_scale:float = 1., 
            lengthscales:torch.Tensor = 1., 
            noise:float = 1e-8, 
            device:torch.device = "cpu",
            save_x = False,
            save_y = False,
            tfs_global_scale = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_lengthscales = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_noise = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            requires_grad_global_scale = True, 
            requires_grad_lengthscales = True, 
            requires_grad_noise = False, 
            compile_fts:bool = False,
            ):
        assert isinstance(alpha,int) and alpha in qp.kernel_methods.util.shift_invar_ops.BERNOULLIPOLYSDICT.keys(), "alpha must be in %s"%list(qp.kernel_methods.util.shift_invar_ops.BERNOULLIPOLYSDICT.keys())
        assert isinstance(lattice,qp.Lattice) and lattice.order=="NATURAL" and lattice.replications==1, "lattice should be a qp.Lattice instance with order='NATURAL' and replications=1"
        ft = torch.compile(qp.fftbr_torch) if compile_fts else qp.fftbr_torch
        ift = torch.compile(qp.ifftbr_torch) if compile_fts else qp.ifftbr_torch
        super().__init__(
            f,
            lattice,
            n,
            alpha,
            global_scale,
            lengthscales,
            noise,device,
            save_x,
            save_y,
            tfs_global_scale,
            tfs_lengthscales,
            tfs_noise,
            requires_grad_global_scale,
            requires_grad_lengthscales,
            requires_grad_noise,
            ft,
            ift,
        )
        self.const = (-1)**(self.alpha+1)*torch.exp(2*self.alpha*np.log(2*np.pi)-torch.lgamma(2*self.alpha+1))
        super()._finish_init()
    def _setup(self):
        x = torch.from_numpy(self.dd_obj.gen_samples(n_min=self.n_min,n_max=self.n_max)).to(torch.get_default_dtype()).to(self.device)
        y = self.f(x)
        delta = (x-x[0])%1
        k1full = self.const*torch.vstack([qp.kernel_methods.bernoulli_poly(2*self.alpha[j].item(),delta[:,j]) for j in range(self.d)]).T
        return x,y,k1full

class FastGPRDigitalNetB2(_FastGP):
    def __init__(self,
            f:callable = lambda x: 1/2*((10*x-5)**4-16*(10*x-5)**2+5*(10*x-5)).sum(1), # https://www.sfu.ca/~ssurjano/stybtang.html
            dnb2:qp.DigitalNetB2 = qp.DigitalNetB2(2,seed=7),
            n:int = 2**16,
            alpha:int = 3,
            global_scale:float = 1., 
            lengthscales:torch.Tensor = 1., 
            noise:float = 1e-8, 
            device:torch.device = "cpu",
            save_x = False,
            save_y = False,
            tfs_global_scale = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_lengthscales = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_noise = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            requires_grad_global_scale = True, 
            requires_grad_lengthscales = True, 
            requires_grad_noise = False, 
            compile_fts:bool = False,
            ):
        assert isinstance(alpha,int) and alpha in qp.kernel_methods.util.dig_shift_invar_ops.WEIGHTEDWALSHFUNCSPOS.keys(), "alpha must be in %s"%list(qp.kernel_methods.util.dig_shift_invar_ops.WEIGHTEDWALSHFUNCSPOS.keys())
        assert isinstance(dnb2,qp.DigitalNetB2) and dnb2.order=="NATURAL" and dnb2.replications==1 and dnb2.t_lms<64 and dnb2.randomize in ['LMS_DS','DS','LMS','FALSE'], "dnb2 should be a qp.DigitalNetB2 instance with order='NATURAL', replications=1, t_lms<64, and randomize in ['LMS_DS','DS','LMS','FALSE']"
        self.t = dnb2.t_lms
        ft = ift = torch.compile(qp.fwht_torch) if compile_fts else qp.fwht_torch
        super().__init__(
            f,
            dnb2,
            n,
            alpha,
            global_scale,
            lengthscales,
            noise,device,
            save_x,
            save_y,
            tfs_global_scale,
            tfs_lengthscales,
            tfs_noise,
            requires_grad_global_scale,
            requires_grad_lengthscales,
            requires_grad_noise,
            ft,
            ift,
        )
        self.const = (-1)**(self.alpha+1)*torch.exp(2*self.alpha*np.log(2*np.pi)-torch.lgamma(2*self.alpha+1))
        super()._finish_init()
    def _setup(self):
        xb = torch.from_numpy(self.dd_obj.gen_samples(n_min=self.n_min,n_max=self.n_max,return_binary=True).astype(np.int64)).to(self.device)
        x = xb*2**(-self.t)
        y = self.f(x)
        delta = xb^xb[0]
        k1full = torch.vstack([qp.kernel_methods.weighted_walsh_funcs(self.alpha[j].item(),delta[:,j],self.t) for j in range(self.d)]).T
        return x,y,k1full