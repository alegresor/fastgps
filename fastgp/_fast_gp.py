from .util import optional_requires_grad_func
import torch 
import numpy as np 
import scipy.stats 
import os


class _FastGP(torch.nn.Module):
    def __init__(self,
        f,
        seq,
        n,
        alpha,
        scale,
        lengthscales,
        noise,
        device,
        save_y,
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
        self.device = torch.device(device)
        assert callable(f), "f must be a callable"
        self.f = f 
        self.seq = seq
        self.d = self.seq.d
        assert np.isscalar(n) and n%1==0 and n>0 and np.log2(n)%1==0, "require n=2^m for some m>=0" 
        self.n_min = 0
        self.n_max = int(n)
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
        self.save_y = save_y
        self.x,self._x = self._sample(self.n_min,self.n_max)
        y = self.f(self.x)
        assert y.size(-1)==self.n_max
        self.d_out = y.numel()/self.n_max
        self.y_shape = list(y.shape)
        self.ytilde = self.ft(y)
        self.k1parts = self._kernel_parts(self._x,self._x[None,0,:])
        k1 = self._kernel_from_parts(self.k1parts)
        k1[0] += self.noise
        self.lam = np.sqrt(self.n_max)*self.ft(k1)
        self.coeffs = self.ift(self.ytilde/self.lam).real
        if self.save_y: 
            self.y = y
        self._reset_next()
    def _reset_next(self):
        self.__x_next_ = None 
        self._x_next_ = None 
        self.raw_scale_next = None 
        self.raw_lengthscales_next = None 
        self.raw_noise_next = None
        self._k1parts_next = None
        self._lam_next_full = None
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
    def scale_next(self):
        return self.tf_scale(self.raw_scale_next)
    @property
    def lengthscales_next(self):
        return self.tf_lengthscales(self.raw_lengthscales_next)
    @property
    def noise_next(self):
        return self.tf_noise(self.raw_noise_next)
    @property
    def _x_next(self):
        if self.__x_next_ is None:
            self._x_next_,self.__x_next_ = self._sample(self.n_max,2*self.n_max)
        return self.__x_next_
    @property
    def x_next(self):
        if self._x_next_ is None:
            self._x_next_,self.__x_next_ = self._sample(self.n_max,2*self.n_max)
        return self._x_next_
    @property 
    def k1parts_next(self):
        if self._k1parts_next is None:
            self._k1parts_next = self._kernel_parts(self._x_next,self._x[None,0,:])
        return self._k1parts_next
    @property 
    def lam_next_full(self):
        if ((self._lam_next_full is None or self.raw_scale_next is None or self.raw_lengthscales_next is None or self.raw_noise_next is None) or 
            ((self.raw_scale_next!=self.raw_scale).any() or (self.raw_lengthscales_next!=self.raw_lengthscales).any() or (self.raw_noise!=self.raw_noise_next).any())):
            self.raw_scale_next = self.raw_scale.clone()
            self.raw_lengthscales_next = self.raw_lengthscales.clone()
            self.raw_noise_next = self.raw_noise.clone()
            k1_next = self._kernel_from_parts(self.k1parts_next)
            lam_next = np.sqrt(self.n_max)*self.ft(k1_next)
            omega_lam_next = self.omega*lam_next
            self._lam_next_full = torch.hstack([
                self.lam+omega_lam_next,
                self.lam-omega_lam_next])
            FASTGP_DEBUG = os.environ.get("FASTGP_DEBUG")
            if FASTGP_DEBUG=="True":
                k1parts_next_full = torch.vstack([self.k1parts,self.k1parts_next])
                k1_next_full = self._kernel_from_parts(k1parts_next_full)
                lam_next_full_ref = np.sqrt(2*self.n_max)*self.ft(k1_next_full)
                assert torch.allclose(self._lam_next_full,lam_next_full_ref,rtol=1e-4,atol=0)
        return self._lam_next_full
    def double_n(self):
        """
        Double the sample size `n` and perform efficient updates.
        """
        y_next = self.f(self.x_next)
        assert list(y_next.shape)==self.y_shape
        self.y_shape[-1] *= 2
        if self.save_y:
            self.y = torch.cat([self.y,y_next],-1)
        ytilde_next = self.ft(y_next)
        omega_ytilde_next = self.omega*ytilde_next
        self.ytilde = 1/np.sqrt(2)*torch.hstack([
            self.ytilde+omega_ytilde_next,
            self.ytilde-omega_ytilde_next])
        FASTGP_DEBUG = os.environ.get("FASTGP_DEBUG")
        if FASTGP_DEBUG=="True":
            assert self.save_y, "os.environ['FASTGP_DEBUG']='True' requires save_y=True"
            ytilde_ref = self.ft(self.y)
            assert torch.allclose(self.ytilde,ytilde_ref,rtol=1e-8,atol=0)
        self.lam = self.lam_next_full
        self.coeffs = self.ift(self.ytilde/self.lam).real
        if self.x.data_ptr()==self._x.data_ptr():
            self.x = self._x = torch.vstack([self.x,self.x_next])
        else:
            self.x = torch.vstack([self.x,self.x_next])
            self._x = torch.vstack([self._x,self._x_next])
        self.k1parts = torch.vstack([self.k1parts,self.k1parts_next])
        self.n_min = self.n_max 
        self.n_max = 2*self.n_max
        self._double_n_omega()
        self._reset_next()
    def _kernel_parts(self, x, z):
        return self._kernel_parts_from_delta(self._ominus(x,z))
    def _kernel_from_parts(self, parts):
        return self.scale*(1+self.lengthscales*parts).prod(-1)
    def _kernel_from_delta(self, delta):
        return self._kernel_from_parts(self._kernel_parts_from_delta(delta))
    def kernel(self, x, z):
        return self._kernel_from_parts(self._kernel_parts(x,z))
    @optional_requires_grad_func
    def post_mean(self, x):
        """
        Posterior mean. 

        Args:
            x (torch.Tensor[N,d]): sampling locations
        
        Returns:
            pmean (torch.Tensor[*batch_shape,N]): posterior mean where `batch_shape` is inferred from `y=f(x)`
        """
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        k = self.kernel(x[:,None,:],self._x[None,:,:])
        return torch.einsum("il,...l->...i",k,self.coeffs)
    @optional_requires_grad_func
    def post_cov(self, x, z):
        """
        Posterior covariance. 
        If `torch.equal(x,z)` then the diagonal of the covariance matrix is forced to be non-negative. 

        Args:
            x (torch.Tensor[N,d]): sampling locations
            z (torch.Tensor[M,d]): sampling locations
        
        Returns:
            pcov (torch.Tensor[N,M]): posterior covariance matrix
        """
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        assert z.ndim==2 and z.size(1)==self.d, "z must a torch.Tensor with shape (-1,d)"
        equal = torch.equal(x,z)
        k = self.kernel(x[:,None,:],z[None,:,:])
        k1t = self.ft(self.kernel(x[:,None,:],self._x[None,:,:]))
        k2t = k1t if equal else self.ft(self.kernel(z[:,None,:],self._x[None,:,:])) 
        kmat = k-torch.einsum("il,rl->ir",k1t.conj(),k2t/self.lam).real
        if equal:
            nrange = torch.arange(x.size(0),device=x.device)
            diag = kmat[nrange,nrange]
            diag[diag<0] = 0 
            kmat[nrange,nrange] = diag 
        return kmat
    @optional_requires_grad_func
    def post_var(self, x):
        """
        Posterior variance. Forced to be non-negative.  

        Args:
            x (torch.Tensor[N,d]): sampling locations

        Returns:
            pvar (torch.Tensor[N]): posterior variance vector
        """
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        k = self.kernel(x,x)
        k1t = self.ft(self.kernel(x[:,None,:],self._x[None,:,:]))
        diag = k-torch.einsum("il,il->i",k1t.conj(),k1t/self.lam).real
        diag[diag<0] = 0 
        return diag
    @optional_requires_grad_func 
    def post_ci(self, x, confidence:float=0.99):
        """
        Posterior credible interval.

        Args:
            x (torch.Tensor[N,d]): sampling locations
            confidence (float): confidence level in (0,1) for the credible interval

        Returns:
            pmean (torch.Tensor[*batch_shape,N]): posterior mean where `batch_shape` is inferred from `y=f(x)`
            pvar (torch.Tensor[N]): posterior variance vector
            q (np.float64): quantile 
                ```python
                scipy.stats.norm.ppf(1-(1-confidence)/2)
                ```
            ci_low (torch.Tensor[*batch_shape,N]): credible interval lower bound
            ci_high (torch.Tensor[*batch_shape,N]): credible interval upper bound
        """
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pmean = self.post_mean(x) 
        pvar = self.post_var(x)
        pstd = torch.sqrt(pvar)
        ci_low = pmean-q*pstd 
        ci_high = pmean+q*pstd 
        return pmean,pvar,q,ci_low,ci_high
    @optional_requires_grad_func
    def post_cubature_mean(self):
        """
        Posterior cubature mean. 

        Returns:
            pcmean (torch.Tensor[*batch_shape]): posterior cubature mean where `batch_shape` is inferred from `y=f(x)`
        """
        pcmean = self.scale*self.coeffs.sum()
        FASTGP_DEBUG = os.environ.get("FASTGP_DEBUG")
        if FASTGP_DEBUG=="True":
            assert self.save_y, "os.environ['FASTGP_DEBUG']='True' requires save_y=True"
            assert torch.allclose(pcmean,self.y.mean(),atol=1e-3), "pcmean-self.y.mean()"
            assert torch.allclose(pcmean,self.ytilde[0].real/np.sqrt(self.n_max),atol=1e-3)
        return pcmean
    @optional_requires_grad_func
    def post_cubature_var(self):
        """
        Posterior cubature variance. 

        Returns:
            pcvar (torch.Tensor[*batch_shape]): posterior cubature variance where `batch_shape` is inferred from `y=f(x)`
        """
        return self.scale-self.scale**2*self.n_max/self.lam[0].real
    @optional_requires_grad_func
    def post_cubature_ci(self, confidence:float=0.99):
        """
        Posterior cubature credible.

        Args:
            confidence (float): confidence level in (0,1) for the credible interval
        
        Returns:
            pcmean (torch.Tensor[*batch_shape]): scalar posterior cubature mean where `batch_shape` is inferred from `y=f(x)`
            pcvar (torch.Tensor[*batch_shape]): scalar posterior cubature variance with shape (N,)
            q (np.float64): quantile
                ```python
                scipy.stats.norm.ppf(1-(1-confidence)/2)
                ```
            cci_low (torch.Tensor[*batch_shape]): scalar credible interval lower bound
            cci_high (torch.Tensor[*batch_shape]): scalar credible interval upper bound
        """
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pmean = self.post_cubature_mean() 
        pvar = self.post_cubature_var()
        pstd = torch.sqrt(pvar)
        ci_low = pmean-q*pstd 
        ci_high = pmean+q*pstd 
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
            optimizer (torch.optim.Optimizer): optimizer defaulted to torch.optim.Rprop(self.parameters(),lr=lr)
            lr (float): learning rate for default optimizer
            store_mll_hist (bool): it True, store and return iteration data for mll
            store_scale_hist (bool): it True, store and return iteration data for the kernel scale parameter
            store_lengthscales_hist (bool): it True, store and return iteration data for the kernel lengthscale parameters
            store_noise_hist (bool): it True, store and return iteration data for noise
            verbose (int): log every verbose iterations, set to 0 for silent mode
            verbose_indent (int): indent to be applied when logging, helpful for logging multiple models
            stop_crit_improvement_threshold (float): stop fitting when the maximum number of iterations is reached or the best mll is note reduced by stop_crit_improvement_threshold for stop_crit_wait_iterations iterations 
            stop_crit_wait_iterations (int): number of iterations to wait for improved mll before early stopping, see the argument description for stop_crit_improvement_threshold
        
        Returns:
            data (dict): iteration data which, dependeing on storage arguments, may include keys in 
                ```python
                ["mll_hist","scale_hist","lengthscales_hist","noise_hist"]
                ```
        """
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
        mll_const = self.d_out*self.n_max*np.log(2*np.pi)
        stop_crit_best_mll = torch.inf 
        stop_crit_save_mll = torch.inf 
        stop_crit_iterations_without_improvement_mll = 0
        for i in range(iterations+1):
            mll = (torch.abs(self.ytilde)**2/self.lam.real).sum()+self.d_out*torch.log(torch.abs(self.lam)).sum()+mll_const
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
            k1 = self._kernel_from_parts(self.k1parts)
            k1[0] += self.noise
            self.lam = np.sqrt(self.n_max)*self.ft(k1)
        self.coeffs = self.ift(self.ytilde/self.lam).real
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


