import torch 
import qmcpy as qmcpy
import numpy as np 
import scipy.stats 
import typing 
import os

class _FastGP(torch.nn.Module):
    def __init__(self,
        f,
        dd_obj,
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
        self.dd_obj = dd_obj
        self.d = self.dd_obj.d
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
        self.k1full = self._kernel_parts(self._x,self._x[None,0,:])
        k1 = self._kernel_from_parts(self.k1full)
        k1[0] += self.noise
        self.lam = np.sqrt(self.n_max)*self.ft(k1)
        self.coeffs = self.ift(self.ytilde/self.lam).real
        if self.save_y: 
            self.y = y
    def double_n(self):
        """
        Double the sample size `n` and perform efficient updates.
        """
        self.n_min = self.n_max 
        self.n_max = 2*self.n_max
        x_new,_x_new = self._sample(self.n_min,self.n_max)
        ynew = self.f(x_new)
        ytilde_new = self.ft(ynew)
        assert list(ynew.shape)==self.y_shape
        k1full_new = self._kernel_parts(_x_new,self._x[None,0,:])
        k1_new = self._kernel_from_parts(k1full_new)
        lam_new = np.sqrt(self.n_min)*self.ft(k1_new)
        omega_ytilde_new = self.omega*ytilde_new
        self.ytilde = 1/np.sqrt(2)*torch.hstack([
            self.ytilde+omega_ytilde_new,
            self.ytilde-omega_ytilde_new])
        omega_lam_new = self.omega*lam_new
        self.lam = torch.hstack([
            self.lam+omega_lam_new,
            self.lam-omega_lam_new])
        self.k1full = torch.vstack([self.k1full,k1full_new])
        self.y_shape[-1] *= 2
        if self.save_y:
            self.y = torch.cat([self.y,ynew],-1)
        FASTGP_DEBUG = os.environ.get("FASTGP_DEBUG")
        if FASTGP_DEBUG=="True":
            assert self.save_y, "os.environ['FASTGP_DEBUG']='True' requires save_y=True"
            ytilde_ref = self.ft(self.y)
            assert torch.allclose(self.ytilde,ytilde_ref,rtol=1e-8,atol=0)
            k1 = self._kernel_from_parts(self.k1full)
            lam_ref = np.sqrt(self.n_max)*self.ft(k1)
            assert torch.allclose(self.lam,lam_ref,rtol=1e-4,atol=0)
        if self.x.data_ptr()==self._x.data_ptr():
            self.x = self._x = torch.vstack([self.x,x_new])
        else:
            self.x = torch.vstack([self.x,x_new])
            self._x = torch.vstack([self._x,_x_new])
        self.coeffs = self.ift(self.ytilde/self.lam).real
        self._double_n_omega()
    @property
    def scale(self):
        return self.tf_scale(self.raw_scale)
    @property
    def lengthscales(self):
        return self.tf_lengthscales(self.raw_lengthscales)
    @property
    def noise(self):
        return self.tf_noise(self.raw_noise)
    def _kernel_parts(self, x, z):
        return self._kernel_parts_from_delta(self._ominus(x,z))
    def _kernel_from_parts(self, parts):
        return self.scale*(1+self.lengthscales*parts).prod(-1)
    def _kernel_from_delta(self, delta):
        return self._kernel_from_parts(self._kernel_parts_from_delta(delta))
    def kernel(self, x, z):
        return self._kernel_from_parts(self._kernel_parts(x,z))
    def post_mean_grad(self, x):
        """
        Posterior mean with gradient. 

        Args:
            x (torch.Tensor): sampling locations with shape (N,d)
        
        Returns:
            torch.Tensor: posterior mean vector with shape (N,) and requires_grad=True
        """
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        k = self.kernel(x[:,None,:],self._x[None,:,:])
        return torch.einsum("il,...l->...i",k,self.coeffs)
    def post_mean(self, x):
        """
        Posterior mean.

        Args:
            x (torch.Tensor): sampling locations with shape (N,d)
        
        Returns:
            torch.Tensor: posterior mean vector with shape (**batch_shape,N)
        """
        with torch.no_grad():
            return self.post_mean_grad(x)
    def post_cov_grad(self, x, z):
        """
        Posterior covariance with gradient. 
        If torch.equal(x,z) then the diagonal of the covariance matrix is forced to be non-negative. 

        Args:
            x (torch.Tensor): sampling locations with shape (N,d)
            z (torch.Tensor): sampling locations with shape (M,d)
        
        Returns:
            torch.Tensor: posterior covariance matrix with shape (N,M) and requires_grad=True
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
    def post_cov(self, x, z):
        """
        Posterior covariance. 
        If torch.equal(x,z) then the diagonal of the covariance matrix is forced to be non-negative. 

        Args:
            x (torch.Tensor): sampling locations with shape (N,d)
            z (torch.Tensor): sampling locations with shape (M,d)
        
        Returns:
            torch.Tensor: posterior covariance matrix with shape (N,M)
        """
        with torch.no_grad():
            return self.post_cov_grad(x,z)
    def post_var_grad(self, x):
        """
        Posterior variance with gradient. Forced to be non-negative.  

        Args:
            x (torch.Tensor): sampling locations with shape (N,d)
        
        Returns:
            torch.Tensor: posterior variance vector with shape (N,) and requires_grad=True
        """
        assert x.ndim==2 and x.size(1)==self.d, "x must a torch.Tensor with shape (-1,d)"
        k = self.kernel(x,x)
        k1t = self.ft(self.kernel(x[:,None,:],self._x[None,:,:]))
        diag = k-torch.einsum("il,il->i",k1t.conj(),k1t/self.lam).real
        diag[diag<0] = 0 
        return diag        
    def post_var(self, x):
        """
        Posterior variance. Forced to be non-negative.  

        Args:
            x (torch.Tensor): sampling locations with shape (N,d)
        
        Returns:
            torch.Tensor: posterior variance vector with shape (N,)
        """
        with torch.no_grad():
            return self.post_var_grad(x)
    def post_ci_grad(self, x, confidence:float=0.99):
        """
        Posterior credible interval with gradients.

        Args:
            x (torch.Tensor): sampling locations with shape (N,d)
            confidence (float): confidence level in (0,1) for the credible interval
        
        Returns:
            torch.Tensor: posterior mean with shape (N,) and requires_grad=True
            torch.Tensor: posterior variance with shape (N,) and requires_grad=True
            np.float64: quantile scipy.stats.norm.ppf(1-(1-confidence)/2)
            torch.Tensor: credible interval lower bound with shape (N,) and requires_grad=True
            torch.Tensor: credible interval upper bound with shape (N,) and requires_grad=True
        """
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pmean = self.post_mean_grad(x) 
        pvar = self.post_var_grad(x)
        pstd = torch.sqrt(pvar)
        ci_low = pmean-q*pstd 
        ci_high = pmean+q*pstd 
        return pmean,pvar,q,ci_low,ci_high
    def post_ci(self, x, confidence=0.99):
        """
        Posterior credible interval.

        Args:
            x (torch.Tensor): sampling locations with shape (N,d)
            confidence (float): confidence level in (0,1) for the credible interval
        
        Returns:
            torch.Tensor: posterior mean with shape (N,)
            torch.Tensor: posterior variance with shape (N,)
            np.float64: quantile scipy.stats.norm.ppf(1-(1-confidence)/2)
            torch.Tensor: credible interval lower bound with shape (N,)
            torch.Tensor: credible interval upper bound with shape (N,)
        """
        with torch.no_grad():
            return self.post_ci_grad(x,confidence=confidence)
    def post_cubature_mean_grad(self):
        """
        Posterior cubature mean with gradient. 

        Returns:
            torch.Tensor: a scalar posterior cubature mean with requires_grad=True
        """
        pcmean = self.scale*self.coeffs.sum()
        FASTGP_DEBUG = os.environ.get("FASTGP_DEBUG")
        if FASTGP_DEBUG=="True":
            assert self.save_y, "os.environ['FASTGP_DEBUG']='True' requires save_y=True"
            assert torch.allclose(pcmean,self.y.mean(),atol=1e-3), "pcmean-self.y.mean()"
            assert torch.allclose(pcmean,self.ytilde[0].real/np.sqrt(self.n_max),atol=1e-3)
        return pcmean
    def post_cubature_mean(self):
        """
        Posterior cubature mean.

        Returns:
            torch.Tensor: a scalar posterior cubature mean
        """
        with torch.no_grad():
            return self.post_cubature_mean_grad()
    def post_cubature_var_grad(self):
        """
        Posterior cubature variance with gradient. 

        Returns:
            torch.Tensor: a scalar posterior cubature variance with requires_grad=True
        """
        return self.scale-self.scale**2*self.n_max/self.lam[0].real
    def post_cubature_var(self):
        """
        Posterior cubature variance.

        Returns:
            torch.Tensor: a scalar posterior cubature variance 
        """
        with torch.no_grad():
            return self.post_cubature_var_grad()
    def post_cubature_ci_grad(self, confidence:float=0.99):
        """
        Posterior cubature credible interval with gradients.

        Args:
            confidence (float): confidence level in (0,1) for the credible interval
        
        Returns:
            torch.Tensor: scalar posterior cubature mean with requires_grad=True
            torch.Tensor: scalar posterior cubature variance with shape (N,) and requires_grad=True
            np.float64: quantile scipy.stats.norm.ppf(1-(1-confidence)/2)
            torch.Tensor: scalar credible interval lower bound with  requires_grad=True
            torch.Tensor: scalar credible interval upper bound with requires_grad=True
        """
        assert np.isscalar(confidence) and 0<confidence<1, "confidence must be between 0 and 1"
        q = scipy.stats.norm.ppf(1-(1-confidence)/2)
        pmean = self.post_cubature_mean() 
        pvar = self.post_cubature_var_grad()
        pstd = torch.sqrt(pvar)
        ci_low = pmean-q*pstd 
        ci_high = pmean+q*pstd 
        return pmean,pvar,q,ci_low,ci_high
    def post_cubature_ci(self, confidence:float=0.99):
        """
        Posterior cubature credible interval with gradients.

        Args:
            confidence (float): confidence level in (0,1) for the credible interval
        
        Returns:
            torch.Tensor: scalar posterior cubature mean with requires_grad=True
            torch.Tensor: scalar posterior cubature variance with shape (N,) and requires_grad=True
            np.float64: quantile scipy.stats.norm.ppf(1-(1-confidence)/2)
            torch.Tensor: scalar credible interval lower bound with  requires_grad=True
            torch.Tensor: scalar credible interval upper bound with requires_grad=True
        """
        with torch.no_grad():
            return self.post_cubature_ci_grad(confidence=confidence)
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
            dict: iteration data, may include keys in ['mll_hist','scale_hist','lengthscales_hist',noise_hist'] dependeing on storage arguments
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
            k1 = self._kernel_from_parts(self.k1full)
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

class FastGPLattice(_FastGP):
    """
    Fast Gaussian process regression using lattice points and shift invariant kernels

    Examples:
        >>> torch.set_default_dtype(torch.float64)

        >>> def f_ackley(x, a=20, b=0.2, c=2*np.pi, scaling=32.768):
        ...     # https://www.sfu.ca/~ssurjano/ackley.html
        ...     assert x.ndim==2
        ...     x = 2*scaling*x-scaling
        ...     t1 = a*torch.exp(-b*torch.sqrt(torch.mean(x**2,1)))
        ...     t2 = torch.exp(torch.mean(torch.cos(c*x),1))
        ...     t3 = a+np.exp(1)
        ...     y = -t1-t2+t3
        ...     return y

        >>> d = 2
        >>> fgp = FastGPLattice(
        ...     f = f_ackley,
        ...     lattice = qmcpy.Lattice(dimension=d,seed=7),
        ...     n = 2**10)

        >>> rng = torch.Generator().manual_seed(17)
        >>> x = torch.rand((2**7,d),generator=rng)
        >>> y = f_ackley(x)
        
        >>> pmean = fgp.post_mean(x)
        >>> pmean.shape
        torch.Size([128])
        >>> torch.linalg.norm(y-pmean)/torch.linalg.norm(y)
        tensor(0.0348)
        >>> assert torch.allclose(fgp.post_mean(fgp.x),fgp.y,atol=1e-3)

        >>> fgp.post_cubature_mean()
        tensor(20.1842)
        >>> fgp.post_cubature_var()
        tensor(7.0015e-09)

        >>> data = fgp.fit()
             iter of 5.0e+03 | NMLL       | noise      | scale      | lengthscales
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    0.00e+00 | 3.70e+05   | 1.00e-08   | 1.00e+00   | [1.00e+00 1.00e+00]
                    5.00e+00 | 4.12e+04   | 1.00e-08   | 2.10e+00   | [2.10e+00 2.10e+00]
                    1.00e+01 | 2.81e+03   | 1.00e-08   | 8.00e+00   | [8.00e+00 8.00e+00]
                    1.50e+01 | 2.82e+03   | 1.00e-08   | 8.18e+00   | [8.18e+00 8.18e+00]
                    2.00e+01 | 2.80e+03   | 1.00e-08   | 7.73e+00   | [7.73e+00 7.73e+00]
                    2.50e+01 | 2.80e+03   | 1.00e-08   | 7.74e+00   | [7.74e+00 7.64e+00]
                    3.00e+01 | 2.80e+03   | 1.00e-08   | 7.75e+00   | [7.66e+00 7.45e+00]
                    3.50e+01 | 2.80e+03   | 1.00e-08   | 8.04e+00   | [7.69e+00 7.17e+00]
                    4.00e+01 | 2.79e+03   | 1.00e-08   | 8.83e+00   | [7.69e+00 6.54e+00]
                    4.50e+01 | 2.79e+03   | 1.00e-08   | 9.01e+00   | [7.68e+00 6.39e+00]
                    5.00e+01 | 2.79e+03   | 1.00e-08   | 9.20e+00   | [7.69e+00 6.18e+00]
                    5.50e+01 | 2.79e+03   | 1.00e-08   | 9.77e+00   | [7.71e+00 5.82e+00]
                    6.00e+01 | 2.79e+03   | 1.00e-08   | 1.02e+01   | [7.74e+00 5.64e+00]
                    6.50e+01 | 2.79e+03   | 1.00e-08   | 1.01e+01   | [7.74e+00 5.58e+00]
                    7.00e+01 | 2.79e+03   | 1.00e-08   | 1.01e+01   | [7.78e+00 5.51e+00]
                    7.50e+01 | 2.79e+03   | 1.00e-08   | 1.02e+01   | [7.87e+00 5.44e+00]
                    8.00e+01 | 2.79e+03   | 1.00e-08   | 1.02e+01   | [8.10e+00 5.29e+00]
                    8.50e+01 | 2.79e+03   | 1.00e-08   | 1.02e+01   | [8.70e+00 4.92e+00]
                    9.00e+01 | 2.79e+03   | 1.00e-08   | 1.02e+01   | [9.15e+00 4.73e+00]
                    9.50e+01 | 2.79e+03   | 1.00e-08   | 1.02e+01   | [9.21e+00 4.62e+00]
                    1.00e+02 | 2.79e+03   | 1.00e-08   | 1.02e+01   | [9.64e+00 4.41e+00]
                    1.05e+02 | 2.79e+03   | 1.00e-08   | 1.02e+01   | [1.02e+01 4.21e+00]
                    1.10e+02 | 2.79e+03   | 1.00e-08   | 1.02e+01   | [1.01e+01 4.17e+00]
                    1.15e+02 | 2.79e+03   | 1.00e-08   | 1.02e+01   | [1.01e+01 4.16e+00]
                    1.17e+02 | 2.79e+03   | 1.00e-08   | 1.02e+01   | [1.02e+01 4.17e+00]
        >>> list(data.keys())
        ['mll_hist', 'scale_hist', 'lengthscales_hist']

        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0359)
        >>> z = torch.rand((2**8,d),generator=rng)
        >>> pcov = fgp.post_cov(x,z)
        >>> pcov.shape
        torch.Size([128, 256])

        >>> pcov = fgp.post_cov(x,x)
        >>> pcov.shape
        torch.Size([128, 128])
        >>> assert (pcov.diagonal()>=0).all()

        >>> pvar = fgp.post_var(x)
        >>> pvar.shape
        torch.Size([128])
        >>> assert (pvar>=0).all()

        >>> pmean,pstd,q,ci_low,ci_high = fgp.post_ci(x,confidence=0.99)
        >>> q
        np.float64(2.5758293035489004)
        >>> ci_low.shape
        torch.Size([128])
        >>> ci_high.shape
        torch.Size([128])

        >>> fgp.post_cubature_mean()
        tensor(20.1842)
        >>> fgp.post_cubature_var()
        tensor(3.0169e-06)

        >>> pcmean,pcvar,q,pci_low,pci_high = fgp.post_cubature_ci(confidence=0.99)
        >>> pci_low
        tensor(20.1797)
        >>> pci_high
        tensor(20.1887)

        >>> fgp.double_n()
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0309)

        >>> data = fgp.fit(verbose=False,store_mll_hist=False,store_scale_hist=False,store_lengthscales_hist=False,store_noise_hist=False)
        >>> assert len(data)==0
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0274)

        >>> fgp.double_n()
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0277)

        >>> data = fgp.fit(verbose=False,store_mll_hist=False,store_scale_hist=False,store_lengthscales_hist=False,store_noise_hist=False)
        >>> assert len(data)==0
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0275)
    """
    def __init__(self,
            f:callable,
            lattice:qmcpy.Lattice,
            n:int = 2**10,
            alpha:int = 2,
            scale:float = 1., 
            lengthscales:torch.Tensor = 1, 
            noise:float = 1e-8, 
            device:torch.device = "cpu",
            save_y:bool = True,
            tfs_scale:typing.Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_lengthscales:typing.Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_noise:typing.Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            requires_grad_scale:bool = True, 
            requires_grad_lengthscales:bool = True, 
            requires_grad_noise:bool = False, 
            compile_fts:bool = False,
            compile_fts_kwargs:dict = {},
            ):
        """
        Args:
            f (callable): function to model where `y=f(x)` with `x.shape==(n,d)` and `y.shape==(...,n)`, e.g. the <a href="https://www.sfu.ca/~ssurjano/stybtang.html" target="_blank">Styblinski-Tang function</a> is 
                ```python
                f = lambda x: 1/2*((10*x-5)**4-16*(10*x-5)**2+5*(10*x-5)).sum(1)
                ```
            lattice (qmcpy.Lattice): lattice generator with order="NATURAL" and randomize in ["SHIFT","FALSE"], e.g.
                ```python
                d = 2 # dimension
                lattice = qmcpy.Lattice(d,seed=7)
                ```
                See the <a href="https://qmcpy.readthedocs.io/en/latest/algorithms.html#module-qmcpy.discrete_distribution.lattice.lattice" target="_blank">`qmcpy.Lattice` docs</a> for more info
            n (int): number of lattice points to generate
            alpha (int): smoothness parameter
            scale (float): kernel global scaling parameter
            lengthscales (torch.Tensor): length d vector of kernel lengthscales
            noise (float): positive noise variance i.e. nugget term
            device (torch.device): torch device which is required to support torch.float64
            save_y (bool): setting to False will save memory by NOT saving `self.y=f(x)`
            tfs_scale (typing.Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            tfs_lengthscales (typing.Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            tfs_noise (typing.Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            requires_grad_scale (bool): wheather or not to optimize the scale parameter
            requires_grad_lengthscales (bool): wheather or not to optimize lengthscale parameters
            requires_grad_noise (bool): wheather or not to optimize the noise parameter
            compile_fts (bool): if True, use `torch.compile(qmcpy.fftbr_torch,**compile_fts)` and `torch.compile(qmcpy.ifftbr_torch,**compile_fts)`, otherwise use the uncompiled versions
            compile_fts_kwargs (dict): keyword arguments to `torch.compile`, see the compile_fts argument
        """
        assert isinstance(alpha,int) and alpha in qmcpy.kernel_methods.util.shift_invar_ops.BERNOULLIPOLYSDICT.keys(), "alpha must be in %s"%list(qmcpy.kernel_methods.util.shift_invar_ops.BERNOULLIPOLYSDICT.keys())
        assert isinstance(lattice,qmcpy.Lattice) and lattice.order=="NATURAL" and lattice.replications==1, "lattice should be a qmcpy.Lattice instance with order='NATURAL' and replications=1"
        ft = torch.compile(qmcpy.fftbr_torch,**compile_fts_kwargs) if compile_fts else qmcpy.fftbr_torch
        ift = torch.compile(qmcpy.ifftbr_torch,**compile_fts_kwargs) if compile_fts else qmcpy.ifftbr_torch
        self.__const_for_kernel = None
        super().__init__(
            f,
            lattice,
            n,
            alpha,
            scale,
            lengthscales,
            noise,device,
            save_y,
            tfs_scale,
            tfs_lengthscales,
            tfs_noise,
            requires_grad_scale,
            requires_grad_lengthscales,
            requires_grad_noise,
            ft,
            ift,
        )
        self.omega = torch.exp(-torch.pi*1j*torch.arange(self.n_max,device=self.device)/self.n_max)
    def _double_n_omega(self):
        omega_new = torch.exp(-torch.pi*1j*(2*torch.arange(self.n_min,device=self.device)+1)/self.n_max)
        self.omega = torch.vstack([self.omega,omega_new]).T.flatten()
    def _sample(self, n_min, n_max):
        x = torch.from_numpy(self.dd_obj.gen_samples(n_min=n_min,n_max=n_max)).to(torch.get_default_dtype()).to(self.device)
        return x,x
    @property
    def const_for_kernel(self):
        if self.__const_for_kernel is None:
            self.__const_for_kernel = (-1)**(self.alpha+1)*torch.exp(2*self.alpha*np.log(2*np.pi)-torch.lgamma(2*self.alpha+1))
        return self.__const_for_kernel
    def _ominus(self, x, z):
        assert ((0<=x)&(x<=1)).all(), "x should have all elements in [0,1]"
        assert ((0<=z)&(z<=1)).all(), "z should have all elements in [0,1]"
        return (x-z)%1
    def _kernel_parts_from_delta(self, delta):
        return self.const_for_kernel*torch.stack([qmcpy.kernel_methods.bernoulli_poly(2*self.alpha[j].item(),delta[...,j]) for j in range(self.d)],-1)

class FastGPDigitalNetB2(_FastGP):
    """
    Fast Gaussian process regression using digitally shifted digital nets paired with digitally shift invariant kernels
    
    Examples:
        >>> torch.set_default_dtype(torch.float64)

        >>> def f_ackley(x, a=20, b=0.2, c=2*np.pi, scaling=32.768):
        ...     # https://www.sfu.ca/~ssurjano/ackley.html
        ...     assert x.ndim==2
        ...     x = 2*scaling*x-scaling
        ...     t1 = a*torch.exp(-b*torch.sqrt(torch.mean(x**2,1)))
        ...     t2 = torch.exp(torch.mean(torch.cos(c*x),1))
        ...     t3 = a+np.exp(1)
        ...     y = -t1-t2+t3
        ...     return y

        >>> d = 2
        >>> fgp = FastGPDigitalNetB2(
        ...     f = f_ackley,
        ...     dnb2 = qmcpy.DigitalNetB2(dimension=d,seed=7),
        ...     n = 2**10)

        >>> rng = torch.Generator().manual_seed(17)
        >>> x = torch.rand((2**7,d),generator=rng)
        >>> y = f_ackley(x)
        
        >>> pmean = fgp.post_mean(x)
        >>> pmean.shape
        torch.Size([128])
        >>> torch.linalg.norm(y-pmean)/torch.linalg.norm(y)
        tensor(0.0336)
        >>> assert torch.allclose(fgp.post_mean(fgp.x),fgp.y)

        >>> data = fgp.fit()
             iter of 5.0e+03 | NMLL       | noise      | scale      | lengthscales
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    0.00e+00 | 3.35e+03   | 1.00e-16   | 1.00e+00   | [1.00e+00 1.00e+00]
                    5.00e+00 | 3.24e+03   | 1.00e-16   | 1.55e+00   | [9.21e-01 9.21e-01]
                    1.00e+01 | 3.13e+03   | 1.00e-16   | 2.94e+00   | [6.32e-01 6.32e-01]
                    1.50e+01 | 3.10e+03   | 1.00e-16   | 3.78e+00   | [5.13e-01 5.13e-01]
                    2.00e+01 | 3.10e+03   | 1.00e-16   | 4.53e+00   | [4.65e-01 4.65e-01]
                    2.50e+01 | 3.09e+03   | 1.00e-16   | 4.93e+00   | [4.35e-01 4.35e-01]
                    3.00e+01 | 3.09e+03   | 1.00e-16   | 5.31e+00   | [4.15e-01 4.20e-01]
                    3.50e+01 | 3.09e+03   | 1.00e-16   | 5.65e+00   | [3.84e-01 4.19e-01]
                    4.00e+01 | 3.09e+03   | 1.00e-16   | 5.71e+00   | [3.83e-01 4.19e-01]
                    4.50e+01 | 3.09e+03   | 1.00e-16   | 5.72e+00   | [3.84e-01 4.19e-01]
                    4.80e+01 | 3.09e+03   | 1.00e-16   | 5.73e+00   | [3.84e-01 4.19e-01]
        >>> list(data.keys())
        ['mll_hist', 'scale_hist', 'lengthscales_hist']

        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0355)
        >>> z = torch.rand((2**8,d),generator=rng)
        >>> pcov = fgp.post_cov(x,z)
        >>> pcov.shape
        torch.Size([128, 256])

        >>> pcov = fgp.post_cov(x,x)
        >>> pcov.shape
        torch.Size([128, 128])
        >>> assert (pcov.diagonal()>=0).all()

        >>> pvar = fgp.post_var(x)
        >>> pvar.shape
        torch.Size([128])
        >>> assert (pvar>=0).all()

        >>> pmean,pstd,q,ci_low,ci_high = fgp.post_ci(x,confidence=0.99)
        >>> q
        np.float64(2.5758293035489004)
        >>> ci_low.shape
        torch.Size([128])
        >>> ci_high.shape
        torch.Size([128])

        >>> fgp.post_cubature_mean()
        tensor(20.1896)
        >>> fgp.post_cubature_var()
        tensor(0.0002)

        >>> pcmean,pcvar,q,pci_low,pci_high = fgp.post_cubature_ci(confidence=0.99)
        >>> pci_low
        tensor(20.1564)
        >>> pci_high
        tensor(20.2228)

        >>> fgp.double_n()
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0258)

        >>> data = fgp.fit(verbose=False,store_mll_hist=False,store_scale_hist=False,store_lengthscales_hist=False,store_noise_hist=False)
        >>> assert len(data)==0
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0258)

        >>> fgp.double_n()
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0192)

        >>> data = fgp.fit(verbose=False,store_mll_hist=False,store_scale_hist=False,store_lengthscales_hist=False,store_noise_hist=False)
        >>> assert len(data)==0
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0187)
    """
    def __init__(self,
            f:callable,
            dnb2:qmcpy.DigitalNetB2,
            n:int = 2**10,
            alpha:int = 2,
            scale:float = 1., 
            lengthscales:torch.Tensor = 1, 
            noise:float = 1e-16, 
            device:torch.device = "cpu",
            save_y:bool = True,
            tfs_scale:typing.Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_lengthscales:typing.Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_noise:typing.Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            requires_grad_scale:bool = True, 
            requires_grad_lengthscales:bool = True, 
            requires_grad_noise:bool = False, 
            compile_fts:bool = False,
            compile_fts_kwargs: dict = {},
            ):
        """
        Args:
            f (callable): function to model where `y=f(x)` with `x.shape==(n,d)` and `y.shape==(...,n)`, e.g. the <a href="https://www.sfu.ca/~ssurjano/stybtang.html" target="_blank">Styblinski-Tang function</a> is 
                ```python
                f = lambda x: 1/2*((10*x-5)**4-16*(10*x-5)**2+5*(10*x-5)).sum(1)
                ```
            dnb2 (qmcpy.DigitalNetB2): digital sequence generator in base $b=2$ with order="NATURAL" and randomize in ["LMS_DS","DS","LMS","FALSE"], e.g.
                ```python
                d = 2 # dimension
                dnb2 = qmcpy.DigitalNetB2(2,seed=7)
                ```
                See the <a href="https://qmcpy.readthedocs.io/en/latest/algorithms.html#module-qmcpy.discrete_distribution.digital_net_b2.digital_net_b2" target="_blank">`qmcpy.DigitalNetB2` docs</a> for more info
            n (int): number of lattice points to generate
            alpha (int): smoothness parameter
            scale (float): kernel global scaling parameter
            lengthscales (torch.Tensor): length d vector of kernel lengthscales
            noise (float): positive noise variance i.e. nugget term
            device (torch.device): torch device which is required to support torch.float64
            save_y (bool): setting to False will save memory by NOT saving `self.y=f(x)`
            tfs_scale (typing.Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            tfs_lengthscales (typing.Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            tfs_noise (typing.Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            requires_grad_scale (bool): wheather or not to optimize the scale parameter
            requires_grad_lengthscales (bool): wheather or not to optimize lengthscale parameters
            requires_grad_noise (bool): wheather or not to optimize the noise parameter
            compile_fts (bool): if True, use `torch.compile(qmcpy.fwht_torch,**compile_fts_kwargs)`, otherwise use the uncompiled version
            compile_fts_kwargs (dict): keyword arguments to `torch.compile`, see the compile_fts argument
        """
        assert isinstance(alpha,int) and alpha in qmcpy.kernel_methods.util.dig_shift_invar_ops.WEIGHTEDWALSHFUNCSPOS.keys(), "alpha must be in %s"%list(qmcpy.kernel_methods.util.dig_shift_invar_ops.WEIGHTEDWALSHFUNCSPOS.keys())
        assert isinstance(dnb2,qmcpy.DigitalNetB2) and dnb2.order=="NATURAL" and dnb2.replications==1 and dnb2.t_lms<64 and dnb2.randomize in ['LMS_DS','DS','LMS','FALSE'], "dnb2 should be a qmcpy.DigitalNetB2 instance with order='NATURAL', replications=1, t_lms<64, and randomize in ['LMS_DS','DS','LMS','FALSE']"
        self.t = dnb2.t_lms
        ft = ift = torch.compile(qmcpy.fwht_torch,**compile_fts_kwargs) if compile_fts else qmcpy.fwht_torch
        super().__init__(
            f,
            dnb2,
            n,
            alpha,
            scale,
            lengthscales,
            noise,device,
            save_y,
            tfs_scale,
            tfs_lengthscales,
            tfs_noise,
            requires_grad_scale,
            requires_grad_lengthscales,
            requires_grad_noise,
            ft,
            ift,
        )
        self.omega = torch.ones(self.n_max,device=self.device)
    def _double_n_omega(self):
        omega_new = torch.ones(self.n_min,device=self.device)
        self.omega = torch.vstack([self.omega,omega_new]).T.flatten()
    def _sample(self, n_min, n_max):
        _x = torch.from_numpy(self.dd_obj.gen_samples(n_min=n_min,n_max=n_max,return_binary=True).astype(np.int64)).to(self.device)
        x = self._convert_from_b(_x)
        return x,_x
    def _convert_to_b(self, x):
        return torch.floor((x%1)*2**(self.t)).to(torch.int64)
    def _convert_from_b(self, xb):
        return xb*2**(-self.t)
    def _ominus(self, x_or_xb, z_or_zb):
        fp_x = torch.is_floating_point(x_or_xb)
        fp_z = torch.is_floating_point(z_or_zb)
        if fp_x:
            assert ((0<=x_or_xb)&(x_or_xb<=1)).all(), "x should have all elements in [0,1]"
        if fp_z:
            assert ((0<=z_or_zb)&(z_or_zb<=1)).all(), "z should have all elements in [0,1]"
        if (not fp_x) and (not fp_z):
            return x_or_xb^z_or_zb
        elif (not fp_x) and fp_z:
            return x_or_xb^self._convert_to_b(z_or_zb)
        elif fp_x and (not fp_z):
            return self._convert_to_b(x_or_xb)^z_or_zb
        else: # fp_x and fp_z
            return self._convert_to_b(x_or_xb)^self._convert_to_b(z_or_zb)
    def _kernel_parts_from_delta(self, delta):
        return torch.stack([qmcpy.kernel_methods.weighted_walsh_funcs(self.alpha[j].item(),delta[...,j],self.t)-1 for j in range(self.d)],-1)