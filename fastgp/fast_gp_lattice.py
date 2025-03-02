from ._fast_gp import _FastGP
import torch 
import qmcpy as qmcpy
import numpy as np
from typing import Tuple,Union

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
        ...     seq = qmcpy.Lattice(dimension=d,seed=7),
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
        >>> assert torch.allclose(pcov.diagonal(),pvar)

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

        >>> pcmean,pcvar,q,cci_low,cci_high = fgp.post_cubature_ci(confidence=0.99)
        >>> cci_low
        tensor(20.1797)
        >>> cci_high
        tensor(20.1887)

        >>> pcov_future = fgp.post_cov(x,z,n=2*fgp.n_max)
        >>> pvar_future = fgp.post_var(x,n=2*fgp.n_max)
        >>> pcvar_future = fgp.post_cubature_var(n=2*fgp.n_max)

        >>> fgp.double_n()
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0309)

        >>> assert torch.allclose(fgp.post_cov(x,z),pcov_future)
        >>> assert torch.allclose(fgp.post_var(x),pvar_future)
        >>> assert torch.allclose(fgp.post_cubature_var(),pcvar_future)

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

        >>> pcov_8n = fgp.post_cov(x,z,n=8*fgp.n_max)
        >>> pvar_8n = fgp.post_var(x,n=8*fgp.n_max)
        >>> pcvar_8n = fgp.post_cubature_var(n=8*fgp.n_max)
        >>> fgp.add_n(n=8*fgp.n_max)
        >>> assert torch.allclose(fgp.post_cov(x,z),pcov_8n)
        >>> assert torch.allclose(fgp.post_var(x),pvar_8n)
        >>> assert torch.allclose(fgp.post_cubature_var(),pcvar_8n)
    """
    def __init__(self,
            f:callable,
            seq:qmcpy.Lattice,
            n:int = 2**10,
            alpha:int = 2,
            scale:float = 1., 
            lengthscales:torch.Tensor = 1, 
            noise:float = 1e-8, 
            device:torch.device = "cpu",
            tfs_scale:Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_lengthscales:Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_noise:Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            requires_grad_scale:bool = True, 
            requires_grad_lengthscales:bool = True, 
            requires_grad_noise:bool = False, 
            compile_fts:bool = False,
            compile_fts_kwargs:dict = {},
            ):
        """
        Args:
            f (callable): function to model where `y=f(x)` with `x.shape==(n,d)` and `y.shape==(*batch_size,n)`, e.g. the <a href="https://www.sfu.ca/~ssurjano/stybtang.html" target="_blank">Styblinski-Tang function</a> is 
                ```python
                f = lambda x: 1/2*((10*x-5)**4-16*(10*x-5)**2+5*(10*x-5)).sum(1)
                ```
            seq (Union[qmcpy.Lattice,int]): lattice generator with order="NATURAL" and randomize in ["SHIFT","FALSE"], where if an int `d` is passed in we use
                ```python
                qmcpy.Lattice(d)
                ```
                See the <a href="https://qmcpy.readthedocs.io/en/latest/algorithms.html#module-qmcpy.discrete_distribution.lattice.lattice" target="_blank">`qmcpy.Lattice` docs</a> for more info
            n (int): number of lattice points to generate
            alpha (int): smoothness parameter
            scale (float): kernel global scaling parameter
            lengthscales (torch.Tensor[d]): vector of kernel lengthscales
            noise (float): positive noise variance i.e. nugget term
            device (torch.device): torch device which is required to support torch.float64
            tfs_scale (Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            tfs_lengthscales (Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            tfs_noise (Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            requires_grad_scale (bool): wheather or not to optimize the scale parameter
            requires_grad_lengthscales (bool): wheather or not to optimize lengthscale parameters
            requires_grad_noise (bool): wheather or not to optimize the noise parameter
            compile_fts (bool): if `True`, use `torch.compile(qmcpy.fftbr_torch,**compile_fts)` and `torch.compile(qmcpy.ifftbr_torch,**compile_fts)`, otherwise use the uncompiled versions
            compile_fts_kwargs (dict): keyword arguments to `torch.compile`, see the `compile_fts argument`
        """
        assert isinstance(alpha,int) and alpha in qmcpy.kernel_methods.util.shift_invar_ops.BERNOULLIPOLYSDICT.keys(), "alpha must be in %s"%list(qmcpy.kernel_methods.util.shift_invar_ops.BERNOULLIPOLYSDICT.keys())
        if isinstance(seq,int):
            seq = qmcpy.Lattice(seq,seed=7)
        assert isinstance(seq,qmcpy.Lattice) and seq.order=="NATURAL" and seq.replications==1, "seq should be a qmcpy.Lattice instance with order='NATURAL' and replications=1"
        ft = torch.compile(qmcpy.fftbr_torch,**compile_fts_kwargs) if compile_fts else qmcpy.fftbr_torch
        ift = torch.compile(qmcpy.ifftbr_torch,**compile_fts_kwargs) if compile_fts else qmcpy.ifftbr_torch
        self.__const_for_kernel = None
        super().__init__(
            f,
            seq,
            n,
            alpha,
            scale,
            lengthscales,
            noise,device,
            tfs_scale,
            tfs_lengthscales,
            tfs_noise,
            requires_grad_scale,
            requires_grad_lengthscales,
            requires_grad_noise,
            ft,
            ift,
        )
    def get_omega(self, m):
        return torch.exp(-torch.pi*1j*torch.arange(2**m,device=self.device)/2**m)
    def _sample(self, n_min, n_max):
        x = torch.from_numpy(self.seq.gen_samples(n_min=n_min,n_max=n_max)).to(torch.get_default_dtype()).to(self.device)
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
