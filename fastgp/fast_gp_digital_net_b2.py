from ._fast_gp import _FastGP
import torch
import numpy as np
import qmcpy as qmcpy
from typing import Tuple,Union

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
        ...     seq = qmcpy.DigitalNetB2(dimension=d,seed=7),
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
        >>> assert torch.allclose(pcov.diagonal(),pvar)

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

        >>> pcmean,pcvar,q,cci_low,cci_high = fgp.post_cubature_ci(confidence=0.99)
        >>> cci_low
        tensor(20.1564)
        >>> cci_high
        tensor(20.2228)
        
        >>> pcov_future = fgp.post_cov(x,z,future=True)
        >>> pvar_future = fgp.post_var(x,future=True)
        >>> pcvar_future = fgp.post_cubature_var(future=True)
        
        >>> fgp.double_n()
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0258)

        >>> assert torch.allclose(fgp.post_cov(x,z),pcov_future)
        >>> assert torch.allclose(fgp.post_var(x),pvar_future)
        >>> assert torch.allclose(fgp.post_cubature_var(),pcvar_future)

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
            seq:Union[qmcpy.DigitalNetB2,int],
            n:int = 2**10,
            alpha:int = 2,
            scale:float = 1., 
            lengthscales:torch.Tensor = 1, 
            noise:float = 1e-16, 
            device:torch.device = "cpu",
            tfs_scale:Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_lengthscales:Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_noise:Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            requires_grad_scale:bool = True, 
            requires_grad_lengthscales:bool = True, 
            requires_grad_noise:bool = False, 
            compile_fts:bool = False,
            compile_fts_kwargs: dict = {},
            ):
        """
        Args:
            f (callable): function to model where `y=f(x)` with `x.shape==(n,d)` and `y.shape==(*batch_size,n)`, e.g. the <a href="https://www.sfu.ca/~ssurjano/stybtang.html" target="_blank">Styblinski-Tang function</a> is 
                ```python
                f = lambda x: 1/2*((10*x-5)**4-16*(10*x-5)**2+5*(10*x-5)).sum(1)
                ```
            seq (Union[qmcpy.DigitalNetB2,int]): digital sequence generator in base $b=2$ with order="NATURAL" and randomize in ["LMS_DS","DS","LMS","FALSE"], where if an int `d` is passed in we use 
                ```python
                qmcpy.DigitalNetB2(d)
                ```
                See the <a href="https://qmcpy.readthedocs.io/en/latest/algorithms.html#module-qmcpy.discrete_distribution.digital_net_b2.digital_net_b2" target="_blank">`qmcpy.DigitalNetB2` docs</a> for more info
            n (int): number of digital sequence points to generate
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
            compile_fts (bool): if `True`, use `torch.compile(qmcpy.fwht_torch,**compile_fts_kwargs)`, otherwise use the uncompiled version
            compile_fts_kwargs (dict): keyword arguments to `torch.compile`, see the `compile_fts` argument
        """
        assert isinstance(alpha,int) and alpha in qmcpy.kernel_methods.util.dig_shift_invar_ops.WEIGHTEDWALSHFUNCSPOS.keys(), "alpha must be in %s"%list(qmcpy.kernel_methods.util.dig_shift_invar_ops.WEIGHTEDWALSHFUNCSPOS.keys())
        if isinstance(seq,int):
            seq = qmcpy.DigitalNetB2(seq)
        assert isinstance(seq,qmcpy.DigitalNetB2) and seq.order=="NATURAL" and seq.replications==1 and seq.t_lms<64 and seq.randomize in ['LMS_DS','DS','LMS','FALSE'], "seq should be a qmcpy.DigitalNetB2 instance with order='NATURAL', replications=1, t_lms<64, and randomize in ['LMS_DS','DS','LMS','FALSE']"
        self.t = seq.t_lms
        ft = ift = torch.compile(qmcpy.fwht_torch,**compile_fts_kwargs) if compile_fts else qmcpy.fwht_torch
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
        return 1
    def _sample(self, n_min, n_max):
        _x = torch.from_numpy(self.seq.gen_samples(n_min=n_min,n_max=n_max,return_binary=True).astype(np.int64)).to(self.device)
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