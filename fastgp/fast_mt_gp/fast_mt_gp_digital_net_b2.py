from ._fast_mt_gp import _FastMultiTaskGP
import torch
import numpy as np
import qmcpy as qmcpy
from typing import Tuple,Union

class FastMultiTaskGPDigitalNetB2(_FastMultiTaskGP):
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

        >>> n = 2**10
        >>> d = 2
        >>> fgp = FastMultiTaskGPDigitalNetB2(qmcpy.DigitalNetB2(dimension=d,seed=7))
        >>> x_next = fgp.get_x_next(n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)

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
        
        >>> pcov_future = fgp.post_cov(x,z,n=2*n)
        >>> pvar_future = fgp.post_var(x,n=2*n)
        >>> pcvar_future = fgp.post_cubature_var(n=2*n)
        
        >>> x_next = fgp.get_x_next(2*n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0258)

        >>> assert torch.allclose(fgp.post_cov(x,z),pcov_future)
        >>> assert torch.allclose(fgp.post_var(x),pvar_future)
        >>> assert torch.allclose(fgp.post_cubature_var(),pcvar_future)

        >>> data = fgp.fit(verbose=False,store_mll_hist=False,store_scale_hist=False,store_lengthscales_hist=False,store_noise_hist=False)
        >>> assert len(data)==0
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0258)

        >>> x_next = fgp.get_x_next(4*n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0192)

        >>> data = fgp.fit(verbose=False,store_mll_hist=False,store_scale_hist=False,store_lengthscales_hist=False,store_noise_hist=False)
        >>> assert len(data)==0
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0187)

        >>> pcov_16n = fgp.post_cov(x,z,n=16*n)
        >>> pvar_16n = fgp.post_var(x,n=16*n)
        >>> pcvar_16n = fgp.post_cubature_var(n=16*n)
        >>> x_next = fgp.get_x_next(16*n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)
        >>> assert torch.allclose(fgp.post_cov(x,z),pcov_16n)
        >>> assert torch.allclose(fgp.post_var(x),pvar_16n)
        >>> assert torch.allclose(fgp.post_cubature_var(),pcvar_16n)
    """
    _XBDTYPE = torch.int64
    _FTOUTDTYPE = torch.float64
    def __init__(self,
            seqs:Union[qmcpy.DigitalNetB2,int],
            num_tasks:int,
            seed_for_seq:int = None,
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
            seqs (Union[qmcpy.DigitalNetB2,int]): digital sequence generator in base $b=2$ with order="NATURAL" and randomize in ["LMS_DS","DS","LMS","FALSE"], where if an int `d` is passed in we use 
                ```python
                qmcpy.DigitalNetB2(d)
                ```
                See the <a href="https://qmcpy.readthedocs.io/en/latest/algorithms.html#module-qmcpy.discrete_distribution.digital_net_b2.digital_net_b2" target="_blank">`qmcpy.DigitalNetB2` docs</a> for more info
            num_tasks (int): number of tasks 
            seed_for_seq (int): seed used for digital net randomization
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
        assert isinstance(num_tasks,int) and num_tasks>0
        if isinstance(seqs,int):
            np_seed_seqs = np.random.SeedSequence(seed_for_seq)
            seeds = np_seed_seqs.spawn(num_tasks)
            seqs = [qmcpy.DigitalNetB2(seqs,seed=seeds[i]) for i in range(num_tasks)]
        if isinstance(seqs,list):
            seqs = np.array(seqs,dtype=object)
        assert seqs.shape==(num_tasks,), "seqs should be a length num_tasks=%d list"%num_tasks
        assert all(isinstance(seqs[i],qmcpy.DigitalNetB2) for i in range(num_tasks)), "each seq should be a qmcpy.DigitalNetB2 instances"
        assert all(seqs[i].order=="NATURAL" for i in range(num_tasks)), "each seq should be in 'NATURAL' order "
        assert all(seqs[i].replications==1 for i in range(num_tasks)) and "each seq should have only 1 replication"
        assert all(seqs[i].t_lms<64 for i in range(num_tasks)), "each seq must have t_lms<64"
        assert all(seqs[i].randomize in ['LMS_DS','DS','LMS','FALSE'] for i in range(num_tasks)), "each seq should have randomize in ['LMS_DS','DS','LMS','FALSE']"
        assert all(seqs[i].t_lms==seqs[0].t_lms for i in range(num_tasks)), "all seqs should have the same t_lms"
        self.t = seqs[0].t_lms 
        ft = ift = torch.compile(qmcpy.fwht_torch,**compile_fts_kwargs) if compile_fts else qmcpy.fwht_torch
        super().__init__(
            seqs,
            num_tasks,
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
    def _sample(self, seq, n_min, n_max):
        _x = torch.from_numpy(seq.gen_samples(n_min=n_min,n_max=n_max,return_binary=True).astype(np.int64)).to(self.device)
        x = self._convert_from_b(_x)
        return x,_x
    def _convert_to_b(self, x):
        return torch.floor((x%1)*2**(self.t)).to(self._XBDTYPE)
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