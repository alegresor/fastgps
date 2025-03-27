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

        >>> n = 2**10
        >>> d = 2
        >>> fgp = FastGPLattice(seqs = qmcpy.Lattice(dimension=d,seed=7))
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
        tensor(0.0348)
        >>> assert torch.allclose(fgp.post_mean(fgp.x),fgp.y,atol=1e-3)

        >>> fgp.post_cubature_mean()
        tensor(20.1842)
        >>> fgp.post_cubature_var()
        tensor(7.0015e-09)

        >>> data = fgp.fit()
             iter of 5.0e+03 | NMLL       | noise      | scale      | lengthscales         | task_kernel 
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    0.00e+00 | 3.70e+05   | 1.00e-08   | 1.00e+00   | [1.0e+00 1.0e+00]    | [[1.0e+00]]
                    5.00e+00 | 4.12e+04   | 1.00e-08   | 2.10e+00   | [2.1e+00 2.1e+00]    | [[1.0e+00]]
                    1.00e+01 | 2.81e+03   | 1.00e-08   | 8.00e+00   | [8.0e+00 8.0e+00]    | [[1.0e+00]]
                    1.50e+01 | 2.82e+03   | 1.00e-08   | 8.18e+00   | [8.2e+00 8.2e+00]    | [[1.0e+00]]
                    2.00e+01 | 2.80e+03   | 1.00e-08   | 7.73e+00   | [7.7e+00 7.7e+00]    | [[1.0e+00]]
                    2.50e+01 | 2.80e+03   | 1.00e-08   | 7.74e+00   | [7.7e+00 7.6e+00]    | [[1.0e+00]]
                    2.60e+01 | 2.80e+03   | 1.00e-08   | 7.74e+00   | [7.7e+00 7.6e+00]    | [[1.0e+00]]
        >>> list(data.keys())
        ['mll_hist', 'scale_hist', 'lengthscales_hist']

        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0361)
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
        tensor(20.1842)
        >>> fgp.post_cubature_var()
        tensor(3.1823e-06)

        >>> pcmean,pcvar,q,cci_low,cci_high = fgp.post_cubature_ci(confidence=0.99)
        >>> cci_low
        tensor(20.1796)
        >>> cci_high
        tensor(20.1888)

        >>> pcov_future = fgp.post_cov(x,z,n=2*n)
        >>> pvar_future = fgp.post_var(x,n=2*n)
        >>> pcvar_future = fgp.post_cubature_var(n=2*n)

        >>> x_next = fgp.get_x_next(2*n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0305)

        >>> assert torch.allclose(fgp.post_cov(x,z),pcov_future)
        >>> assert torch.allclose(fgp.post_var(x),pvar_future)
        >>> assert torch.allclose(fgp.post_cubature_var(),pcvar_future)

        >>> data = fgp.fit(verbose=False,store_mll_hist=False,store_scale_hist=False,store_lengthscales_hist=False,store_noise_hist=False)
        >>> assert len(data)==0
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0274)

        >>> x_next = fgp.get_x_next(4*n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0277)

        >>> data = fgp.fit(verbose=False,store_mll_hist=False,store_scale_hist=False,store_lengthscales_hist=False,store_noise_hist=False)
        >>> assert len(data)==0
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0276)

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
    _XBDTYPE = torch.float64
    _FTOUTDTYPE = torch.complex128
    def __init__(self,
            seqs:qmcpy.Lattice,
            num_tasks:int = None,
            seed_for_seq:int = None,
            alpha:int = 2,
            scale:float = 1., 
            lengthscales:Union[torch.Tensor,float] = 1, 
            noise:float = 1e-8, 
            factor_task_kernel:Union[torch.Tensor,int] = None, 
            noise_task_kernel:Union[torch.Tensor,float] = 1,
            device:torch.device = "cpu",
            tfs_scale:Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_lengthscales:Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_noise:Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            tfs_factor_task_kernel:Tuple[callable,callable] = ((lambda x: x, lambda x: x)),#((lambda x: x**(1/3)),(lambda x: x**3)),
            tfs_noise_task_kernel:Tuple[callable,callable] = ((lambda x: torch.log(x)),(lambda x: torch.exp(x))),
            requires_grad_scale:bool = True, 
            requires_grad_lengthscales:bool = True, 
            requires_grad_noise:bool = False, 
            requires_grad_factor_task_kernel:bool = None,
            requires_grad_noise_task_kernel:bool = None,
            compile_fts:bool = False,
            compile_fts_kwargs:dict = {},
            ):
        """
        Args:
            seqs (Union[List[num_tasks] of qmcpy.Lattice],int]): list of lattice sequence generators
                with order="NATURAL" and randomize in `["FALSE","SHIFT"]`. If an int `d` is passed in we use 
                ```python
                [qmcpy.Lattice(d,seed=seed,randomize="SHIFT") for seed in np.random.SeedSequence(seed_for_seq).spawn(num_tasks)]
                ```
                See the <a href="https://qmcpy.readthedocs.io/en/latest/algorithms.html#module-qmcpy.discrete_distribution.lattice.lattice" target="_blank">`qmcpy.Lattice` docs</a> for more info
            num_tasks (int): number of tasks
            seed_for_seq (int): seed used for lattice randomization
            alpha (int): smoothness parameter
            scale (float): kernel global scaling parameter
            lengthscales (Union[torch.Tensor[d],float]): vector of kernel lengthscales. 
                If a scalar is passed in then `lengthscales` is set to a constant vector. 
            noise (float): positive noise variance i.e. nugget term
            factor_task_kernel (Union[Tensor[num_tasks,r],int]): for $F$ the `factor_task_kernel` the task kernel is $FF^T + \\text{diag}(\\boldsymbol{v})$ 
                where `r<num_tasks` is the rank and $\\boldsymbol{v}$ is the `noise_task_kernel`. If an int `r` is passed in $F$ is initialized to zeros. 
            noise_task_kernel (Union[torch.Tensor[num_tasks],float]): positive $\\boldsymbol{v}$ in the description of `factor_task_kernel` above. 
                If a scalar is passed in then `noise_task_kernel` is set to a constant vector.
            device (torch.device): torch device which is required to support torch.float64
            tfs_scale (Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            tfs_lengthscales (Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            tfs_noise (Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            tfs_factor_task_kernel (Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            tfs_noise_task_kernel (Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            requires_grad_scale (bool): wheather or not to optimize the scale parameter
            requires_grad_lengthscales (bool): wheather or not to optimize lengthscale parameters
            requires_grad_noise (bool): wheather or not to optimize the noise parameter
            requires_grad_factor_task_kernel (bool): wheather or not to optimize the factor for the task kernel
            requires_grad_noise_task_kernel (bool): wheather or not to optimize the noise for the task kernel
            compile_fts (bool): if `True`, use `torch.compile(qmcpy.fftbr_torch,**compile_fts)` and `torch.compile(qmcpy.ifftbr_torch,**compile_fts)`, otherwise use the uncompiled versions
            compile_fts_kwargs (dict): keyword arguments to `torch.compile`, see the `compile_fts argument`
        """
        assert isinstance(alpha,int) and alpha in qmcpy.kernel_methods.util.shift_invar_ops.BERNOULLIPOLYSDICT.keys(), "alpha must be in %s"%list(qmcpy.kernel_methods.util.shift_invar_ops.BERNOULLIPOLYSDICT.keys())
        if num_tasks is None: 
            solo_task = True
            default_task = 0 
            num_tasks = 1
        else:
            assert isinstance(num_tasks,int) and num_tasks>0
            solo_task = False
            default_task = torch.arange(num_tasks)
        if isinstance(seqs,int):
            seqs = np.array([qmcpy.Lattice(seqs,seed=seed,randomize="SHIFT") for seed in np.random.SeedSequence(seed_for_seq).spawn(num_tasks)],dtype=object)
        if isinstance(seqs,qmcpy.Lattice):
            seqs = np.array([seqs],dtype=object)
        if isinstance(seqs,list):
            seqs = np.array(seqs,dtype=object)
        assert seqs.shape==(num_tasks,), "seqs should be a length num_tasks=%d list"%num_tasks
        assert all(isinstance(seqs[i],qmcpy.Lattice) for i in range(num_tasks)), "each seq should be a qmcpy.Lattice instances"
        assert all(seqs[i].order=="NATURAL" for i in range(num_tasks)), "each seq should be in 'NATURAL' order "
        assert all(seqs[i].replications==1 for i in range(num_tasks)) and "each seq should have only 1 replication"
        assert all(seqs[i].randomize in ['FALSE','SHIFT'] for i in range(num_tasks)), "each seq should have randomize in ['FALSE','SHIFT']"
        ft = torch.compile(qmcpy.fftbr_torch,**compile_fts_kwargs) if compile_fts else qmcpy.fftbr_torch
        ift = torch.compile(qmcpy.ifftbr_torch,**compile_fts_kwargs) if compile_fts else qmcpy.ifftbr_torch
        self.__const_for_kernel = None
        super().__init__(
            seqs,
            num_tasks,
            default_task,
            solo_task,
            alpha,
            scale,
            lengthscales,
            noise,
            factor_task_kernel,
            noise_task_kernel,
            device,
            tfs_scale,
            tfs_lengthscales,
            tfs_noise,
            tfs_factor_task_kernel,
            tfs_noise_task_kernel,
            requires_grad_scale,
            requires_grad_lengthscales,
            requires_grad_noise,
            requires_grad_factor_task_kernel,
            requires_grad_noise_task_kernel,
            ft,
            ift,
        )
    def get_omega(self, m):
        return torch.exp(-torch.pi*1j*torch.arange(2**m,device=self.device)/2**m)
    def _sample(self, seq, n_min, n_max):
        x = torch.from_numpy(seq.gen_samples(n_min=n_min,n_max=n_max)).to(torch.get_default_dtype()).to(self.device)
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
