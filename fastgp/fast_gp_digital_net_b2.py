from .abstract_fast_gp import AbstractFastGP
import torch
import numpy as np
import qmcpy as qmcpy
from typing import Tuple,Union

class FastGPDigitalNetB2(AbstractFastGP):
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
        >>> fgp = FastGPDigitalNetB2(qmcpy.DigitalNetB2(dimension=d,seed=7))
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

        >>> data = fgp.fit(verbose=0)
        >>> list(data.keys())
        ['iterations']

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

        >>> pcmean,pcvar,q,pcci_low,pcci_high = fgp.post_cubature_ci(confidence=0.99)
        >>> pcci_low
        tensor(20.1564)
        >>> pcci_high
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

        >>> data = fgp.fit(verbose=False)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0259)

        >>> x_next = fgp.get_x_next(4*n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0191)

        >>> data = fgp.fit(verbose=False)
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
            num_tasks:int = None,
            seed_for_seq:int = None,
            alpha:int = 2,
            scale:float = 1., 
            lengthscales:Union[torch.Tensor,float] = 1., 
            noise:float = 1e-16,
            factor_task_kernel:Union[torch.Tensor,int] = 1.,
            rank_factor_task_kernel:int = None,
            noise_task_kernel:Union[torch.Tensor,float] = 1.,
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
            shape_batch:torch.Size = torch.Size([]),
            shape_scale:torch.Size = torch.Size([1]), 
            shape_lengthscales:torch.Size = None,
            shape_noise:torch.Size = torch.Size([1]),
            shape_factor_task_kernel:torch.Size = None, 
            shape_noise_task_kernel:torch.Size = None,
            derivatives:list = None,
            derivatives_coeffs:list = None,
            compile_fts:bool = False,
            compile_fts_kwargs: dict = {},
            adaptive_nugget:bool = False,
            ):
        """
        Args:
            seqs (Union[int,qmcpy.DigitalNetB2,List]]): list of digital sequence generators in base $b=2$ 
                with order="NATURAL" and randomize in `["FALSE","DS"]`. If an int `d` is passed in we use 
                ```python
                [qmcpy.DigitalNetB2(d,seed=seed,randomize="DS") for seed in np.random.SeedSequence(seed_for_seq).spawn(num_tasks)]
                ```
                See the <a href="https://qmcpy.readthedocs.io/en/latest/algorithms.html#module-qmcpy.discrete_distribution.digital_net_b2.digital_net_b2" target="_blank">`qmcpy.DigitalNetB2` docs</a> for more info. 
                If `num_tasks==1` then randomize may be in `["FALSE","DS","LMS","LMS_DS"]`. 
            num_tasks (int): number of tasks 
            seed_for_seq (int): seed used for digital net randomization
            alpha (int): smoothness parameter
            scale (float): kernel global scaling parameter
            lengthscales (Union[torch.Tensor[d],float]): vector of kernel lengthscales. 
                If a scalar is passed in then `lengthscales` is set to a constant vector. 
            noise (float): positive noise variance i.e. nugget term
            factor_task_kernel (Union[Tensor[num_tasks,rank_factor_task_kernel],int]): for $F$ the `factor_task_kernel` the task kernel is $FF^T + \\text{diag}(\\boldsymbol{v})$ 
                where `rank_factor_task_kernel<=num_tasks` and $\\boldsymbol{v}$ is the `noise_task_kernel`.
            rank_factor_task_kernel (int): see the description of `factor_task_kernel` above. Defaults to 0 for single task problems and 1 for multi task problems.
            noise_task_kernel (Union[torch.Tensor[num_tasks],float]): see the description of `factor_task_kernel` above 
            device (torch.device): torch device which is required to support `torch.float64`
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
            shape_batch (torch.Size): shape of the batch output for each task
            shape_scale (torch.Size): shape of the scale parameter, defaults to `torch.Size([1])`
            shape_lengthscales (torch.Size): shape of the lengthscales parameter, defaults to `torch.Size([d])` where `d` is the dimension
            shape_noise (torch.Size): shape of the noise parameter, defaults to `torch.Size([1])`
            shape_factor_task_kernel (torch.Size): shape of the factor for the task kernel, defaults to `torch.Size([num_tasks,r])` where `r` is the rank, see the description of `factor_task_kernel`
            shape_noise_task_kernel (torch.Size): shape of the noise for the task kernel, defaults to `torch.Size([num_tasks])`
            derivatives (list): list of derivative orders e.g. to include a function and its gradient set 
                ```python
                derivatives = [torch.zeros(d,dtype=int)]+[ej for ej in torch.eye(d,dtype=int)]
                ```
            derivatives_coeffs (list): list of derivative coefficients where if `derivatives[k].shape==(p,d)` then we should have `derivatives_coeffs[k].shape==(p,)`
            compile_fts (bool): if `True`, use `torch.compile(qmcpy.fwht_torch,**compile_fts_kwargs)`, otherwise use the uncompiled version
            compile_fts_kwargs (dict): keyword arguments to `torch.compile`, see the `compile_fts` argument
            adaptive_nugget (bool): if True, use the adaptive nugget which modifies noises based on trace ratios.  
        """
        assert isinstance(alpha,int) and alpha in qmcpy.kernel_methods.util.dig_shift_invar_ops.WEIGHTEDWALSHFUNCSPOS.keys(), "alpha must be in %s"%list(qmcpy.kernel_methods.util.dig_shift_invar_ops.WEIGHTEDWALSHFUNCSPOS.keys())
        if num_tasks is None: 
            solo_task = True
            default_task = 0 
            num_tasks = 1
        else:
            assert isinstance(num_tasks,int) and num_tasks>0
            solo_task = False
            default_task = torch.arange(num_tasks)
        if isinstance(seqs,int):
            seqs = np.array([qmcpy.DigitalNetB2(seqs,seed=seed,randomize="DS") for seed in np.random.SeedSequence(seed_for_seq).spawn(num_tasks)],dtype=object)
        if isinstance(seqs,qmcpy.DigitalNetB2):
            seqs = np.array([seqs],dtype=object)
        if isinstance(seqs,list):
            seqs = np.array(seqs,dtype=object)
        assert seqs.shape==(num_tasks,), "seqs should be a length num_tasks=%d list"%num_tasks
        assert all(isinstance(seqs[i],qmcpy.DigitalNetB2) for i in range(num_tasks)), "each seq should be a qmcpy.DigitalNetB2 instances"
        assert all(seqs[i].order=="NATURAL" for i in range(num_tasks)), "each seq should be in 'NATURAL' order "
        assert all(seqs[i].replications==1 for i in range(num_tasks)) and "each seq should have only 1 replication"
        assert all(seqs[i].t_lms<64 for i in range(num_tasks)), "each seq must have t_lms<64"
        if num_tasks==1:
            assert seqs[0].randomize in ['FALSE','DS','LMS','LMS_DS'], "seq should have randomize in ['FALSE','DS','LMS','LMS_DS']"
        else:
            assert all(seqs[i].randomize in ['FALSE','DS'] for i in range(num_tasks)), "each seq should have randomize in ['FALSE','DS']"
        assert all(seqs[i].t_lms==seqs[0].t_lms for i in range(num_tasks)), "all seqs should have the same t_lms"
        self.t = seqs[0].t_lms
        ift = ft = torch.compile(qmcpy.fwht_torch,**compile_fts_kwargs) if compile_fts else qmcpy.fwht_torch
        super().__init__(
            alpha,
            ft,
            ift,
            seqs,
            num_tasks,
            default_task,
            solo_task,
            scale,
            lengthscales,
            noise,
            factor_task_kernel,
            rank_factor_task_kernel,
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
            shape_batch,
            shape_scale, 
            shape_lengthscales,
            shape_noise,
            shape_factor_task_kernel, 
            shape_noise_task_kernel,
            derivatives,
            derivatives_coeffs,
            adaptive_nugget,
        )
        assert (self.alpha<=4).all() and (self.alpha>=2).all()
    def get_omega(self, m):
        return 1
    def _sample(self, seq, n_min, n_max):
        _x = torch.from_numpy(seq.gen_samples(n_min=int(n_min),n_max=int(n_max),return_binary=True).astype(np.int64)).to(self.device)
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
    def _kernel_parts_from_delta(self, delta, beta, kappa):
        assert delta.size(-1)==self.d and beta.shape==(self.d,) and kappa.shape==(self.d,)
        beta_plus_kappa = beta+kappa
        ind = (beta_plus_kappa>0).to(torch.int64)
        order = self.alpha-beta_plus_kappa
        assert (2<=order).all() and (order<=4).all(), "order must all be between 2 and 4, but got order = %s. Try increasing alpha"%str(order)
        return (-2)**beta_plus_kappa*(ind+torch.stack([qmcpy.kernel_methods.weighted_walsh_funcs(order[j].item(),delta[...,j],self.t)-1 for j in range(self.d)],-1))
