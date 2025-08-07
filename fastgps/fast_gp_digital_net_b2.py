from .abstract_fast_gp import AbstractFastGP
import torch
import numpy as np
import qmcpy
from typing import Tuple,Union

class FastGPDigitalNetB2(AbstractFastGP):
    """
    Fast Gaussian process regression using digitally shifted digital nets paired with digitally shift invariant kernels
    
    Examples:
        >>> device = "cpu"
        >>> if device!="mps":
        ...     torch.set_default_dtype(torch.float64)

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
        >>> fgp = FastGPDigitalNetB2(
        ...     qmcpy.KernelDigShiftInvar(d,torchify=True,device=device),
        ...     qmcpy.DigitalNetB2(dimension=d,seed=7))
        >>> x_next = fgp.get_x_next(n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)

        >>> rng = torch.Generator().manual_seed(17)
        >>> x = torch.rand((2**7,d),generator=rng).to(device)
        >>> y = f_ackley(x)
        
        >>> pmean = fgp.post_mean(x)
        >>> pmean.shape
        torch.Size([128])
        >>> torch.linalg.norm(y-pmean)/torch.linalg.norm(y)
        tensor(0.0319)
        >>> torch.allclose(fgp.post_mean(fgp.x),fgp.y)
        True

        >>> data = fgp.fit(verbose=0)
        >>> list(data.keys())
        []

        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0308)
        >>> z = torch.rand((2**8,d),generator=rng).to(device)
        >>> pcov = fgp.post_cov(x,z)
        >>> pcov.shape
        torch.Size([128, 256])

        >>> pcov = fgp.post_cov(x,x)
        >>> pcov.shape
        torch.Size([128, 128])
        >>> (pcov.diagonal()>=0).all()
        tensor(True)

        >>> pvar = fgp.post_var(x)
        >>> pvar.shape
        torch.Size([128])
        >>> torch.allclose(pcov.diagonal(),pvar,atol=1e-5)
        True

        >>> pmean,pstd,q,ci_low,ci_high = fgp.post_ci(x,confidence=0.99)
        >>> ci_low.shape
        torch.Size([128])
        >>> ci_high.shape
        torch.Size([128])

        >>> fgp.post_cubature_mean()
        tensor(20.1841)
        >>> fgp.post_cubature_var()
        tensor(0.0002)

        >>> pcmean,pcvar,q,pcci_low,pcci_high = fgp.post_cubature_ci(confidence=0.99)
        >>> pcci_low
        tensor(20.1494)
        >>> pcci_high
        tensor(20.2188)
        
        >>> pcov_future = fgp.post_cov(x,z,n=2*n)
        >>> pvar_future = fgp.post_var(x,n=2*n)
        >>> pcvar_future = fgp.post_cubature_var(n=2*n)
        
        >>> x_next = fgp.get_x_next(2*n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0245)

        >>> torch.allclose(fgp.post_cov(x,z),pcov_future)
        True
        >>> torch.allclose(fgp.post_var(x),pvar_future)
        True
        >>> torch.allclose(fgp.post_cubature_var(),pcvar_future)
        True

        >>> data = fgp.fit(verbose=False)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0247)

        >>> x_next = fgp.get_x_next(4*n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0149)

        >>> data = fgp.fit(verbose=False)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0131)

        >>> pcov_16n = fgp.post_cov(x,z,n=16*n)
        >>> pvar_16n = fgp.post_var(x,n=16*n)
        >>> pcvar_16n = fgp.post_cubature_var(n=16*n)
        >>> x_next = fgp.get_x_next(16*n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)
        >>> torch.allclose(fgp.post_cov(x,z),pcov_16n)
        True
        >>> torch.allclose(fgp.post_var(x),pvar_16n)
        True
        >>> torch.allclose(fgp.post_cubature_var(),pcvar_16n)
        True

        Different loss metrics for fitting 
        
        >>> n = 2**6
        >>> d = 3
        >>> sgp = FastGPDigitalNetB2(
        ...     qmcpy.KernelDigShiftInvar(d,torchify=True,device=device),
        ...     qmcpy.DigitalNetB2(dimension=d,seed=7))
        >>> x_next = sgp.get_x_next(n)
        >>> y_next = torch.stack([torch.sin(x_next).sum(-1),torch.cos(x_next).sum(-1)],axis=0)
        >>> sgp.add_y_next(y_next)
        >>> data = sgp.fit(loss_metric="MLL",iterations=5,verbose=0)
        >>> data = sgp.fit(loss_metric="CV",iterations=5,verbose=0,cv_weights=1/torch.arange(1,2*n+1,device=device).reshape((2,n)))
        >>> data = sgp.fit(loss_metric="CV",iterations=5,verbose=0,cv_weights="L2R")
        >>> data = sgp.fit(loss_metric="GCV",iterations=5,verbose=0)
    """
    def __init__(self,
            kernel:qmcpy.KernelDigShiftInvar,
            seqs:Union[qmcpy.DigitalNetB2,int],
            alpha:int = 2,
            noise:float = 2*qmcpy.util.transforms.EPS64,
            tfs_noise:Tuple[callable,callable] = (qmcpy.util.transforms.tf_exp_eps_inv,qmcpy.util.transforms.tf_exp_eps),
            requires_grad_noise:bool = False, 
            shape_noise:torch.Size = torch.Size([1]),
            derivatives:list = None,
            derivatives_coeffs:list = None,
            adaptive_nugget:bool = False,
            ):
        """
        Args:
            kernel (qmcpy.KernelDigShiftInvar): Kernel object. Set to `qmcpy.KernelMultiTask` for a multi-task GP.
            seqs (Union[int,qmcpy.DigitalNetB2,List]]): list of digital sequence generators in base $b=2$ 
                with order="RADICAL INVERSE" and randomize in `["FALSE","DS"]`. If an int `seed` is passed in we use 
                ```python
                [qmcpy.DigitalNetB2(d,seed=seed_i,randomize="DS") for seed_i in np.random.SeedSequence(seed).spawn(num_tasks)]
                ```
                See the <a href="https://qmcpy.readthedocs.io/en/latest/algorithms.html#module-qmcpy.discrete_distribution.digital_net_b2.digital_net_b2" target="_blank">`qmcpy.DigitalNetB2` docs</a> for more info. 
                If `num_tasks==1` then randomize may be in `["FALSE","DS","LMS","LMS DS"]`. 
            alpha (int): smoothness parameter
            noise (float): positive noise variance i.e. nugget term
            tfs_noise (Tuple[callable,callable]): the first argument transforms to the raw value to be optimized, the second applies the inverse transform
            requires_grad_noise (bool): wheather or not to optimize the noise parameter
            shape_noise (torch.Size): shape of the noise parameter, defaults to `torch.Size([1])`
            derivatives (list): list of derivative orders e.g. to include a function and its gradient set 
                ```python
                derivatives = [torch.zeros(d,dtype=int)]+[ej for ej in torch.eye(d,dtype=int)]
                ```
            derivatives_coeffs (list): list of derivative coefficients where if `derivatives[k].shape==(p,d)` then we should have `derivatives_coeffs[k].shape==(p,)`
            adaptive_nugget (bool): if True, use the adaptive nugget which modifies noises based on trace ratios.  
        """
        self._XBDTYPE = torch.int64
        self._FTOUTDTYPE = torch.get_default_dtype()
        if isinstance(kernel,qmcpy.KernelMultiTask):
            solo_task = False
            num_tasks = kernel.num_tasks
            default_task = torch.arange(num_tasks)
        else:
            solo_task = True
            default_task = 0 
            num_tasks = 1
        if isinstance(seqs,int):
            global_seed = seqs
            seqs = np.array([qmcpy.DigitalNetB2(kernel.d,seed=seed,randomize="DS") for seed in np.random.SeedSequence(global_seed).spawn(num_tasks)],dtype=object)
        if isinstance(seqs,qmcpy.DigitalNetB2):
            seqs = np.array([seqs],dtype=object)
        if isinstance(seqs,list):
            seqs = np.array(seqs,dtype=object)
        assert seqs.shape==(num_tasks,), "seqs should be a length num_tasks=%d list"%num_tasks
        assert all(isinstance(seqs[i],qmcpy.DigitalNetB2) for i in range(num_tasks)), "each seq should be a qmcpy.DigitalNetB2 instances"
        assert all(seqs[i].order=="RADICAL INVERSE" for i in range(num_tasks)), "each seq should be in 'RADICAL INVERSE' order "
        assert all(seqs[i].replications==1 for i in range(num_tasks)) and "each seq should have only 1 replication"
        if num_tasks==1:
            assert seqs[0].randomize in ['FALSE','DS','LMS','LMS DS'], "seq should have randomize in ['FALSE','DS','LMS','LMS DS']"
        else:
            assert all(seqs[i].randomize in ['FALSE','DS'] for i in range(num_tasks)), "each seq should have randomize in ['FALSE','DS']"
        ts = torch.tensor([seqs[i].t for i in range(num_tasks)])
        assert (ts<64).all(), "each seq must have t<64"
        assert (ts==ts[0]).all(), "all seqs should have the same t"
        self.t = ts[0].item()
        if isinstance(kernel,qmcpy.KernelMultiTask):
            kernel.base_kernel.set_t(self.t)
        else:
            kernel.set_t(self.t)
        ift = ft = qmcpy.fwht_torch
        omega = qmcpy.omega_fwht_torch
        super().__init__(
            alpha,
            ft,
            ift,
            omega,
            kernel,
            seqs,
            num_tasks,
            default_task,
            solo_task,
            noise,
            tfs_noise,
            requires_grad_noise,
            shape_noise,
            derivatives,
            derivatives_coeffs,
            adaptive_nugget,
        )
        assert (1<=self.alpha).all() and (self.alpha<=4).all()
    def _sample(self, seq, n_min, n_max):
        _x = torch.from_numpy(seq(n_min=int(n_min),n_max=int(n_max),return_binary=True).astype(np.int64)).to(self.device)
        x = qmcpy.util.dig_shift_invar_ops.to_float(_x,self.t)
        return x,_x
    def _kernel_parts_from_delta(self, delta, beta, kappa):
        assert delta.size(-1)==self.d and beta.shape==(self.d,) and kappa.shape==(self.d,)
        beta_plus_kappa = beta+kappa
        ind = (beta_plus_kappa>0).to(torch.int64)
        order = self.alpha-beta_plus_kappa
        assert (1<=order).all() and (order<=4).all(), "order must all be between 2 and 4, but got order = %s. Try increasing alpha"%str(order)
        omega = torch.empty(delta.shape,device=self.device) 
        for j in range(self.d):
            if order[j]==1:
                omega[...,j] = 6*(1/6-2**(torch.log2(delta[...,j]).floor()-self.t-1))
            else:
                omega[...,j] = qmcpy.kernel_methods.weighted_walsh_funcs(order[j].item(),delta[...,j],self.t)-1
        return (-2)**beta_plus_kappa*(ind+omega)
