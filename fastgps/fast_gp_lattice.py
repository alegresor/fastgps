from .abstract_fast_gp import AbstractFastGP
import torch 
import qmcpy
import numpy as np
from typing import Tuple,Union

class FastGPLattice(AbstractFastGP):
    """
    Fast Gaussian process regression using lattice points and shift invariant kernels

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
        >>> fgp = FastGPLattice(
        ...     qmcpy.KernelShiftInvar(d,torchify=True,device=device),
        ...     seqs = qmcpy.Lattice(dimension=d,seed=7))
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
        tensor(0.0348)
        >>> torch.allclose(fgp.post_mean(fgp.x),fgp.y,atol=1e-3)
        True

        >>> fgp.post_cubature_mean()
        tensor(20.1842)
        >>> fgp.post_cubature_var()
        tensor(6.9917e-09)

        >>> data = fgp.fit(verbose=0)
        >>> list(data.keys())
        []

        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0361)
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
        >>> torch.allclose(pcov.diagonal(),pvar)
        True

        >>> pmean,pstd,q,ci_low,ci_high = fgp.post_ci(x,confidence=0.99)
        >>> ci_low.shape
        torch.Size([128])
        >>> ci_high.shape
        torch.Size([128])

        >>> fgp.post_cubature_mean()
        tensor(20.1842)
        >>> fgp.post_cubature_var()
        tensor(3.1129e-06)

        >>> pcmean,pcvar,q,pcci_low,pcci_high = fgp.post_cubature_ci(confidence=0.99)
        >>> pcci_low
        tensor(20.1797)
        >>> pcci_high
        tensor(20.1888)

        >>> pcov_future = fgp.post_cov(x,z,n=2*n)
        >>> pvar_future = fgp.post_var(x,n=2*n)
        >>> pcvar_future = fgp.post_cubature_var(n=2*n)

        >>> x_next = fgp.get_x_next(2*n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0304)

        >>> torch.allclose(fgp.post_cov(x,z),pcov_future)
        True
        >>> torch.allclose(fgp.post_var(x),pvar_future)
        True
        >>> torch.allclose(fgp.post_cubature_var(),pcvar_future)
        True

        >>> data = fgp.fit(verbose=False)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0274)

        >>> x_next = fgp.get_x_next(4*n)
        >>> y_next = f_ackley(x_next)
        >>> fgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0277)

        >>> data = fgp.fit(verbose=False)
        >>> torch.linalg.norm(y-fgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0276)

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
        >>> sgp = FastGPLattice(
        ...     qmcpy.KernelShiftInvar(d,torchify=True,device=device),
        ...     qmcpy.Lattice(dimension=d,seed=7))
        >>> x_next = sgp.get_x_next(n)
        >>> y_next = torch.stack([torch.sin(x_next).sum(-1),torch.cos(x_next).sum(-1)],axis=0)
        >>> sgp.add_y_next(y_next)
        >>> data = sgp.fit(loss_metric="MLL",iterations=5,verbose=0)
        >>> data = sgp.fit(loss_metric="CV",iterations=5,verbose=0,cv_weights=1/torch.arange(1,2*n+1,device=device).reshape((2,n)))
        >>> data = sgp.fit(loss_metric="CV",iterations=5,verbose=0,cv_weights="L2R")
        >>> data = sgp.fit(loss_metric="GCV",iterations=5,verbose=0)
    """
    def __init__(self,
            kernel:qmcpy.KernelShiftInvar,
            seqs:qmcpy.Lattice,
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
            kernel (qmcpy.KernelShiftInvar): Kernel object. Set to `qmcpy.KernelMultiTask` for a multi-task GP.
            seqs ([int,qmcpy.Lattice,List]): list of lattice sequence generators
                with order="RADICAL INVERSE" and randomize in `["FALSE","SHIFT"]`. If an int `seed` is passed in we use 
                ```python
                [qmcpy.Lattice(d,seed=seed_i,randomize="SHIFT") for seed_i in np.random.SeedSequence(seed).spawn(num_tasks)]
                ```
                See the <a href="https://qmcpy.readthedocs.io/en/latest/algorithms.html#module-qmcpy.discrete_distribution.lattice.lattice" target="_blank">`qmcpy.Lattice` docs</a> for more info
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
        self._XBDTYPE = torch.get_default_dtype()
        self._FTOUTDTYPE = torch.complex64 if torch.get_default_dtype()==torch.float32 else torch.complex128
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
            seqs = np.array([qmcpy.Lattice(kernel.d,seed=seed,randomize="SHIFT") for seed in np.random.SeedSequence(global_seed).spawn(num_tasks)],dtype=object)
        if isinstance(seqs,qmcpy.Lattice):
            seqs = np.array([seqs],dtype=object)
        if isinstance(seqs,list):
            seqs = np.array(seqs,dtype=object)
        assert seqs.shape==(num_tasks,), "seqs should be a length num_tasks=%d list"%num_tasks
        assert all(isinstance(seqs[i],qmcpy.Lattice) for i in range(num_tasks)), "each seq should be a qmcpy.Lattice instances"
        assert all(seqs[i].order=="RADICAL INVERSE" for i in range(num_tasks)), "each seq should be in 'RADICAL INVERSE' order "
        assert all(seqs[i].replications==1 for i in range(num_tasks)) and "each seq should have only 1 replication"
        assert all(seqs[i].randomize in ['FALSE','SHIFT'] for i in range(num_tasks)), "each seq should have randomize in ['FALSE','SHIFT']"
        ft = qmcpy.fftbr_torch
        ift = qmcpy.ifftbr_torch
        omega = qmcpy.omega_fftbr_torch
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
    def _kernel_parts_from_delta(self, delta, beta, kappa):
        assert delta.size(-1)==self.d and beta.shape==(self.d,) and kappa.shape==(self.d,)
        beta_plus_kappa = beta+kappa
        order = 2*self.alpha-beta_plus_kappa
        assert (2<=order).all(), "order must all be at least 2, but got order = %s"%str(order)
        coeff = (-1)**(self.alpha+kappa+1)*torch.exp(2*self.alpha*np.log(2*np.pi)-torch.lgamma(order+1))
        return coeff*torch.stack([qmcpy.kernel_methods.bernoulli_poly(order[j].item(),delta[...,j]) for j in range(self.d)],-1)
