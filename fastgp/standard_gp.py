from .abstract_fast_gp import AbstractFastGP
import torch
import numpy as np
import qmcpy as qmcpy
from typing import Tuple,Union

class StandardGP():
    """
    Standard Gaussian process regression
    
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
        >>> sgp = StandardGP(qmcpy.IIDStdUniform(dimension=d,seed=7))
        >>> x_next = sgp.get_x_next(n)
        >>> y_next = f_ackley(x_next)
        >>> sgp.add_y_next(y_next)

        >>> rng = torch.Generator().manual_seed(17)
        >>> x = torch.rand((2**7,d),generator=rng)
        >>> y = f_ackley(x)
        
        >>> pmean = sgp.post_mean(x)
        >>> pmean.shape
        torch.Size([128])
        >>> torch.linalg.norm(y-pmean)/torch.linalg.norm(y)
        tensor(0.0336)
        >>> assert torch.allclose(sgp.post_mean(sgp.x),sgp.y)

        >>> data = sgp.fit()
             iter of 5.0e+03 | NMLL       | norm term  | logdet term
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    0.00e+00 | 3.35e+03   | 1.46e+03   | 1.32e+01  
                    5.00e+00 | 3.24e+03   | 1.05e+03   | 3.11e+02  
                    1.00e+01 | 3.13e+03   | 9.55e+02   | 2.96e+02  
                    1.50e+01 | 3.10e+03   | 1.03e+03   | 1.86e+02  
                    2.00e+01 | 3.10e+03   | 1.01e+03   | 2.02e+02  
                    2.50e+01 | 3.09e+03   | 1.04e+03   | 1.71e+02  
                    3.00e+01 | 3.09e+03   | 1.03e+03   | 1.75e+02  
                    3.50e+01 | 3.09e+03   | 1.04e+03   | 1.72e+02  
                    4.00e+01 | 3.09e+03   | 1.03e+03   | 1.80e+02  
                    4.30e+01 | 3.09e+03   | 1.03e+03   | 1.84e+02  
        >>> list(data.keys())
        ['mll_hist', 'scale_hist', 'lengthscales_hist']

        >>> torch.linalg.norm(y-sgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0355)
        >>> z = torch.rand((2**8,d),generator=rng)
        >>> pcov = sgp.post_cov(x,z)
        >>> pcov.shape
        torch.Size([128, 256])

        >>> pcov = sgp.post_cov(x,x)
        >>> pcov.shape
        torch.Size([128, 128])
        >>> assert (pcov.diagonal()>=0).all()

        >>> pvar = sgp.post_var(x)
        >>> pvar.shape
        torch.Size([128])
        >>> assert torch.allclose(pcov.diagonal(),pvar)

        >>> pmean,pstd,q,ci_low,ci_high = sgp.post_ci(x,confidence=0.99)
        >>> ci_low.shape
        torch.Size([128])
        >>> ci_high.shape
        torch.Size([128])

        >>> sgp.post_cubature_mean()
        tensor(20.1896)
        >>> sgp.post_cubature_var()
        tensor(0.0002)

        >>> pcmean,pcvar,q,pcci_low,pcci_high = sgp.post_cubature_ci(confidence=0.99)
        >>> pcci_low
        tensor(20.1564)
        >>> pcci_high
        tensor(20.2228)
        
        >>> pcov_future = sgp.post_cov(x,z,n=2*n)
        >>> pvar_future = sgp.post_var(x,n=2*n)
        >>> pcvar_future = sgp.post_cubature_var(n=2*n)
        
        >>> x_next = sgp.get_x_next(2*n)
        >>> y_next = f_ackley(x_next)
        >>> sgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-sgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0258)

        >>> assert torch.allclose(sgp.post_cov(x,z),pcov_future)
        >>> assert torch.allclose(sgp.post_var(x),pvar_future)
        >>> assert torch.allclose(sgp.post_cubature_var(),pcvar_future)

        >>> data = sgp.fit(verbose=False,store_mll_hist=False,store_scale_hist=False,store_lengthscales_hist=False,store_noise_hist=False)
        >>> assert len(data)==0
        >>> torch.linalg.norm(y-sgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0259)

        >>> x_next = sgp.get_x_next(4*n)
        >>> y_next = f_ackley(x_next)
        >>> sgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-sgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0191)

        >>> data = sgp.fit(verbose=False,store_mll_hist=False,store_scale_hist=False,store_lengthscales_hist=False,store_noise_hist=False)
        >>> assert len(data)==0
        >>> torch.linalg.norm(y-sgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0187)

        >>> pcov_16n = sgp.post_cov(x,z,n=16*n)
        >>> pvar_16n = sgp.post_var(x,n=16*n)
        >>> pcvar_16n = sgp.post_cubature_var(n=16*n)
        >>> x_next = sgp.get_x_next(16*n)
        >>> y_next = f_ackley(x_next)
        >>> sgp.add_y_next(y_next)
        >>> assert torch.allclose(sgp.post_cov(x,z),pcov_16n)
        >>> assert torch.allclose(sgp.post_var(x),pvar_16n)
        >>> assert torch.allclose(sgp.post_cubature_var(),pcvar_16n)
    """
    _FTOUTDTYPE = torch.float64
    def __init__(self,
            seqs:Union[qmcpy.IIDStdUniform,int],
            num_tasks:int = None,
            seed_for_seq:int = None,
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
            ):
        """
        Args:
            seqs (Union[int,qmcpy.IIDStdUniform,List]]): list of digital sequence generators in base $b=2$ 
                with order="NATURAL" and randomize in `["FALSE","DS"]`. If an int `d` is passed in we use 
                ```python
                [qmcpy.IIDStdUniform(d,seed=seed,randomize="DS") for seed in np.random.SeedSequence(seed_for_seq).spawn(num_tasks)]
                ```
                See the <a href="https://qmcpy.readthedocs.io/en/latest/algorithms.html#module-qmcpy.discrete_distribution.digital_net_b2.digital_net_b2" target="_blank">`qmcpy.DigitalNetB2` docs</a> for more info. 
                If `num_tasks==1` then randomize may be in `["FALSE","DS","LMS","LMS_DS"]`. 
            num_tasks (int): number of tasks 
            seed_for_seq (int): seed used for digital net randomization
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
        """
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
        super().__init__(
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
        )
        assert (self.alpha<=4).all() and (self.alpha>=2).all()
    def fit(self,
        iterations:int = 5000,
        lr:float = 1e-1,
        optimizer:torch.optim.Optimizer = None,
        stop_crit_improvement_threshold:float = 1e-1,
        stop_crit_wait_iterations:int = 10,
        store_mll_hist:bool = True, 
        store_scale_hist:bool = True, 
        store_lengthscales_hist:bool = True,
        store_noise_hist:bool = True,
        store_task_kernel_hist:bool = True,
        verbose:int = 5,
        verbose_indent:int = 4,
        ):
        """
        Args:
            iterations (int): number of optimization iterations
            lr (float): learning rate for default optimizer
            optimizer (torch.optim.Optimizer): optimizer defaulted to `torch.optim.Rprop(self.parameters(),lr=lr)`
            stop_crit_improvement_threshold (float): stop fitting when the maximum number of iterations is reached or the best mll is note reduced by `stop_crit_improvement_threshold` for `stop_crit_wait_iterations` iterations 
            stop_crit_wait_iterations (int): number of iterations to wait for improved mll before early stopping, see the argument description for `stop_crit_improvement_threshold`
            store_mll_hist (bool): if `True`, store and return iteration data for mll
            store_scale_hist (bool): if `True`, store and return iteration data for the kernel scale parameter
            store_lengthscales_hist (bool): if `True`, store and return iteration data for the kernel lengthscale parameters
            store_noise_hist (bool): if `True`, store and return iteration data for noise
            store_task_kernel_hist (bool): if `True`, store and return iteration data for the task kernel
            verbose (int): log every `verbose` iterations, set to `0` for silent mode
            verbose_indent (int): size of the indent to be applied when logging, helpful for logging multiple models
            
        Returns:
            data (dict): iteration data which, dependeing on storage arguments, may include keys in 
                ```python
                ["mll_hist","scale_hist","lengthscales_hist","noise_hist","task_kernel_hist"]
                ```
        """
        assert (self.n>0).any(), "cannot fit without data"
        assert isinstance(iterations,int) and iterations>=0
        if optimizer is None:
            assert np.isscalar(lr) and lr>0, "require lr is a positive float"
            optimizer = torch.optim.Rprop(self.parameters(),lr=lr)
        assert isinstance(optimizer,torch.optim.Optimizer)
        assert isinstance(store_mll_hist,bool), "require bool store_mll_hist" 
        assert isinstance(store_scale_hist,bool), "require bool store_scale_hist" 
        assert isinstance(store_lengthscales_hist,bool), "require bool store_lengthscales_hist" 
        assert isinstance(store_noise_hist,bool), "require bool store_noise_hist"
        assert isinstance(store_task_kernel_hist,bool), "require bool store_task_kernel_hist"
        assert (isinstance(verbose,int) or isinstance(verbose,bool)) and verbose>=0, "require verbose is a non-negative int"
        assert isinstance(verbose_indent,int) and verbose_indent>=0, "require verbose_indent is a non-negative int"
        assert np.isscalar(stop_crit_improvement_threshold) and 0<stop_crit_improvement_threshold, "require stop_crit_improvement_threshold is a positive float"
        assert isinstance(stop_crit_wait_iterations,int) and stop_crit_wait_iterations>0
        logtol = np.log(1+stop_crit_improvement_threshold)
        if store_mll_hist:
            mll_hist = torch.empty(iterations+1)
        store_scale_hist = store_scale_hist and self.raw_scale.requires_grad
        store_lengthscales_hist = store_lengthscales_hist and self.raw_lengthscales.requires_grad
        store_noise_hist = store_noise_hist and self.raw_noise.requires_grad
        store_task_kernel_hist = store_task_kernel_hist and (self.raw_factor_task_kernel.requires_grad or self.raw_noise_task_kernel.requires_grad)
        if store_scale_hist: scale_hist = torch.empty(torch.Size([iterations+1])+self.raw_scale.shape)
        if store_lengthscales_hist: lengthscales_hist = torch.empty(torch.Size([iterations+1])+self.raw_lengthscales.shape)
        if store_noise_hist: noise_hist = torch.empty(torch.Size([iterations+1])+self.raw_noise.shape)
        if store_task_kernel_hist: task_kernel_hist = torch.empty(torch.Size([iterations+1])+self.gram_matrix_tasks.shape)
        if verbose:
            _s = "%16s | %-10s | %-10s | %-10s"%("iter of %.1e"%iterations,"NMLL","norm term","logdet term")
            print(" "*verbose_indent+_s)
            print(" "*verbose_indent+"~"*len(_s))
        mll_const = self.d_out*self.n.sum()*np.log(2*np.pi)
        stop_crit_best_mll = torch.inf 
        stop_crit_save_mll = torch.inf 
        stop_crit_iterations_without_improvement_mll = 0
        ytildes = [self.get_ytilde(i) for i in range(self.num_tasks)]
        ytildescat = torch.cat(ytildes,dim=-1)
        os.environ["FASTGP_FORCE_RECOMPILE"] = "True"
        inv_log_det_cache = self.get_inv_log_det_cache()
        for i in range(iterations+1):
            ztildes,logdet = inv_log_det_cache._gram_matrix_solve_tilde_to_tilde(ytildes)
            ztildescat = torch.cat(ztildes,dim=-1)
            norm_term = (ytildescat.conj()*ztildescat).real.sum()
            logdet_term = self.d_out/torch.tensor(logdet.shape).prod()*logdet.sum()
            mll = norm_term+logdet_term+mll_const
            if mll.item()<stop_crit_best_mll:
                stop_crit_best_mll = mll.item()
            if (stop_crit_save_mll-mll.item())>logtol:
                stop_crit_iterations_without_improvement_mll = 0
                stop_crit_save_mll = stop_crit_best_mll
            else:
                stop_crit_iterations_without_improvement_mll += 1
            break_condition = i==iterations or stop_crit_iterations_without_improvement_mll==stop_crit_wait_iterations
            if store_mll_hist: mll_hist[i] = mll.item()
            if store_scale_hist: scale_hist[i] = self.scale.detach().to(scale_hist.device)
            if store_lengthscales_hist: lengthscales_hist[i] = self.lengthscales.detach().to(lengthscales_hist.device)
            if store_noise_hist: noise_hist[i] = self.noise.detach().to(noise_hist.device)
            if store_task_kernel_hist: task_kernel_hist[i] = self.gram_matrix_tasks.detach().to(task_kernel_hist.device)
            if verbose and (i%verbose==0 or break_condition):
                _s = "%16.2e | %-10.2e | %-10.2e | %-10.2e"%(i,mll.item(),norm_term.item(),logdet_term.item())
                print(" "*verbose_indent+_s)
            if break_condition: break
            mll.backward()
            optimizer.step()
            optimizer.zero_grad()
        del os.environ["FASTGP_FORCE_RECOMPILE"]
        data = {}
        if store_mll_hist: data["mll_hist"] = mll_hist
        if store_scale_hist: data["scale_hist"] = scale_hist
        if store_lengthscales_hist: data["lengthscales_hist"] = lengthscales_hist
        if store_noise_hist: data["noise_hist"] = noise_hist
        if store_task_kernel_hist: data["task_kernel_hist"] = task_kernel_hist
        return data
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
    def post_cubature_mean(self, task:Union[int,torch.Tensor]=None, eval:bool=True):
        """
        Posterior cubature mean. 

        Args:
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`
            task (Union[int,torch.Tensor[T]]): task indices

        Returns:
            pcmean (torch.Tensor[...,T]): posterior cubature mean
        """
        kmat_tasks = self.gram_matrix_tasks
        coeffs = self.coeffs
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int)
        if isinstance(task,list): task = torch.tensor(task,dtype=int)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        coeffs_split = coeffs.split(self.n.tolist(),-1)
        coeffs_split_scaled = [(self.scale*coeffs_split[l])[...,None,:]*kmat_tasks[...,task,l,None] for l in range(self.num_tasks)]
        pcmean = torch.cat(coeffs_split_scaled,-1).sum(-1)
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pcmean[...,0] if inttask else pcmean
    def post_cubature_var(self, task:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True):
        """
        Posterior cubature variance. 

        Args:
            task (Union[int,torch.Tensor[T]]): task indices
            n (Union[int,torch.Tensor[num_tasks]]): number of points at which to evaluate the posterior cubature variance.
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`

        Returns:
            pcvar (torch.Tensor[T]): posterior cubature variance
        """
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and (n&(n-1)==0).all() and (n>=self.n).all(), "require n are all power of two greater than or equal to self.n"
        kmat_tasks = self.gram_matrix_tasks
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        inv = inv_log_det_cache()[0]
        to = inv_log_det_cache.task_order
        nord = n[to]
        mvec = torch.hstack([torch.zeros(1,device=self.device),(nord/nord[-1]).cumsum(0)]).to(int)[:-1]
        nsqrts = torch.sqrt(nord[:,None]*nord[None,:])
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int)
        if isinstance(task,list): task = torch.tensor(task,dtype=int)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        inv_cut = inv[...,mvec,:,:][...,:,mvec,:][...,0]
        kmat_tasks_left = kmat_tasks[...,task,:][...,:,to].to(self._FTOUTDTYPE)
        kmat_tasks_right = kmat_tasks[...,to,:][...,:,task].to(self._FTOUTDTYPE)
        term = torch.einsum("...ij,...jk,...ki->...i",kmat_tasks_left,nsqrts*inv_cut,kmat_tasks_right).real
        pcvar = self.scale*kmat_tasks[...,task,task]-self.scale**2*term
        pcvar[pcvar<0] = 0.
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pcvar[...,0] if inttask else pcvar
    def post_cubature_cov(self, task0:Union[int,torch.Tensor]=None, task1:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True):
        """
        Posterior cubature covariance. 

        Args:
            task0 (Union[int,torch.Tensor[T1]]): task indices
            task1 (Union[int,torch.Tensor[T2]]): task indices
            n (Union[int,torch.Tensor[num_tasks]]): number of points at which to evaluate the posterior cubature covariance.
            eval (bool): if `True`, disable gradients, otherwise use `torch.is_grad_enabled()`

        Returns:
            pcvar (torch.Tensor[T1,T2]): posterior cubature covariance
        """
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor) and (n&(n-1)==0).all() and (n>=self.n).all(), "require n are all power of two greater than or equal to self.n"
        kmat_tasks = self.gram_matrix_tasks
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        inv = inv_log_det_cache()[0]
        to = inv_log_det_cache.task_order
        nord = n[to]
        mvec = torch.hstack([torch.zeros(1,device=self.device),(nord/nord[-1]).cumsum(0)]).to(int)[:-1]
        nsqrts = torch.sqrt(nord[:,None]*nord[None,:])
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task0 is None: task0 = self.default_task
        inttask0 = isinstance(task0,int)
        if inttask0: task0 = torch.tensor([task0],dtype=int)
        if isinstance(task0,list): task0 = torch.tensor(task0,dtype=int)
        assert task0.ndim==1 and (task0>=0).all() and (task0<self.num_tasks).all()
        if task1 is None: task1 = self.default_task
        inttask1 = isinstance(task1,int)
        if inttask1: task1 = torch.tensor([task1],dtype=int)
        if isinstance(task1,list): task1 = torch.tensor(task1,dtype=int)
        assert task1.ndim==1 and (task1>=0).all() and (task1<self.num_tasks).all()
        equal = torch.equal(task0,task1)
        inv_cut = inv[...,mvec,:,:][...,:,mvec,:][...,0]
        kmat_tasks_left = kmat_tasks[...,task0,:][...,:,to].to(self._FTOUTDTYPE)
        kmat_tasks_right = kmat_tasks[...,to,:][...,:,task1].to(self._FTOUTDTYPE)
        term = torch.einsum("...ij,...jk,...kl->...il",kmat_tasks_left,nsqrts*inv_cut,kmat_tasks_right).real
        pccov = self.scale[...,None]*kmat_tasks[...,task0,:][...,:,task1]-self.scale[...,None]**2*term
        if equal:
            tvec = torch.arange(pccov.size(-1))
            diag = pccov[...,tvec,tvec]
            diag[diag<0] = 0. 
            pccov[...,tvec,tvec] = diag
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        if inttask0 and inttask1:
            return pccov[...,0,0]
        elif inttask0 and not inttask1:
            return pccov[...,0,:]
        elif not inttask0 and inttask1:
            return pccov[...,:,0]
        else: #not inttask0 and not inttask1
            return pccov
    def get_inv_log_det_cache(self, n=None):
        if n is None: n = self.n
        assert isinstance(n,torch.Tensor) and n.shape==(self.num_tasks,) and (n>=self.n).all()
        ntup = tuple(n.tolist())
        if ntup not in self.inv_log_det_cache_dict.keys():
            self.inv_log_det_cache_dict[ntup] = _InverseLogDetCache(self,n)
        return self.inv_log_det_cache_dict[ntup]
    def _kernel(self, x:torch.Tensor, z:torch.Tensor, beta0:torch.Tensor, beta1: torch.Tensor, c0:torch.Tensor, c1:torch.Tensor):
        return self._kernel_from_parts(self._kernel_parts(x,z,beta0,beta1),beta0,beta1,c0,c1)
