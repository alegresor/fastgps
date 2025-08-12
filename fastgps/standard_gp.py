from .abstract_gp import AbstractGP
from .util import (
    DummyDiscreteDistrib,
    _StandardInverseLogDetCache,
)
import torch
import numpy as np
import qmcpy
from typing import Tuple,Union

class StandardGP(AbstractGP):
    """
    Standard Gaussian process regression
    
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

        >>> n = 2**6
        >>> d = 2
        >>> sgp = StandardGP(
        ...     qmcpy.KernelSquaredExponential(d,torchify=True,device=device),
        ...     qmcpy.DigitalNetB2(dimension=d,seed=7))
        >>> x_next = sgp.get_x_next(n)
        >>> y_next = f_ackley(x_next)
        >>> sgp.add_y_next(y_next)

        >>> rng = torch.Generator().manual_seed(17)
        >>> x = torch.rand((2**7,d),generator=rng).to(device)
        >>> y = f_ackley(x)
        
        >>> pmean = sgp.post_mean(x)

        >>> pmean.shape
        torch.Size([128])
        >>> torch.linalg.norm(y-pmean)/torch.linalg.norm(y)
        tensor(0.0817)
        >>> torch.linalg.norm(sgp.post_mean(sgp.x)-sgp.y)/torch.linalg.norm(y)
        tensor(0.0402)

        >>> data = sgp.fit(verbose=0)
        >>> list(data.keys())
        []

        >>> torch.linalg.norm(y-sgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0577)
        >>> z = torch.rand((2**8,d),generator=rng).to(device)
        >>> pcov = sgp.post_cov(x,z)
        >>> pcov.shape
        torch.Size([128, 256])

        >>> pcov = sgp.post_cov(x,x)
        >>> pcov.shape
        torch.Size([128, 128])
        >>> (pcov.diagonal()>=0).all()
        tensor(True)

        >>> pvar = sgp.post_var(x)
        >>> pvar.shape
        torch.Size([128])
        >>> torch.allclose(pcov.diagonal(),pvar)
        True


        >>> pmean,pstd,q,ci_low,ci_high = sgp.post_ci(x,confidence=0.99)
        >>> ci_low.shape
        torch.Size([128])
        >>> ci_high.shape
        torch.Size([128])

        >>> sgp.post_cubature_mean()
        tensor(20.3584)
        >>> sgp.post_cubature_var()
        tensor(0.0035)

        >>> pcmean,pcvar,q,pcci_low,pcci_high = sgp.post_cubature_ci(confidence=0.99)
        >>> pcci_low
        tensor(20.2069)
        >>> pcci_high
        tensor(20.5098)
        
        >>> pcov_future = sgp.post_cov(x,z,n=2*n)
        >>> pvar_future = sgp.post_var(x,n=2*n)
        >>> pcvar_future = sgp.post_cubature_var(n=2*n)
        
        >>> x_next = sgp.get_x_next(2*n)
        >>> y_next = f_ackley(x_next)
        >>> sgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-sgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0848)

        >>> torch.allclose(sgp.post_cov(x,z),pcov_future)
        True
        >>> torch.allclose(sgp.post_var(x),pvar_future)
        True
        >>> torch.allclose(sgp.post_cubature_var(),pcvar_future)
        True

        >>> data = sgp.fit(verbose=False)
        >>> torch.linalg.norm(y-sgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0565)

        >>> x_next = sgp.get_x_next(4*n)
        >>> y_next = f_ackley(x_next)
        >>> sgp.add_y_next(y_next)
        >>> torch.linalg.norm(y-sgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.1515)

        >>> data = sgp.fit(verbose=False)
        >>> torch.linalg.norm(y-sgp.post_mean(x))/torch.linalg.norm(y)
        tensor(0.0555)

        >>> pcov_16n = sgp.post_cov(x,z,n=16*n)
        >>> pvar_16n = sgp.post_var(x,n=16*n)
        >>> pcvar_16n = sgp.post_cubature_var(n=16*n)
        >>> x_next = sgp.get_x_next(16*n)
        >>> y_next = f_ackley(x_next)
        >>> sgp.add_y_next(y_next)
        >>> torch.allclose(sgp.post_cov(x,z),pcov_16n)
        True
        >>> torch.allclose(sgp.post_var(x),pvar_16n)
        True
        >>> torch.allclose(sgp.post_cubature_var(),pcvar_16n)
        True

        Different loss metrics for fitting 
        
        >>> n = 2**6
        >>> d = 3
        >>> sgp = StandardGP(
        ...     qmcpy.KernelSquaredExponential(d,torchify=True,device=device),
        ...     qmcpy.DigitalNetB2(dimension=d,seed=7))
        >>> x_next = sgp.get_x_next(n)
        >>> y_next = torch.stack([torch.sin(x_next).sum(-1),torch.cos(x_next).sum(-1)],axis=0)
        >>> sgp.add_y_next(y_next)
        >>> data = sgp.fit(loss_metric="MLL",iterations=5,verbose=0)
        >>> data = sgp.fit(loss_metric="CV",iterations=5,verbose=0,cv_weights=1/torch.arange(1,2*n+1,device=device).reshape((2,n)))
        >>> data = sgp.fit(loss_metric="CV",iterations=5,verbose=0,cv_weights="L2R")
        >>> data = sgp.fit(loss_metric="GCV",iterations=5,verbose=0)

        Data Driven

        >>> x = [
        ...     torch.rand((3,1),generator=rng).to(device),
        ...     torch.rand((12,1),generator=rng).to(device),
        ... ]
        >>> y = [
        ...     torch.stack([torch.sin(2*np.pi*x[0][:,0]),torch.cos(2*np.pi*x[0][:,0]),torch.acos(x[0][:,0])],dim=0).to(device),
        ...     torch.stack([4*torch.sin(2*np.pi*x[1][:,0]),4*torch.cos(2*np.pi*x[1][:,0]),4*torch.acos(x[1][:,0])],dim=0).to(device),
        ... ]
        >>> sgp = StandardGP(
        ...     qmcpy.KernelMultiTask(
        ...         qmcpy.KernelGaussian(d=1,torchify=True,device=device),
        ...         num_tasks = 2,
        ...     ),
        ...     seqs={"x":x,"y":y},
        ...     noise = 1e-3,
        ... )
        >>> data = sgp.fit(verbose=0,iterations=10)
        >>> xticks = torch.linspace(0,1,101,device=device) 
        >>> pmean,pvar,q,pci_low,pci_high = sgp.post_ci(xticks[:,None])
        >>> pcmean,pcvar,q,pcci_low,pcci_high =  sgp.post_cubature_ci()
    """
    def __init__(self,
            kernel:qmcpy.kernel.abstract_kernel.AbstractKernel,
            seqs:Union[qmcpy.IIDStdUniform,int],
            noise:float = 1e-4,
            tfs_noise:Tuple[callable,callable] = (qmcpy.util.transforms.tf_exp_eps_inv,qmcpy.util.transforms.tf_exp_eps),
            requires_grad_noise:bool = False, 
            shape_noise:torch.Size = torch.Size([1]),
            derivatives:list = None,
            derivatives_coeffs:list = None,
            adaptive_nugget:bool = True,
            data:dict = None,
            ):
        """
        Args:
            kernel (qmcpy.AbstractKernel): Kernel object. Set to `qmcpy.KernelMultiTask` for a multi-task GP.
            seqs (Union[int,qmcpy.DiscreteDistribution,List]]): list of sequence generators. If an int `seed` is passed in we use 
                ```python
                [qmcpy.DigitalNetB2(d,seed=seed_i) for seed_i in np.random.SeedSequence(seed).spawn(num_tasks)]
                ```
                See the <a href="https://qmcpy.readthedocs.io/en/latest/algorithms.html#discrete-distribution-class" target="_blank">`qmcpy.DiscreteDistribution` docs</a> for more info. 
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
            data (dict): dictory of data with keys 'x' and 'y' where data['x'] and data['y'] are both `torch.Tensor`s or list of `torch.Tensor`s with lengths equal to the number of tasks
        """
        self._XBDTYPE = torch.get_default_dtype()
        self._FTOUTDTYPE = torch.get_default_dtype()
        if isinstance(kernel,qmcpy.KernelMultiTask):
            solo_task = False
            num_tasks = kernel.num_tasks
            default_task = torch.arange(num_tasks)
        else:
            solo_task = True
            default_task = 0 
            num_tasks = 1
        if isinstance(seqs,dict):
            data = seqs
            assert "x" in data and "y" in data, "dict seqs must have keys 'x' and 'y'"
            if isinstance(data["x"],torch.Tensor): data["x"] = [data["x"]]
            if isinstance(data["y"],torch.Tensor): data["y"] = [data["y"]]
            assert isinstance(data["x"],list) and len(data["x"])==num_tasks and all(isinstance(x_l,torch.Tensor) and x_l.ndim==2 and x_l.size(1)==kernel.d for x_l in data["x"]), "data['x'] should be a list of 2d tensors of length num_tasks with each number of columns equal to the dimension"
            assert isinstance(data["y"],list) and len(data["y"])==num_tasks and all(isinstance(y_l,torch.Tensor) and y_l.ndim>=1 for y_l in data["y"]), "data['y'] should be a list of tensors of length num_tasks"
            seqs = np.array([DummyDiscreteDistrib(data["x"][l].cpu().detach().numpy()) for l in range(num_tasks)],dtype=object)
        else:
            data = None
            if isinstance(seqs,int):
                global_seed = seqs
                seqs = np.array([qmcpy.DigitalNetB2(kernel.d,seed=seed,order="GRAY") for seed in np.random.SeedSequence(global_seed).spawn(num_tasks)],dtype=object)
            if isinstance(seqs,qmcpy.DiscreteDistribution):
                seqs = np.array([seqs],dtype=object)
            if isinstance(seqs,list):
                seqs = np.array(seqs,dtype=object)
        assert seqs.shape==(num_tasks,), "seqs should be a length num_tasks=%d list"%num_tasks
        assert all(seqs[i].replications==1 for i in range(num_tasks)) and "each seq should have only 1 replication"
        super().__init__(
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
        if data is not None:
            self.add_y_next(data["y"],task=torch.arange(self.num_tasks))
    def get_inv_log_det_cache(self, n=None):
        if n is None: n = self.n
        assert isinstance(n,torch.Tensor) and n.shape==(self.num_tasks,) and (n>=self.n).all()
        ntup = tuple(n.tolist())
        if ntup not in self.inv_log_det_cache_dict.keys():
            self.inv_log_det_cache_dict[ntup] = _StandardInverseLogDetCache(self,n)
        return self.inv_log_det_cache_dict[ntup]
    def post_cubature_mean(self, task:Union[int,torch.Tensor]=None, eval:bool=True, integrate_unit_cube:bool=True):
        coeffs = self.coeffs
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int,device=self.device)
        if isinstance(task,list): task = torch.tensor(task,dtype=int,device=self.device)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        kints = torch.cat([self.kernel.single_integral_01d(task[:,None],l,self.get_x(l)) for l in range(self.num_tasks)],dim=-1)
        pcmean = (kints*coeffs[...,None,:]).sum(-1)
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pcmean[...,0] if inttask else pcmean
    def post_cubature_var(self, task:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True, integrate_unit_cube:bool=True):
        assert integrate_unit_cube, "undefinted posterior variance when integrating first term over all reals"
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor)
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task is None: task = self.default_task
        inttask = isinstance(task,int)
        if inttask: task = torch.tensor([task],dtype=int,device=self.device)
        if isinstance(task,list): task = torch.tensor(task,dtype=int,device=self.device)
        assert task.ndim==1 and (task>=0).all() and (task<self.num_tasks).all()
        kints = torch.cat([self.kernel.single_integral_01d(task[:,None],l,self.get_x(l,n=n[l])) for l in range(self.num_tasks)],dim=-1)
        v = inv_log_det_cache.gram_matrix_solve(kints.movedim(-2,0)).movedim(0,-2)
        tval = self.kernel.double_integral_01d(task,task)
        pcvar = tval-(kints*v).sum(-1)
        pcvar[pcvar<0] = 0.
        if eval:
            torch.set_grad_enabled(incoming_grad_enabled)
        return pcvar[...,0] if inttask else pcvar
    def post_cubature_cov(self, task0:Union[int,torch.Tensor]=None, task1:Union[int,torch.Tensor]=None, n:Union[int,torch.Tensor]=None, eval:bool=True, integrate_unit_cube:bool=True):
        assert integrate_unit_cube, "undefinted posterior variance when integrating first term over all reals"
        if n is None: n = self.n
        if isinstance(n,int): n = torch.tensor([n],dtype=int,device=self.device)
        assert isinstance(n,torch.Tensor)
        inv_log_det_cache = self.get_inv_log_det_cache(n)
        if eval:
            incoming_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        if task0 is None: task0 = self.default_task
        inttask0 = isinstance(task0,int)
        if inttask0: task0 = torch.tensor([task0],dtype=int,device=self.device)
        if isinstance(task0,list): task0 = torch.tensor(task0,dtype=int,device=self.device)
        assert task0.ndim==1 and (task0>=0).all() and (task0<self.num_tasks).all()
        if task1 is None: task1 = self.default_task
        inttask1 = isinstance(task1,int)
        if inttask1: task1 = torch.tensor([task1],dtype=int,device=self.device)
        if isinstance(task1,list): task1 = torch.tensor(task1,dtype=int,device=self.device)
        assert task1.ndim==1 and (task1>=0).all() and (task1<self.num_tasks).all()
        equal = torch.equal(task0,task1)
        kints0 = torch.cat([self.kernel.single_integral_01d(task0[:,None],l,self.get_x(l,n=n[l])) for l in range(self.num_tasks)],dim=-1)
        kints1 = kints0 if equal else torch.cat([self.kernel.single_integral_01d(task1[:,None],l,self.get_x(l,n=n[l])) for l in range(self.num_tasks)],dim=-1)
        v = inv_log_det_cache.gram_matrix_solve(kints1.movedim(-2,0)).movedim(0,-2)
        tval = self.kernel.double_integral_01d(task0[:,None],task1[None,:])
        pccov = tval-(kints0[...,:,None,:]*v[...,None,:,:]).sum(-1)
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
    
