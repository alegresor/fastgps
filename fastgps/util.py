import torch 
import os 
import numpy as np 
import qmcpy as qp

class DummyDiscreteDistrib(qp.discrete_distribution.abstract_discrete_distribution.AbstractDiscreteDistribution):
    def __init__(self, x):
        assert isinstance(x,np.ndarray)
        self.x = x
        assert self.x.ndim==2 
        self.n,self.d = x.shape
        super(DummyDiscreteDistrib,self).__init__(dimension=x.shape[1],replications=None,seed=None,d_limit=np.inf,n_limit=np.inf)
    def _gen_samples(self, n_min, n_max, return_binary, warn):
        assert return_binary is False
        assert n_min==0 and n_max==self.n, "trying to generate samples other than the one provided is invalid"
        return self.x[None]

def _freeze(fgp):
    return {pname:pval.data.detach().clone() for pname,pval in fgp.state_dict().items()}

def _frozen_equal(fgp, state_dict):
    return not any((state_dict[pname]!=pval).any() for pname,pval in fgp.named_parameters())

def _force_recompile(fgp):
    return os.environ.get("FASTGP_FORCE_RECOMPILE")=="True" and any(pval.requires_grad for pname,pval in fgp.named_parameters())



