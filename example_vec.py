import fastgp
import torch
from matplotlib import pyplot
import os
import pandas as pd 
import numpy as np 

torch.set_default_dtype(torch.float64)
os.environ["FASTGP_DEBUG"] = "True"
torch.autograd.set_detect_anomaly(True)

colors = ["xkcd:"+color[:-1] for color in pd.read_csv("./xkcd_colors.txt",comment="#").iloc[:,0].tolist()][::-1]
_alpha = 0.25

def f_ackley(x, a=20, b=0.2, c=2*np.pi, scaling=32.768):
    # https://www.sfu.ca/~ssurjano/ackley.html
    assert x.ndim==2
    x = 2*scaling*x-scaling
    t1 = a*torch.exp(-b*torch.sqrt(torch.mean(x**2,1)))
    t2 = torch.exp(torch.mean(torch.cos(c*x),1))
    t3 = a+np.exp(1)
    y = -t1-t2+t3
    return y

d = 1
fs = [
    lambda x: f_ackley(x,c=0),
    #lambda x: torch.cos(2*np.pi*x).sum(1),
    lambda x: f_ackley(x),
]
n = torch.tensor([2**5,2**3])
n_new = n.clone(); n_new[0] = 8*n_new[0]
num_tasks = len(n)


fgp_indep = fastgp.FastMultiTaskGPDigitalNetB2(seqs=d,seed_for_seq=7,num_tasks=num_tasks,factor_task_kernel=0,requires_grad_factor_task_kernel=False,requires_grad_noise_task_kernel=False)
#fgp_indep = fastgp.FastMultiTaskGPLattice(seqs=d,seed_for_seq=7,num_tasks=num_tasks,factor_task_kernel=0,requires_grad_factor_task_kernel=False,requires_grad_noise_task_kernel=False)

fgp_multitask = fastgp.FastMultiTaskGPDigitalNetB2(seqs=d,seed_for_seq=7,num_tasks=num_tasks)
#fgp_multitask = fastgp.FastMultiTaskGPLattice(seqs=d,seed_for_seq=7,num_tasks=num_tasks,requires_grad_lengthscales=False)

xticks = torch.linspace(0,1,101)[1:-1,None]
yticks = torch.vstack([fs[i](xticks) for i in range(num_tasks)])
fig,ax = pyplot.subplots(nrows=3,ncols=num_tasks,figsize=(10,8),sharex=True,sharey="col")
ax = np.atleast_1d(ax).reshape((3,num_tasks))
for i,fgp in enumerate([fgp_indep,fgp_multitask]):
    x_next = fgp.get_x_next(n=n)
    y_next = [fs[i](x_next[i]) for i in range(num_tasks)]
    fgp.add_y_next(y_next)
    fgp.fit(
        #iterations=5,
        #lr=1e-3,
        #optimizer=torch.optim.Adam(fgp.parameters(),lr=1e-1,amsgrad=True),
    )
    fgp.post_cov(xticks,xticks[::2],n=None)
    fgp.post_cov(xticks,xticks,n=n_new)
    pmean,pvar,q,ci_low,ci_high = fgp.post_ci(xticks)
    pvar_new = fgp.post_var(xticks,n=n_new)
    for l in range(num_tasks):
        ax[i,l].plot(xticks[:,0],yticks[l],color="k")
        ax[i,l].plot(xticks[:,0],pmean[l],color=colors[i])
        ax[i,l].fill_between(xticks[:,0],ci_low[l],ci_high[l],color=colors[i],alpha=_alpha)
        ax[i,l].scatter(fgp.get_x(l)[:,0],fgp.y[l],color="k")
        ax[2,l].plot(xticks[:,0],pvar[l],color=colors[i])
        ax[2,l].plot(xticks[:,0],pvar_new[l],color=colors[i],linestyle='--')
ax[0,0].set_ylabel("FGP Independent")
ax[1,0].set_ylabel("FGP Multitask")
for l in range(num_tasks):
    ax[2,l].set_yscale("log",base=10)
    ax[0,l].set_title("Task %d"%l)
fig.tight_layout()
fig.savefig("example_vec.pdf")

# fgp.fit(iterations=10)

# x_next = fgp.get_x_next(n=32)
# print(x_next.shape)
# y_next = f(x_next) 
# fgp.add_y_next(y_next)
# fgp.fit(iterations=10)
