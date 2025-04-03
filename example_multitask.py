import fastgp
import torch
from matplotlib import pyplot
import os
import pandas as pd 
import numpy as np 
import qmcpy as qp 

torch.set_default_dtype(torch.float64)
#os.environ["FASTGP_DEBUG"] = "True"
torch.autograd.set_detect_anomaly(True)

colors = ["xkcd:"+color[:-1] for color in pd.read_csv("./xkcd_colors.txt",comment="#").iloc[:,0].tolist()][::-1]
_alpha = 0.25
WIDTH = 2*(500/72)
LINEWIDTH = 3
MARKERSIZE = 100

device = "cpu"

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
fs = [lambda x: f_ackley(x,c=0), lambda x: f_ackley(x)]
num_tasks = len(fs)
n = torch.tensor([2**4,2**3])

FGPClass = fastgp.FastGPLattice
seqs = [qp.Lattice(1,seed=7),qp.Lattice(1,seed=11)]
# FGPClass = fastgp.FastGPDigitalNetB2
# seqs = [qp.DigitalNetB2(1,seed=7,randomize="DS"),qp.DigitalNetB2(1,seed=11,randomize="DS")]

xticks = torch.linspace(0,1,101,device=device)[1:-1,None]
yticks = torch.vstack([fs[i](xticks) for i in range(num_tasks)])

pmeans = torch.empty((num_tasks,len(xticks)))
ci_lows = torch.empty((num_tasks,len(xticks)))
ci_highs = torch.empty((num_tasks,len(xticks)))
fgp_indep = [FGPClass(seqs=seqs[l],device=device) for l in range(num_tasks)]
for l in range(num_tasks):
    x_next = fgp_indep[l].get_x_next(n=n[l].item())
    y_next = torch.vstack([fs[i](x_next) for i in range(num_tasks)])
    fgp_indep[l].add_y_next(y_next[l])
    fgp_indep[l].fit()
    pmeans[l],_,_,ci_lows[l],ci_highs[l] = fgp_indep[l].post_ci(xticks)

fgp_multitask = FGPClass(seqs=seqs,num_tasks=num_tasks,device=device)
x_next = fgp_multitask.get_x_next(n=n)
y_next = [fs[i](x_next[i]) for i in range(num_tasks)]
fgp_multitask.add_y_next(y_next)
fgp_multitask.fit()
pmean_mt,_,_,ci_low_mt,ci_high_mt = fgp_multitask.post_ci(xticks)

fig,ax = pyplot.subplots(nrows=1,ncols=num_tasks,figsize=(WIDTH,WIDTH/3),sharex=True,sharey=True)
for l in range(num_tasks):
    ax[l].plot(xticks[:,0].cpu(),yticks[l].cpu(),color="k",linewidth=LINEWIDTH)
    ax[l].scatter(fgp_indep[l].x[:,0].cpu(),fgp_indep[l].y.cpu(),color="k",s=MARKERSIZE)
pltmin = min([ci_lows[l].min() for l in range(num_tasks)]+[ci_low_mt.min()])
pltmax = max([ci_highs[l].max() for l in range(num_tasks)]+[ci_high_mt.max()])
ax[0].set_ylim([pltmin-0*(pltmax-pltmin),pltmax+0*(pltmax-pltmin)])
fig.savefig("example_multitask_0.pdf",bbox_inches="tight")
for l in range(num_tasks):
    ax[l].plot(xticks[:,0].cpu(),pmeans[l].cpu(),color=colors[0],linewidth=LINEWIDTH)
    ax[l].fill_between(xticks[:,0].cpu(),ci_lows[l].cpu(),ci_highs[l].cpu(),color=colors[0],alpha=_alpha,label="independent GPs")
ax[0].legend(frameon=False,loc="upper left",ncols=2,fontsize="xx-large") 
fig.savefig("example_multitask_1.pdf",bbox_inches="tight")
for l in range(num_tasks):
    ax[l].plot(xticks[:,0].cpu(),pmean_mt[l].cpu(),color=colors[1],linewidth=LINEWIDTH)
    ax[l].fill_between(xticks[:,0].cpu(),ci_low_mt[l].cpu(),ci_high_mt[l].cpu(),color=colors[1],alpha=_alpha,label="MIGP")
ax[0].legend(frameon=False,loc="upper left",ncols=2,fontsize="xx-large") 
fig.savefig("example_multitask_2.pdf",bbox_inches="tight")
