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

seqs_lattice = [qp.Lattice(1,seed=7),qp.Lattice(1,seed=2)]
seqs_dnb2s = [qp.DigitalNetB2(1,seed=7,randomize="DS"),qp.DigitalNetB2(1,seed=2,randomize="DS")]

FGPClasses = [fastgp.FastGPLattice,fastgp.FastGPDigitalNetB2]

xticks = torch.linspace(0,1,1001,device=device)[1:-1,None]
yticks = torch.vstack([fs[i](xticks) for i in range(num_tasks)])


pmeans = [torch.empty((num_tasks,len(xticks))) for i in range(2)]
ci_lows = [torch.empty((num_tasks,len(xticks))) for i in range(2)]
ci_highs = [torch.empty((num_tasks,len(xticks))) for i in range(2)]
fgp_indep = [[FGPClass(seqs=seqs[l],device=device) for l in range(num_tasks)] for FGPClass,seqs in zip(FGPClasses,[seqs_lattice,seqs_dnb2s])]
for i in range(2):
    for l in range(num_tasks):
        x_next = fgp_indep[i][l].get_x_next(n=n[l].item())
        y_next = torch.vstack([fs[i](x_next) for i in range(num_tasks)])
        fgp_indep[i][l].add_y_next(y_next[l])
        fgp_indep[i][l].fit()
        pmeans[i][l],_,_,ci_lows[i][l],ci_highs[i][l] = fgp_indep[i][l].post_ci(xticks)

fgp_multitask = [FGPClass(seqs=seqs,num_tasks=num_tasks,device=device) for FGPClass,seqs in zip(FGPClasses,[seqs_lattice,seqs_dnb2s])]
pmean_mt = [None for i in range(2)]
ci_low_mt = [None for i in range(2)]
ci_high_mt = [None for i in range(2)]
for i in range(2):
    x_next = fgp_multitask[i].get_x_next(n=n)
    y_next = [fs[i](x_next[i]) for i in range(num_tasks)]
    fgp_multitask[i].add_y_next(y_next)
    fgp_multitask[i].fit()
    pmean_mt[i],_,_,ci_low_mt[i],ci_high_mt[i] = fgp_multitask[i].post_ci(xticks)

fig,ax = pyplot.subplots(nrows=2,ncols=num_tasks,figsize=(WIDTH,WIDTH/num_tasks*2/1.5),sharex=True,sharey="row")
ax = ax.reshape((2,num_tasks))
for i in range(2):
    for l in range(num_tasks):
        ax[i,l].plot(xticks[:,0].cpu(),yticks[l].cpu(),color="k",linewidth=LINEWIDTH)
        ax[i,l].scatter(fgp_indep[i][l].x[:,0].cpu(),fgp_indep[i][l].y.cpu(),color="k",s=MARKERSIZE)
    pltmin = min([ci_lows[i][l].min() for l in range(num_tasks)]+[ci_low_mt[i].min()])
    pltmax = max([ci_highs[i][l].max() for l in range(num_tasks)]+[ci_high_mt[i].max()])
    ax[i,0].set_ylim([pltmin-.05*(pltmax-pltmin),pltmax+.05*(pltmax-pltmin)])
ax[0,0].set_ylabel("Lattice + SI Kernel",fontsize="xx-large")
ax[1,0].set_ylabel("Digital Net + DSI Kernel",fontsize="xx-large")
fig.savefig("example_multitask_0.pdf",bbox_inches="tight")
for i in range(2):
    for l in range(num_tasks):
        ax[i,l].plot(xticks[:,0].cpu(),pmeans[i][l].cpu(),color=colors[0],linewidth=LINEWIDTH)
        ax[i,l].fill_between(xticks[:,0].cpu(),ci_lows[i][l].cpu(),ci_highs[i][l].cpu(),color=colors[0],alpha=_alpha,label="independent GPs")
    ax[i,0].legend(frameon=False,loc="upper left",ncols=2,fontsize="xx-large") 
fig.savefig("example_multitask_1.pdf",bbox_inches="tight")
for i in range(2):
    for l in range(num_tasks):
        ax[i,l].plot(xticks[:,0].cpu(),pmean_mt[i][l].cpu(),color=colors[1],linewidth=LINEWIDTH)
        ax[i,l].fill_between(xticks[:,0].cpu(),ci_low_mt[i][l].cpu(),ci_high_mt[i][l].cpu(),color=colors[1],alpha=_alpha,label="MTGP")
    ax[i,0].legend(frameon=False,loc="upper left",ncols=2,fontsize="xx-large") 
fig.savefig("example_multitask_2.pdf",bbox_inches="tight")
