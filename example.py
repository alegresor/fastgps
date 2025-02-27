import fastgp
import qmcpy as qp 
import numpy as np 
import os 
import pandas as pd

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"

os.environ["FASTGP_DEBUG"] = "True"

import torch
torch.set_default_dtype(torch.float64)
device = "cuda"

from matplotlib import pyplot
import tueplots.bundles
#pyplot.rcParams.update(tueplots.bundles.icml2024())
# pyplot.rcParams.update(tueplots.cycler.cycler(color=tueplots.constants.color.palettes.pn))
# colors = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
colors = ["xkcd:"+color[:-1] for color in pd.read_csv("./xkcd_colors.txt",comment="#").iloc[:,0].tolist()][::-1]

d = 1

def f_ackley(x, a=20, b=0.2, c=2*np.pi, scaling=32.768):
    # https://www.sfu.ca/~ssurjano/ackley.html
    assert x.ndim==2
    x = 2*scaling*x-scaling
    t1 = a*torch.exp(-b*torch.sqrt(torch.mean(x**2,1)))
    t2 = torch.exp(torch.mean(torch.cos(c*x),1))
    t3 = a+np.exp(1)
    y = -t1-t2+t3
    return y

xticks = torch.linspace(0,1,501,device=device)
yticks = f_ackley(xticks[:,None])

n = 2**2

print("lattice")
print("  n = %d"%n)
fgp_lattice = fastgp.FastGPLattice(
    f = f_ackley,
    n = n,
    seq = qp.Lattice(dimension=d,seed=7),
    device = device,
    )
fgp_lattice.fit()
pmean_lattice_n,pstd_lattice_n,q,ci_low_lattice_n,ci_high_lattice_n = fgp_lattice.post_ci(xticks[:,None])
x_lattice_n,y_lattice_n = fgp_lattice.x.clone(),fgp_lattice.y.clone()

print("  n = %d"%(2*n))
fgp_lattice.double_n()
fgp_lattice.fit()
pmean_lattice_2n,pstd_lattice_2n,q,ci_low_lattice_2n,ci_high_lattice_2n = fgp_lattice.post_ci(xticks[:,None])
x_lattice_2n,y_lattice_2n = fgp_lattice.x.clone(),fgp_lattice.y.clone()

print("  n = %d"%(4*n))
fgp_lattice.double_n()
fgp_lattice.fit()
pmean_lattice_4n,pstd_lattice_4n,q,ci_low_lattice_4n,ci_high_lattice_4n = fgp_lattice.post_ci(xticks[:,None])
x_lattice_4n,y_lattice_4n = fgp_lattice.x.clone(),fgp_lattice.y.clone()

print()

print("digital net")
print("  n = %d"%n)
fgp_dnb2 = fastgp.FastGPDigitalNetB2(
    f = f_ackley,
    n = n,
    seq = qp.DigitalNetB2(dimension=1,seed=7),
    device = device,
    )
fgp_dnb2.fit()
pmean_dnb2_n,pstd_dnb2_n,q,ci_low_dnb2_n,ci_high_dnb2_n = fgp_dnb2.post_ci(xticks[:,None])
x_dnb2_n,y_dnb2_n = fgp_dnb2.x.clone(),fgp_dnb2.y.clone()

print("  n = %d"%(2*n))
fgp_dnb2.double_n()
fgp_dnb2.fit()
pmean_dnb2_2n,pstd_dnb2_2n,q,ci_low_dnb2_2n,ci_high_dnb2_2n = fgp_dnb2.post_ci(xticks[:,None])
x_dnb2_2n,y_dnb2_2n = fgp_dnb2.x.clone(),fgp_dnb2.y.clone()

print("  n = %d"%(4*n))
fgp_dnb2.double_n()
fgp_dnb2.fit()
pmean_dnb2_4n,pstd_dnb2_4n,q,ci_low_dnb2_4n,ci_high_dnb2_4n = fgp_dnb2.post_ci(xticks[:,None])
x_dnb2_4n,y_dnb2_4n = fgp_dnb2.x.clone(),fgp_dnb2.y.clone()

print("collecting data and plotting")
data = [
    [   (x_lattice_n,y_lattice_n,pmean_lattice_n,ci_low_lattice_n,ci_high_lattice_n),
        (x_dnb2_n,y_dnb2_n,pmean_dnb2_n,ci_low_dnb2_n,ci_high_dnb2_n)],
    [   (x_lattice_2n,y_lattice_2n,pmean_lattice_2n,ci_low_lattice_2n,ci_high_lattice_2n),
        (x_dnb2_2n,y_dnb2_2n,pmean_dnb2_2n,ci_low_dnb2_2n,ci_high_dnb2_2n)],
    [   (x_lattice_4n,y_lattice_4n,pmean_lattice_4n,ci_low_lattice_4n,ci_high_lattice_4n),
        (x_dnb2_4n,y_dnb2_4n,pmean_dnb2_4n,ci_low_dnb2_4n,ci_high_dnb2_4n)],
]
nrows = 3
ncols = 2
_alpha = 0.25
pyplot.rcParams.update(tueplots.figsizes.icml2024_full(nrows=nrows,ncols=ncols))
fig,ax = pyplot.subplots(nrows=nrows,ncols=ncols)
for i in range(3):
    for j in range(2):
        x,y,pmean,ci_low,ci_high = data[i][j]
        ax[i,j].plot(xticks.cpu(),yticks.cpu(),color="k")
        ax[i,j].scatter(x[:,0].cpu(),y.cpu(),color="k")
        ax[i,j].plot(xticks.cpu(),pmean.cpu(),color=colors[j])
        ax[i,j].fill_between(xticks.cpu(),ci_low.cpu(),ci_high.cpu(),color=colors[j],alpha=_alpha)
fig.savefig("example.pdf")