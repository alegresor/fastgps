import fastgp
import torch

torch.set_default_dtype(torch.float64)

d = 3
num_tasks = 2
fs = [
    lambda x: torch.sin(x).sum(1),
    lambda x: torch.cos(x).sum(1)
]

fgps = [
    fastgp.FastMultiTaskGPLattice(seqs=d,num_tasks=num_tasks),
    fastgp.FastMultiTaskGPDigitalNetB2(seqs=d,num_tasks=num_tasks)
]

for fgp in fgps:
    print(fgp)
    x_next = fgp.get_x_next(n=[4,0])
    y_next = [fs[i](x_next[i]) for i in range(num_tasks)]
    fgp.add_y_next(y_next)

    x_next = fgp.get_x_next(n=[16,4])
    y_next = [fs[i](x_next[i]) for i in range(num_tasks)]
    fgp.add_y_next(y_next)

    print(fgp.get_lam(0,0).shape)
    print(fgp.get_lam(0,1).shape)
    print(fgp.get_lam(1,0).shape)
    print(fgp.get_lam(1,1).shape)
    print()

# fgp.fit(iterations=10)

# x_next = fgp.get_x_next(n=32)
# print(x_next.shape)
# y_next = f(x_next) 
# fgp.add_y_next(y_next)
# fgp.fit(iterations=10)
