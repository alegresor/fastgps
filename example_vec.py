import fastgp
import torch

torch.set_default_dtype(torch.float64)

d = 1 
f1 = lambda x: torch.sin(x).sum(1)
f2 = lambda x: torch.cos(x).sum(1)
f = lambda x: torch.vstack([f1(x),f2(x)])

fgp = fastgp.FastGPLattice(d)

x_next = fgp.get_x_next(n=8)
y_next = f(x_next) 
fgp.add_y_next(y_next)
fgp.fit(iterations=10)

x_next = fgp.get_x_next(n=32)
print(x_next.shape)
y_next = f(x_next) 
fgp.add_y_next(y_next)
fgp.fit(iterations=10)
