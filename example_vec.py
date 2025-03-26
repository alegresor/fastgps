import fastgp
import torch

import os
#os.environ["FASTGP_DEBUG"] = "True"

#torch.autograd.set_detect_anomaly(True)

torch.set_default_dtype(torch.float64)

d = 1
num_tasks = 3
fs = [
    lambda x: torch.sin(x).sum(1),
    lambda x: torch.cos(x).sum(1),
    lambda x: torch.tan(x).sum(1),
    lambda x: torch.arctan(x).sum(1),
]

fgps = [
    fastgp.FastMultiTaskGPDigitalNetB2(
        seqs=d,
        seed_for_seq=7,
        num_tasks=num_tasks,
        requires_grad_factor_task_kernel=True,
        requires_grad_noise_task_kernel=True,
    ),
    fastgp.FastMultiTaskGPLattice(
        seqs=d,
        seed_for_seq=7,
        num_tasks=num_tasks,
        requires_grad_factor_task_kernel=True,
        requires_grad_noise_task_kernel=True,
    ),
]

for fgp in fgps:
    print(fgp)
    # x_next = fgp.get_x_next(n=[4,2,1])
    # y_next = [fs[i](x_next[i]) for i in range(num_tasks)]
    # fgp.add_y_next(y_next)

    x_next = fgp.get_x_next(n=[2**8,2**6,2**3])
    y_next = [fs[i](x_next[i]) for i in range(num_tasks)]
    fgp.add_y_next(y_next)
    
    #t = fgp.gram_matrix_solve(torch.rand(4,5,fgp.n.sum(),3))
    
    fgp.fit(
        #optimizer = torch.optim.Adam(fgp.parameters(),lr=1,amsgrad=True),
        verbose=10,
        #lr=1e-2
    )
# fgp.fit(iterations=10)

# x_next = fgp.get_x_next(n=32)
# print(x_next.shape)
# y_next = f(x_next) 
# fgp.add_y_next(y_next)
# fgp.fit(iterations=10)
