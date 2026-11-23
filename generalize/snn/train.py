"""Train one parametric SNN per family on n_train grids.  usage: python train.py family OUT.pt [steps] [M] [seed]"""
import sys
from time import perf_counter

import numpy as np
import torch

from cases_param import make, sample
from snn_defl import SNN, amg, ritz_loss, torch_sparse

N_TRAIN, N_INST, BATCH, LR = 127, 64, 4, 1e-3

if __name__ == "__main__":
    fam, out = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    M = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    F = int(sys.argv[6]) if len(sys.argv) > 6 else M          # F > M: "feat" variant (F features, M lowest Ritz)
    torch.manual_seed(seed); rng = np.random.default_rng(1000 + seed)
    t0 = perf_counter(); bank = []
    for p, chi in sample(rng, N_INST):
        A, _, X = make(fam, N_TRAIN, chi, p)
        P, _ = amg(A)
        bank.append((torch.from_numpy(X), torch.from_numpy(p)[None, :], torch_sparse(A), P))
    print(f"bank {N_INST} instances {perf_counter()-t0:.1f}s", flush=True)
    net = SNN(M=F); opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []; t0 = perf_counter()
    for it in range(steps):
        idx = rng.choice(N_INST, BATCH, replace=False)
        loss = sum(ritz_loss(net(bank[i][0], bank[i][1]), bank[i][2], bank[i][3], M) for i in idx) / BATCH
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(loss.item())
        if it % 100 == 0 or it == steps - 1:
            print(f"step {it:5d} loss {np.mean(hist[-100:]):.4f}  {perf_counter()-t0:.0f}s", flush=True)
    torch.save(dict(state=net.state_dict(), M=M, F=F, fam=fam, steps=steps, seed=seed, hist=hist,
                    train_s=perf_counter() - t0), out)
