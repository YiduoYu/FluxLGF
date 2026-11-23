"""Supervised SNN (subspace least squares, as in Xu & Sheng's SNN where the span is fitted by least squares):
L = mean_i (1 - ||Q^T u_i||^2),  Q = qr(Phi(X; p)),  u_i = unit-norm slow modes of the training instance.
usage: python train_sup.py targets.npz OUT.pt [steps] [M] [seed]"""
import sys
from time import perf_counter

import numpy as np
import torch

from snn_defl import make_net

BATCH, LR, NPTS = 4, 1e-3, 8192

if __name__ == "__main__":
    data, out = np.load(sys.argv[1]), sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    M = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    arch = sys.argv[6] if len(sys.argv) > 6 else "mlp"
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X = torch.from_numpy(data["X"]); U = data["U"]; Pp = data["p"]; S = len(Pp)
    net = make_net(arch, M); opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist, t0 = [], perf_counter()
    for it in range(steps):
        loss = 0
        for i in rng.choice(S, BATCH, replace=False):
            j = torch.from_numpy(rng.choice(X.shape[0], NPTS, replace=False))       # Monte-Carlo L2 projection
            Q, _ = torch.linalg.qr(net(X[j], torch.from_numpy(Pp[i])[None, :]))
            Ui = torch.from_numpy(U[i][j.numpy()].astype(np.float64)); Ui = Ui / Ui.norm(dim=0)
            loss = loss + (1 - ((Q.T @ Ui) ** 2).sum(0)).mean() / BATCH
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); hist.append(loss.item())
        if it % 200 == 0 or it == steps - 1:
            print(f"step {it:5d} loss {np.mean(hist[-200:]):.4f} {perf_counter()-t0:.0f}s", flush=True)
    torch.save(dict(state=net.state_dict(), M=M, F=M, arch=arch, steps=steps, seed=seed, hist=hist, train_s=perf_counter() - t0,
                    targets=sys.argv[1]), out)
