import sys
import os
import platform

system = platform.system()
paths = []
if system == "Linux":
    paths.append(
        "/lustre/fswork/projects/rech/tln/urh44lu/pinocchio-minimal-main/build/python"
    )
elif system == "Darwin":  # macOS
    paths.append("/Users/mscheffl/Desktop/pinocchio-minimal-main/build/python")
else:
    raise RuntimeError(f"Système non supporté : {system}")
for p in paths:
    if os.path.exists(p):
        if p not in sys.path:
            sys.path.insert(0, p)
import tartempion  # type: ignore
import numpy as np
import torch
from torch.autograd import Function
import pinocchio as pin
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from colorama import Fore, Style, init

init(autoreset=True)

Joint_ID = 15
tartempion.check()
print("querie1")


class QPkkt(Function):
    @staticmethod
    def forward(
        ctx,
        states_init: np.ndarray,
        p: torch.Tensor,
        A: torch.Tensor,
        b: torch.Tensor,
        rmodel: pin.Model,
        workspace: tartempion.QPworkspace,
        batch_size: int,
        seq_len: int,
        eq_dim: int,
        targets,
        dt,
        n_thread=1,
    ):
        ctx.workspace = workspace
        ctx.rmodel = rmodel
        ctx.device = p.device
        ctx.batch_size = batch_size
        ctx.eq_dim = eq_dim
        ctx.seq_len = seq_len
        ctx.dt = dt
        ctx.dof = rmodel.nq
        ctx.p = p
        ctx.q = states_init
        ctx.target = targets
        torch.save(ctx.p, "p.pth")
        torch.save(ctx.q, "q.pth")
        torch.save(ctx.target, "target.pth")
        ctx.n_thread = n_thread
        states_init = states_init.astype(np.float64)
        p_np: np.ndarray = p.detach().cpu().numpy()
        p_np = p_np.reshape((batch_size, seq_len, 6))
        p_np = p_np.astype(np.float64)
        if A is not None:
            A_np: np.ndarray = A.detach().cpu().numpy()
            A_np = A_np.reshape((-1, eq_dim, 6))
            b_np: np.ndarray = b.detach().cpu().numpy()
            b_np = b_np.reshape((-1, seq_len, eq_dim))
            A_np = A_np.astype(np.float64)
            b_np = b_np.astype(np.float64)
            print("doing forward")
            articular_speed: np.ndarray = tartempion.forward_pass(
                ctx.workspace,
                p_np,
                A_np,
                b_np,
                states_init,
                rmodel,
                n_thread,
                targets,
                dt,
            )
        else:
            A_np = None
            b_np = None
            articular_speed: np.ndarray = tartempion.forward_pass(
                ctx.workspace, p_np, states_init, rmodel, ctx.n_thread, targets, dt
            )
        ctx.articular_speed = p_np
        # if p.dtype != torch.float64:
        #     raise
        return torch.from_numpy(articular_speed).to(torch.float64).to(ctx.device)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):  # type: ignore
        grad_output = torch.zeros(ctx.batch_size, ctx.seq_len, 2 * ctx.dof + ctx.eq_dim)
        print("doing backward")
        torch.save(ctx.p, "p.pth")
        torch.save(ctx.q, "q.pth")
        torch.save(ctx.target, "target.pth")
        tartempion.backward_pass(
            ctx.workspace,
            ctx.rmodel,
            grad_output.cpu().numpy(),
            ctx.n_thread,
            grad_output.size(0),
        )
        print("bacward done")
        p_ = np.array(ctx.workspace.grad_p())
        p_ = np.reshape(p_, (ctx.batch_size, ctx.seq_len, 6))
        a_ = np.array(ctx.workspace.grad_A())
        a_ = np.reshape(a_, (-1, ctx.eq_dim, 6))
        b_ = np.array(ctx.workspace.grad_b())
        b_ = np.reshape(b_, (-1, ctx.eq_dim))
        p_tensor = torch.from_numpy(p_).to(ctx.device).to(torch.float64)
        a_tensor = torch.from_numpy(a_).to(ctx.device).to(torch.float64)
        b_tensor = torch.from_numpy(b_).to(ctx.device).to(torch.float64)
        # threshold = 1e3
        # grad_flat = p_tensor.view(-1)
        # top20_vals, top20_idx = torch.topk(grad_flat.abs(), k=20)
        # norms = p_tensor.norm(dim=(1, 2))
        # top_vals, top_idx = torch.topk(norms, k=5)
        # print("Top 5 blocs par norme sur dim (1,2):")
        # for idx in top_idx:
        #     i = idx.item()
        #     print(
        #         f"Index {i} → norme = {norms[i].item()} → ctx.articular_speed = {ctx.articular_speed[i, 0]}"
        #     )
        #     print(p_tensor[i].sum(0))

        # mask = (p_tensor.abs() > threshold).any(dim=(1, 2))
        # p_tensor[mask] = 0.0

        # print("grad", p_tensor)
        # print("norm", p_tensor.norm())
        # print("p_grad", p_tensor.sum(1))
        nan_mask = torch.isnan(p_tensor).any(dim=tuple(range(1, p_tensor.dim())))
        nan_indices = nan_mask.nonzero(as_tuple=True)[0]

        if len(nan_indices) > 0:
            for idx in nan_indices:
                print(f"NaN détecté dans le batch {idx.item()}:")
                print(ctx.p[idx, 0])
                print(ctx.q[idx])
                print(ctx.target[idx])
            print(ctx.p[0, 0])
            exit(1)
        mask = p_tensor.abs() > 1e5
        if mask.any():
            bad_indices = torch.where(mask.any(dim=1))[0]

            print(
                Fore.YELLOW
                + f"Indices problématiques sur l'axe 0 : {bad_indices.tolist()}"
            )
        if (p_tensor > 1e5).any():

            print(
                Fore.YELLOW + "Une valeur du tenseur dépasse 1e5. Arrêt du programme ?"
            )
            exit(1)

        if (p_tensor.abs() > 1e10).any():
            print(p_tensor)
            print(Fore.RED + "Une valeur du tenseur dépasse 1e10. Arrêt du programme ?")
            exit(1)

        # norms = p_tensor.norm(dim=(1, 2))
        # topk = 3
        # values, indices = torch.topk(norms, k=topk, largest=True)
        # p_tensor[indices, :, :] = 0
        return (
            None,
            p_tensor,
            a_tensor,
            b_tensor,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
