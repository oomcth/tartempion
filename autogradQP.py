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
    paths.append("/Users/mathisscheffler/Desktop/pinocchio-minimal-main/build/python")
else:
    raise RuntimeError(f"Système non supporté : {system}")
for p in paths:
    if os.path.exists(p):
        if p not in sys.path:
            sys.path.insert(0, p)
import tartempion
import numpy as np
import torch
from torch.autograd import Function
import pinocchio as pin
from colorama import Fore, Style, init

init(autoreset=True)

Joint_ID = 15
tartempion.check()


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
            # this is not implemented yet
            raise
            A_np = None
            b_np = None
            articular_speed: np.ndarray = tartempion.forward_pass(
                ctx.workspace, p_np, states_init, rmodel, ctx.n_thread, targets, dt
            )
        ctx.articular_speed = p_np
        return torch.from_numpy(articular_speed).to(torch.float64).to(ctx.device)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        grad_output = torch.zeros(ctx.batch_size, ctx.seq_len, 2 * ctx.dof + ctx.eq_dim)
        tartempion.backward_pass(
            ctx.workspace,
            ctx.rmodel,
            grad_output.cpu().numpy(),
            ctx.n_thread,
            grad_output.size(0),
        )
        p_ = np.array(ctx.workspace.grad_p())
        p_ = np.reshape(p_, (ctx.batch_size, ctx.seq_len, 6))
        p_tensor = torch.from_numpy(p_).to(ctx.device).to(torch.float64)

        nan_mask = torch.isnan(p_tensor).any(dim=tuple(range(1, p_tensor.dim())))
        nan_indices = nan_mask.nonzero(as_tuple=True)[0]

        if len(nan_indices) > 0:
            for idx in nan_indices:
                print(f"NaN detected in batch {idx.item()}:")
                print(ctx.p[idx, 0])
                print(ctx.q[idx])
                print(ctx.target[idx])
            print(ctx.p[0, 0])
            exit(1)
        mask = p_tensor.abs() > 1e5
        if mask.any():
            print(
                Fore.YELLOW
                + f"Gradient has norm greater than 1e5, stopping training"
            )

        return (
            None,
            p_tensor,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
