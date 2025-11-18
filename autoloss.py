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


Joint_ID = 15
tartempion.check()


class torch_SE3_loss(Function):
    @staticmethod
    def forward(
        ctx,
        fixed: torch.Tensor,
        updated: torch.Tensor,
        SE3_loss_workspace: tartempion.SE3_loss_workspace,
    ):
        fixed_np = fixed.detach().cpu().numpy()
        updated_np = updated.detach().cpu().numpy()
        ctx.SE3_loss_workspace = SE3_loss_workspace
        out = SE3_loss_workspace.SE3_loss(updated_np, fixed_np, 1)
        return torch.from_numpy(out).to(fixed.device).to(fixed.dtype)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        grad = np.array(ctx.SE3_loss_workspace.d_SE3_loss())
        return (
            None,
            torch.from_numpy(grad).to(grad_output.device).to(grad_output.dtype),
            None,
        )
