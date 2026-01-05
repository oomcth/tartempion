import tartempion
import numpy as np
import torch
from torch.autograd import Function


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
        out = SE3_loss_workspace.SE3_loss(updated_np, fixed_np)
        return torch.from_numpy(out).to(fixed.device).to(fixed.dtype)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        grad = np.array(ctx.SE3_loss_workspace.d_SE3_loss())
        has_nan = np.isnan(grad).any()
        has_large = np.abs(grad).max() > 1e-5
        if has_nan or has_large:
            raise
        return (
            None,
            torch.from_numpy(grad).to(grad_output.device).to(grad_output.dtype),
            None,
        )
