import tartempion
import numpy as np
import torch
from torch.autograd import Function
import pinocchio as pin


class torch_SE3_Inductive_bias(Function):
    @staticmethod
    def forward(
        ctx,
        log_position: torch.Tensor,
        start_position: list[pin.SE3],
        SE3_Inductive_Bias: tartempion.SE3_Inductive_Bias,
    ):
        assert log_position.size(-1) == 6
        ctx.SE3_Inductive_Bias = SE3_Inductive_Bias
        ctx.reshape = False
        log_position_np = log_position.detach().cpu().numpy()
        log_position_np = log_position_np.astype(np.float64)

        final_pos = SE3_Inductive_Bias.Inductive_Bias(log_position_np, start_position)

        return (
            torch.from_numpy(final_pos).to(log_position.device).to(log_position.dtype)
        )

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        assert grad_output.size(-1) == 6

        grad = np.array(
            ctx.SE3_Inductive_Bias.d_Inductive_Bias(
                grad_output.detach().cpu().to(torch.float64).numpy()
            )
        )

        return (
            torch.from_numpy(grad).to(grad_output.device).to(grad_output.dtype),
            None,
            None,
        )
