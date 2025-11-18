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


class torch_normalizer(Function):
    @staticmethod
    def forward(
        ctx,
        log_position: torch.Tensor,
        normalizer: tartempion.Normalizer,
        scale,
        min_scale,
    ):
        ctx.normalizer = normalizer
        assert log_position.size(-1) == 6
        log_position_np = log_position.detach().cpu().numpy()
        log_position_np = log_position_np.astype(np.float64)
        if len(log_position_np.shape) == 2:
            ctx.reshape = False
            normalized_pos = normalizer.normalize(log_position_np, scale, min_scale)
            if len(normalized_pos.shape) == 1:
                normalized_pos = normalized_pos[np.newaxis, :]
        elif len(log_position_np.shape) == 1:
            ctx.reshape = True
            normalized_pos = np.array(
                normalizer.normalize(log_position_np[np.newaxis, :], scale, min_scale)
            )[np.newaxis, :]
        else:
            raise Exception("invalid pos shape")
        # if log_position.dtype != torch.float64:
        #     raise
        return (
            torch.from_numpy(normalized_pos)
            .to(log_position.device)
            .to(log_position.dtype)
        )

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        assert grad_output.size(-1) == 6
        if len(grad_output.size()) == 1:
            grad_output.unsqueeze(0)

        grad = np.array(
            ctx.normalizer.d_normalize(
                grad_output.detach().cpu().to(torch.float64).numpy()
            )
        )
        if len(grad.shape) == 1:
            grad = grad[np.newaxis, :]
        return (
            torch.from_numpy(grad).to(grad_output.device).to(grad_output.dtype),
            None,
            None,
            None,
        )
