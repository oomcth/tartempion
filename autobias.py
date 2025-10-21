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


class torch_SE3_Inductive_bias(Function):
    @staticmethod
    def forward(
        ctx,
        log_position: torch.Tensor,
        start_position: list[pin.SE3],
        SE3_Inductive_Bias: tartempion.SE3_Inductive_Bias,
    ):
        ctx.SE3_Inductive_Bias = SE3_Inductive_Bias
        assert log_position.size(-1) == 6
        log_position_np = log_position.detach().cpu().numpy()
        log_position_np = log_position_np.astype(np.float64)
        ctx.reshape = False
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
        # grad_tensor = (
        #     torch.from_numpy(grad).to(grad_output.device).to(grad_output.dtype)
        # )
        # x = grad_tensor
        # normes = x.view(x.size(0), -1).norm(dim=1)

        # # 2. Trie décroissant
        # valeurs_triees, indices = torch.sort(normes, descending=True)

        # # 3. On prend les 5 plus grandes
        # top5_valeurs = valeurs_triees[:5]
        # top5_indices = indices[:5]

        # print("Indices des 5 normes max :", top5_indices.tolist())
        # print("Valeurs correspondantes :", top5_valeurs.tolist())
        return (
            torch.from_numpy(grad).to(grad_output.device).to(grad_output.dtype),
            None,
            None,
        )
