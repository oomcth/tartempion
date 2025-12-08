import numpy as np
import torch
import torch.nn as nn
import sys
from autogradQP import QPkkt
import os
import platform
import example_robot_data as erd
import time
from autonorm import torch_normalizer
from autobias import torch_SE3_Inductive_bias
import viewer
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
)
import pinocchio as pin
import tartempion

pin.seed(21)
np.random.seed(21)

dtype = torch.float64
system = platform.system()
if not system == "Linux":
    import meshcat.geometry as g

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if False and torch.mps.is_available()
    else "cpu"
)


def get_gemma():
    model_name = "google/gemma-3-1b-pt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Gemma3ForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    model = model.to(device).to(dtype)
    return model, tokenizer


class Gemma3ActivationLayer(nn.Module):
    def __init__(self, model_name="google/gemma-3-1b-pt"):
        super(Gemma3ActivationLayer, self).__init__()
        self.model, self.tokenizer = get_gemma()
        self.layernorm = nn.LayerNorm(1152)
        self.layernorm2 = nn.LayerNorm(1152)
        self.last_token_activations = None

    def forward(self, sentence: str) -> torch.Tensor:
        if self.layernorm.weight.dtype != torch.float64:
            self.layernorm.to(torch.float64)
            self.layernorm2.to(torch.float64)
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = (
            inputs["attention_mask"].to(self.model.device).to(self.model.dtype)
        )

        outputs = self.model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        last_hidden_state2 = outputs.hidden_states[18]
        self.last_token_activations = last_hidden_state[:, -1, :].double()
        last_token_activations2 = last_hidden_state2[:, -1, :].double()
        if self.last_token_activations.requires_grad:
            self.last_token_activations.retain_grad()
        return (
            self.layernorm2(last_token_activations2.double()),
            self.layernorm(self.last_token_activations),
        )


def rot_x(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack(
        [
            torch.stack(
                [torch.ones_like(c), torch.zeros_like(c), torch.zeros_like(c)], dim=-1
            ),
            torch.stack([torch.zeros_like(c), c, -s], dim=-1),
            torch.stack([torch.zeros_like(c), s, c], dim=-1),
        ],
        dim=-2,
    )
    return R


def rot_y(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack(
        [
            torch.stack([c, torch.zeros_like(c), s], dim=-1),
            torch.stack(
                [torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)], dim=-1
            ),
            torch.stack([-s, torch.zeros_like(c), c], dim=-1),
        ],
        dim=-2,
    )
    return R


def euler_rotation(rx, ry, rz):
    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    return R


def rot_z(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack(
        [
            torch.stack([c, -s, torch.zeros_like(c)], dim=-1),
            torch.stack([s, c, torch.zeros_like(c)], dim=-1),
            torch.stack(
                [torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], dim=-1
            ),
        ],
        dim=-2,
    )
    return R


def mat_from_a1a2(a1, a2):
    eps = 1e-10
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + eps)
    b3 = torch.cross(b1, a2, dim=-1)
    b3 = b3 / (b3.norm(dim=-1, keepdim=True) + eps)
    b2 = torch.cross(b3, b1, dim=-1)
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + eps)
    R = torch.stack((b1, b2, b3), dim=-1)
    return R


def hat(v):
    O = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype, device=v.device)
    O[..., 0, 1] = -v[..., 2]
    O[..., 0, 2] = v[..., 1]
    O[..., 1, 0] = v[..., 2]
    O[..., 1, 2] = -v[..., 0]
    O[..., 2, 0] = -v[..., 1]
    O[..., 2, 1] = v[..., 0]
    return O


def logSO3(R, eps=1e-12):
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    logR = torch.zeros_like(R)
    mask = sin_theta.abs() > eps
    logR[mask] = (theta[mask] / (2 * sin_theta[mask]))[..., None, None] * (
        R[mask] - R[mask].transpose(-2, -1)
    )
    logR[~mask] = 0.5 * (R[~mask] - R[~mask].transpose(-2, -1))
    return logR, theta


def omega_from_logR(logR):
    wx = logR[..., 2, 1] - logR[..., 1, 2]
    wy = logR[..., 0, 2] - logR[..., 2, 0]
    wz = logR[..., 1, 0] - logR[..., 0, 1]
    return 0.5 * torch.stack((wx, wy, wz), dim=-1)


def logSE3(R, t, eps=1e-14):
    B = R.shape[0]

    logR, theta = logSO3(R, eps)
    omega = omega_from_logR(logR)
    Omega = hat(omega)

    Id = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).expand(B, 3, 3)

    theta = theta.view(B, 1, 1)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    theta2 = theta * theta
    theta3 = theta2 * theta

    term1 = (1 - cos_theta) / (theta2 + eps)
    term2 = (theta - sin_theta) / (theta3 + eps)

    V = Id + term1 * Omega + term2 * (Omega @ Omega)
    V_inv = torch.linalg.inv(V)
    v = torch.bmm(V_inv, t.unsqueeze(-1)).squeeze(-1)
    pin_like_log = torch.cat([v, omega], dim=-1)
    return pin_like_log


class MLP(nn.Module):  # gemma : 1152 ; gwen 2.5-3b = 2048
    def __init__(self, embedding_dim=1152, motion_dim=9, q_dim=6, hidden_dim=1024):
        super().__init__()
        self.Qwen = Gemma3ActivationLayer()
        self.emb_enc = nn.Linear(embedding_dim, hidden_dim)
        self.motion_enc = nn.Linear(6, hidden_dim)
        self.q_enc = nn.Linear(6, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, motion_dim),
        )
        self.R_proj = nn.Linear(embedding_dim, 6)
        self.t_proj = nn.Linear(embedding_dim, 3)
        self.Qwen.to(device)

    def forward(
        self,
        sentence,
        start_motion,
        q_start,
        target_placement,
        start_position,
        end_placement=None,
    ):
        embedding_t, embedding_R = self.Qwen(sentence)

        t = self.t_proj(embedding_t / 1000)
        data = self.R_proj(embedding_R)
        a1 = data[:, :3]
        a2 = data[:, 3:]
        R = mat_from_a1a2(a1, a2)

        out = logSE3(R, t)
        out = torch_SE3_Inductive_bias.apply(
            out, start_position, Inductive_bias_workspace
        )
        out = torch_normalizer.apply(out, normalizer, 1.1, 0.2)
        out = out.cpu()
        A_np = np.zeros((len(end_placement) * seq_len, eq_dim, 6)).astype(np.float64)
        b_np = np.zeros((len(end_placement), seq_len, 1)).astype(np.float64)
        A_np = torch.from_numpy(A_np)
        A_np = A_np.reshape(-1, 1, 6).requires_grad_(True)
        b_np = torch.from_numpy(b_np)
        b_np = b_np.reshape(-1, 1).requires_grad_(True)
        return (
            QPkkt.apply(
                q_start.detach().cpu().numpy(),
                out.unsqueeze(1).repeat(1, seq_len, 1),
                A_np * 0,
                b_np * 0,
                rmodel,
                workspace,
                len(end_placement),
                seq_len,
                eq_dim,
                end_placement,
                dt,
                40,
            ),
            out,
            target_placement,
            q_start,
        )


print("loading model")
model = MLP().to(device).to(dtype)
if not system == "Linux":
    checkpoint_path = (
        "/Users/mathisscheffler/Desktop/checkpoint/run2_checkpoint_epoch_79.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    print(checkpoint["loss"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
model.eval()
model.Qwen.to(torch.bfloat16)

normalizer = tartempion.Normalizer()
Inductive_bias_workspace = tartempion.SE3_Inductive_Bias()
SE3_loss_workspace = tartempion.SE3_loss_workspace()

q_reg = 1e-3
bound = -1000
workspace = tartempion.QPworkspace()
workspace.set_q_reg(q_reg)
workspace.set_bound(bound)
workspace.set_lambda(-2)
workspace.set_L1(0.00)
workspace.set_rot_w(1.0)

robot = erd.load("ur5")
rmodel, gmodel, vmodel = robot.model, robot.collision_model, robot.visual_model
rmodel.data = rmodel.createData()
tool_id = 21
workspace.set_tool_id(tool_id)
seq_len = 1500
dt = 1e-2
eq_dim = 1
n_threads = 50
os.environ["OMP_PROC_BIND"] = "spread"

viz = viewer.Viewer(rmodel, gmodel, vmodel, True)
viz.viz.viewer["start"].set_object(
    g.Sphere(0.01),
    g.MeshLambertMaterial(color=0x0000FF, transparent=True, opacity=0.5),
)
viz.viz.viewer["ideal"].set_object(
    g.Sphere(0.01),
    g.MeshLambertMaterial(color=0x00FFFF, transparent=True, opacity=0.5),
)
viz.viz.viewer["current"].set_object(
    g.Sphere(0.01),
    g.MeshLambertMaterial(color=0xFFFF00, transparent=True, opacity=0.5),
)

model.eval()


def sample_p_start():
    while True:
        q = pin.randomConfiguration(rmodel)
        pin.framesForwardKinematics(rmodel, rmodel.data, q)
        T = rmodel.data.oMf[tool_id]
        if T.translation[2] > 0.2:
            return q


q = sample_p_start()
viz.display(q)


while True:
    viz.display(q)
    embedding = input("Provide robot prompt : ")
    embedding = [embedding]
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    start_motion = torch.from_numpy(
        np.array([pin.log6(rmodel.data.oMf[tool_id]).vector])
    )
    q_start = torch.from_numpy(np.array([q]))
    end_placement = [rmodel.data.oMf[tool_id]]

    start_motion = start_motion.to(device)
    q_start = q_start.to(device)

    output, out, target_placement, q_start = model(
        embedding,
        start_motion.float(),
        q_start.float(),
        start_motion,
        [rmodel.data.oMf[tool_id]],
        end_placement,
    )

    arr = np.array(workspace.get_q())
    q0 = arr[0, 0]
    pin.framesForwardKinematics(rmodel, rmodel.data, q0)
    viz.viz.viewer["start"].set_transform(rmodel.data.oMf[tool_id].homogeneous.copy())
    viz.viz.viewer["ideal"].set_transform(end_placement[0].homogeneous)
    for plot_time in range(0, arr.shape[1]):
        viz.display(arr[0, plot_time])
        pin.framesForwardKinematics(rmodel, rmodel.data, arr[0, plot_time])
        time.sleep(dt / seq_len)
    q = arr[0, -1]
