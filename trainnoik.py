import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from dataset_to_torch import TrajectoryDataset as MyDataset
from dataset_to_torch import TrajectoryDataset, custom_collate_fn
import sys
from autogradQP import QPkkt
import os
from collections import deque
import platform
import example_robot_data as erd
from peft import LoraConfig, get_peft_model  # type: ignore
from tqdm import tqdm
import time
from autonorm import torch_normalizer
from autobias import torch_SE3_Inductive_bias
from autoloss import torch_SE3_loss
import viewer
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
)
import pinocchio as pin
import re


dtype = torch.float64
system = platform.system()
if not system == "Linux":
    import meshcat.geometry as g
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

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if False and torch.mps.is_available() else "cpu"
)
print(device)

batch_size = 256
collate_fn = custom_collate_fn


def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            print(f"Trainable: {name} | shape {param.shape}")
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Percentage trainable: {100 * trainable / total:.2f}%")


def get_gemma():
    if system == "Linux":
        work_dir = os.environ.get("WORK")
        if work_dir is None:
            raise RuntimeError("L'environnement WORK n'est pas défini")

        save_dir = os.path.join(work_dir, "model_saves")

        model = Gemma3ForCausalLM.from_pretrained(save_dir, attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.0,
            bias="lora_only",
            task_type="CAUSAL_LM",
        )
        # model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
        print("initial dtype", model.dtype)
        model = model.to(device).to(dtype)
        return model, tokenizer
    else:
        model_name = "google/gemma-3-1b-pt"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            use_auth_token=token,
            attn_implementation="eager",
        )
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # model = get_peft_model(model, lora_config)
        model = model.to(device).to(dtype)
        print_trainable_parameters(model)
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
        last_hidden_state2 = outputs.hidden_states[-1]
        self.last_token_activations = last_hidden_state[:, -1, :].double()
        last_token_activations2 = last_hidden_state2[:, -1, :].double()
        if self.last_token_activations.requires_grad:
            self.last_token_activations.retain_grad()
        return (
            self.layernorm2(last_token_activations2.double()),
            self.layernorm(self.last_token_activations),
        )


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

        t = self.t_proj(embedding_t)
        data = self.R_proj(embedding_R)
        a1 = data[:, :3]
        a2 = data[:, 3:]
        R = mat_from_a1a2(a1, a2)

        out = logSE3(R, t)
        out = torch_SE3_Inductive_bias.apply(
            out, start_position, Inductive_bias_workspace
        )
        out = torch_normalizer.apply(out, normalizer, 0.7, 0.2)
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


target = torch.randn(batch_size, 6).to(device).to(dtype)

with open("test_qp.pkl", "rb") as f:
    train_data = pickle.load(f)

with open("test_qp.pkl", "rb") as f:
    test_data = pickle.load(f)

print("load data done")

train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

model = MLP().to(device).to(dtype)
if not system == "Linux":
    print("loading model")
    checkpoint_path = "/Users/mathisscheffler/Desktop/checkpoint/run2_checkpoint_epoch_79.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    print(checkpoint["loss"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
model.eval()
model.Qwen.to(torch.bfloat16)
criterion = nn.MSELoss()
optimizer = optim.AdamW(
    model.parameters(),
    weight_decay=1e-5,
    lr=1e-4,
)



num_epochs = 1000
save_dir = "debug_batches"
os.makedirs(save_dir, exist_ok=True)
print("training v2")
num_epochs = 1000
running_loss = 0.0
normalizer = tartempion.Normalizer()
Inductive_bias_workspace = tartempion.SE3_Inductive_Bias()
SE3_loss_workspace = tartempion.SE3_loss_workspace()

q_reg = 1e-2
bound = -1000
workspace = tartempion.QPworkspace()
workspace.set_q_reg(q_reg)
workspace.set_bound(bound)
workspace.set_lambda(-2)
workspace.set_L1(0.00)
workspace.set_rot_w(1.0)
collate_fn = custom_collate_fn
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

A_np = np.zeros((batch_size * seq_len, eq_dim, 6)).astype(np.float64)
b_np = np.zeros((batch_size, seq_len, 1)).astype(np.float64)
A_np = torch.from_numpy(A_np)
A_np = A_np.reshape(-1, 1, 6).requires_grad_(True)
b_np = torch.from_numpy(b_np)
b_np = b_np.reshape(-1, 1).requires_grad_(True)

if not system == "Linux":
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

b = None


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for step, batch in tqdm(enumerate(train_loader)):
        # if b is None:
        #     b = batch
        # batch = b
        embedding = batch["sentence"]
        start_motion = torch.stack(
            [
                torch.tensor(motion.vector, dtype=torch.float32)
                for motion in batch["start_motion"]
            ]
        )
        end_motion = torch.stack(
            [
                torch.tensor(motion.vector, dtype=torch.float32)
                for motion in batch["end_motion"]
            ]
        )
        q_start = batch["q_start"]
        end_placement = batch["end_SE3"]

        start_motion = start_motion.to(device)
        q_start = q_start.to(device)
        end_motion = end_motion.to(device)

        output, out, target_placement, q_start = model(
            embedding,
            start_motion.float(),
            q_start.float(),
            end_motion,
            batch["start_SE3"],
            end_placement,
        )
        loss = output.mean()

        loss.backward()
        print("mean", loss.item())
        print("median", torch.median(output))
        if not system == "Linux":
            print(batch["sentence"])
            arr = np.array(workspace.get_q())
            q0 = arr[0, 0]
            pin.framesForwardKinematics(rmodel, rmodel.data, q0)
            viz.viz.viewer["start"].set_transform(
                rmodel.data.oMf[tool_id].homogeneous.copy()
            )
            viz.viz.viewer["ideal"].set_transform(end_placement[0].homogeneous)
            for plot_time in range(0, arr.shape[1]):
                viz.display(arr[0, plot_time])
                pin.framesForwardKinematics(rmodel, rmodel.data, arr[0, plot_time])
                viz.viz.viewer["current"].set_transform(
                    rmodel.data.oMf[tool_id].homogeneous.copy()
                )
                time.sleep(dt)
        total_loss += loss.item() * len(embedding)

    avg_loss = total_loss / len(train_loader.dataset)  # type: ignore
    print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {avg_loss:.6f}")
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            embedding = batch["sentence"]
            start_motion = batch["start_motion"]
            start_motion = torch.stack(
                [
                    torch.tensor(motion.vector, dtype=torch.float32)
                    for motion in start_motion
                ]
            )
            q_start = batch["q_start"]
            end_motion = batch["end_motion"]
            end_motion = torch.stack(
                [
                    torch.tensor(motion.vector, dtype=torch.float32)
                    for motion in end_motion
                ]
            )
            start_motion = start_motion.to(device)
            q_start = q_start.to(device)
            end_motion = end_motion.to(device)
            end_placement = batch["end_SE3"]

            output, out, target_placement, q_start = model(
                embedding,
                start_motion.float(),
                q_start.float(),
                end_motion,
                batch["start_SE3"],
                end_placement,
            )
            loss = output.mean()
            print("### val loss", loss.item())
            optimizer.zero_grad()
            val_loss += loss.item() * len(embedding)

    avg_val_loss = val_loss / len(test_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} Validation Loss: {avg_val_loss:.6f}")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
    }
    # torch.save(checkpoint, f"run3_checkpoint_epoch_{epoch}.pt")
