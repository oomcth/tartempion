import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from dataset_to_torch import TrajectoryDataset as MyDataset
from dataset_to_torch import TrajectoryDataset, custom_collate_fn
import sys
import os
from collections import deque
import platform
from peft import LoraConfig, get_peft_model  # type: ignore
from tqdm import tqdm
from autonorm import torch_normalizer
from autobias import torch_SE3_Inductive_bias
from autoloss import torch_SE3_loss
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    Gemma3ForCausalLM,
)
from torch.optim.lr_scheduler import LambdaLR
import pinocchio as pin
import re

print("version datasetbalanced")

# torch.set_default_dtype(torch.float64)
dtype = torch.float64
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

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if False and torch.mps.is_available() else "cpu"
)
print(device)
print("key5")

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

        model = get_peft_model(model, lora_config)
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
    # theta : (...,) angles en radians
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
    return R  # (..., 3, 3)


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

    # Calcule un troisième axe orthogonal au premier et au second brut
    b3 = torch.cross(b1, a2, dim=-1)
    b3 = b3 / (b3.norm(dim=-1, keepdim=True) + eps)

    # Déduit le second axe comme produit vectoriel du troisième et du premier
    b2 = torch.cross(b3, b1, dim=-1)
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + eps)

    # Assemble en matrice (les 3 vecteurs sont les colonnes)
    R = torch.stack((b1, b2, b3), dim=-1)  # shape: (..., 3, 3)
    return R
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-11)
    proj = b1 * (a2 * b1).sum(dim=-1, keepdim=True)
    b2 = a2 - proj
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-11)
    b3 = torch.cross(b1, b2, dim=-1)
    b3 = b3 / (b3.norm(dim=-1, keepdim=True) + 1e-11)
    R = torch.stack((b1, b2, b3), dim=-1)
    # with torch.no_grad():
    # Q, _ = torch.linalg.qr(R)
    # det = torch.det(Q)
    # fix = (
    #     torch.eye(3, device=R.device, dtype=R.dtype)
    #     .unsqueeze(0)
    #     .repeat(Q.shape[0], 1, 1)
    # )
    # fix[:, 2, 2] = torch.sign(det)
    # R_renorm = Q @ fix
    # R = R + (R_renorm - R).detach()
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


def normalize_quaternion(q):
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)


def quaternion_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    B = q.shape[:-1]
    R = torch.empty(*B, 3, 3, device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x**2 + y**2)
    return R


def normalize_quaternion_wpos(q, eps=1e-8):
    """
    q : (..., 4)  format [w, x, y, z]
    -> (..., 4)  quaternion unitaire avec w >= 0
    """

    # Normalisation numérique stable
    q = q / (q.norm(dim=-1, keepdim=True).clamp(min=eps))

    # Force w >= 0  (supprime la double couverture +q / -q)
    sign = torch.sign(q[..., 0])
    # Remplace 0 par +1 pour ne pas multiplier par 0
    sign[sign == 0] = 1.0
    q = q * sign.unsqueeze(-1)
    return q


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
        self, sentence, start_motion, q_start, target_placement, start_position
    ):
        embedding_t, embedding_R = self.Qwen(sentence)
        # llm_out = self.emb_enc(embedding)
        # data = self.net(llm_out)

        t = self.t_proj(embedding_t / 1000)
        data = self.R_proj(embedding_R)
        a1 = data[:, :3]
        a2 = data[:, 3:]
        # q = data[:, :4]
        # R = quaternion_to_matrix(normalize_quaternion_wpos(q))
        R = mat_from_a1a2(a1, a2)
        # R = euler_rotation(a1[..., 0], a1[..., 1], a1[..., 2])

        out = logSE3(R, t)
        out = torch_SE3_Inductive_bias.apply(
            out, start_position, Inductive_bias_workspace
        )
        out = torch_normalizer.apply(out, normalizer, 0.7, 0.2)
        return (
            torch_SE3_loss.apply(target_placement.to(dtype), out, SE3_loss_workspace),
            out,
            target_placement,
            q_start,
        )


target = torch.randn(batch_size, 6).to(device).to(dtype)

with open("train_data_local_all.pkl", "rb") as f:
    train_data = pickle.load(f)

with open("test_data_local_all.pkl", "rb") as f:
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

print("loading model")
model = MLP().to(device).to(dtype)
model.Qwen.to(torch.bfloat16)
criterion = nn.MSELoss()
optimizer = optim.AdamW(
    model.parameters(),
    weight_decay=1e-5,
    lr=5e-4,
)
# optimizer = optim.SGD(
#     model.parameters(),
#     weight_decay=0e-5,
#     lr=5e-3,
# )


def lr_lambda(step):
    if step < 300:
        return 1.0  # car 1e-4 * 10 = 1e-3
    if step < 600:
        return 1.0  # car 1e-4 * 10 = 1e-3
    return 1  # ensuite lr normal


scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

num_epochs = 1000
save_dir = "debug_batches"
os.makedirs(save_dir, exist_ok=True)
print("training v2")
num_epochs = 1000
running_loss = 0.0
normalizer = tartempion.Normalizer()
Inductive_bias_workspace = tartempion.SE3_Inductive_Bias()
SE3_loss_workspace = tartempion.SE3_loss_workspace()

alpha = 0.025
last_loss = 100000
recent_batches = deque(maxlen=5)
b = None

# b = torch.load("/Users/mscheffl/Desktop/el/epoch0_step11_batch3.pth")


def print_colored_outputs(output, batch, key="sentence"):
    from colorama import Fore, Style, init

    units = [
        "mm",
        "millimeter",
        "millimeters",
        "cm",
        "centimeter",
        "centimeters",
        "dm",
        "decimeter",
        "decimeters",
    ]

    sentences = batch[key]
    for i, sent in enumerate(sentences):
        text = sent.lower()
        if any(u in text for u in units):
            color = Fore.BLUE
        else:
            color = Style.RESET_ALL
        print(f"{color}{sent:<50} --> {output[i].item():.4f}{Style.RESET_ALL}")


def compute_loss(output, batch, mode="all", verbose=True):
    """
    Calcule la loss sur 'translation', 'rotation', 'static' ou 'all'.
    """
    units_translation = [
        "mm",
        "millimeter",
        "millimeters",
        "cm",
        "centimeter",
        "centimeters",
        "dm",
        "decimeter",
        "decimeters",
    ]
    units_rotation = ["degree", "degrees", "°"]

    sentences = batch["sentence"]

    mask_translation = torch.tensor(
        [any(u in s.lower() for u in units_translation) for s in sentences],
        device=output.device,
    )

    mask_rotation = torch.tensor(
        [any(u in s.lower() for u in units_rotation) for s in sentences],
        device=output.device,
    )

    # tout le reste est "immobile"
    mask_static = ~(mask_translation | mask_rotation)

    if mode == "translation":
        selected = output[mask_translation]
    elif mode == "rotation":
        selected = output[mask_rotation]
    elif mode == "static":
        selected = output[mask_static]
    else:  # "all"
        selected = output

    if verbose:
        n_t = mask_translation.sum().item()
        n_r = mask_rotation.sum().item()
        n_s = mask_static.sum().item()
        print(f"[Batch] translation={n_t}, rotation={n_r}, static={n_s}")

    if selected.numel() == 0:
        # éviter erreur si aucun exemple de ce type
        return torch.tensor(0.0, device=output.device, requires_grad=True)

    loss = selected.mean()
    print(selected.min())
    print(selected.max())
    return loss


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

        # delta_motion = torch.zeros_like(end_motion)
        # for i in range(len(delta_motion)):
        #     delta_motion[i] = (
        #         torch.from_numpy(
        #             pin.log(  # type: ignore
        #                 pin.exp6(
        #                     pin.Motion(start_motion[i].detach().cpu().numpy())
        #                 ).actInv(
        #                     pin.exp6(pin.Motion(end_motion[i].detach().cpu().numpy()))
        #                 )
        #             ).vector
        #         )
        #         .to(torch.float64)
        #         .to(device)
        #     )

        output, out, target_placement, q_start = model(
            embedding,
            start_motion.float(),
            q_start.float(),
            end_motion,
            batch["start_SE3"],
        )
        loss = output.mean()
        # loss = compute_loss(output, batch, "rotation")
        # print_colored_outputs(output, batch)
        if loss.item() > last_loss + 0.1:
            running_loss = 100000
        if running_loss == 0.0:
            running_loss = loss.item()
        else:
            running_loss = alpha * running_loss + (1 - alpha) * loss.item()
            running_loss = 100000

        recent_batches.append(
            {
                "output": output,
                "out": out,
                "target": target_placement,
                "embedding": embedding,
                "start_motion": start_motion.cpu(),
                "end_motion": end_motion.cpu(),
                "q_start": q_start.cpu(),
                "end_SE3": end_placement,
                "loss": loss.item(),
                "epoch": epoch,
                "step": step,
            }
        )

        if False and (loss.item() > 10 * running_loss or loss.item() > last_loss * 5):
            print(
                f"\n🔥 Pic de loss détecté : {loss.item():.6f} (10x la running loss {running_loss:.6f})"
            )
            print(f"--> Sauvegarde des {len(recent_batches)} derniers mini‑batches")

            for i, bdata in enumerate(list(recent_batches)[-5:]):
                save_path = os.path.join(
                    save_dir, f"epoch{epoch}_step{step}_batch{i}.pth"
                )
                torch.save(
                    {
                        "batch": bdata,
                    },
                    save_path,
                )
                print(f"Batch sauvegardé : {save_path}")
        last_loss = loss.item()
        loss.backward()
        print(loss.item())
        with torch.no_grad():
            g = model.Qwen.last_token_activations.grad
            norms = g.norm(dim=1)
            print(
                f"loss={loss.item():.8f} | "
                f"grad_norm_mean={model.Qwen.last_token_activations.grad.norm(dim=1).mean():.2e}, "
                f"min={model.Qwen.last_token_activations.grad.norm(dim=1).min():.2e}, "
                f"max={model.Qwen.last_token_activations.grad.norm(dim=1).max():.2e}",
            )
        if step % 1 == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

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
            )
            loss = output.mean()
            print("### val loss", loss.item())
            optimizer.zero_grad()
            val_loss += loss.item() * len(embedding)

    # avg_val_loss = val_loss / len(test_loader.dataset)
    # print(f"Epoch {epoch+1}/{num_epochs} Validation Loss: {avg_val_loss:.6f}")
