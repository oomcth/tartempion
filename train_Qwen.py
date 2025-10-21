import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from dataset_to_torch import TrajectoryDataset as MyDataset
from dataset_to_torch import TrajectoryDataset, custom_collate_fn
from autogradQP import QPkkt
import sys
import os
from collections import deque
import platform
from peft import LoraConfig, get_peft_model
import example_robot_data as erd
from tqdm import tqdm
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    Gemma3ForCausalLM,
)
from autonorm import torch_normalizer
import re

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
print("key3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_reg = 1e-2
bound = -1000
kinematic_workspace = tartempion.KinematicsWorkspace()
normalizer = tartempion.Normalizer()
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
batch_size = 128
eq_dim = 1
n_threads = 20
os.environ["OMP_PROC_BIND"] = "spread"

A_np = np.zeros((batch_size * seq_len, eq_dim, 6)).astype(np.float64)
b_np = np.zeros((batch_size, seq_len, 1)).astype(np.float64)
A_np = torch.from_numpy(A_np)
A_np = A_np.reshape(-1, 1, 6).requires_grad_(True)
b_np = torch.from_numpy(b_np)
b_np = b_np.reshape(-1, 1).requires_grad_(True)


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


def clean_floats_in_sentence(sentence, decimals=3):
    def replacer(match):
        num = float(match.group())
        return f"{num:.{decimals}f}"

    pattern = r"\d+\.\d+"
    cleaned_sentence = re.sub(pattern, replacer, sentence)
    return cleaned_sentence


def get_qwen():
    if system == "Linux":
        local_cache_dir = "/lustre/fswork/projects/rech/tln/urh44lu/model/saves/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            local_cache_dir, device_map="auto", torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(local_cache_dir)
        return model, tokenizer
    elif system == "Darwin":  # macOS
        model_name = "google/gemma-3-1b-pt"
        token = ""
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            use_auth_token=token,
            attn_implementation="eager",
        )
        return model, tokenizer


def get_gemma():
    if system == "Linux":
        work_dir = os.environ.get("WORK")
        if work_dir is None:
            raise RuntimeError("L'environnement WORK n'est pas défini")

        save_dir = os.path.join(work_dir, "model_saves")

        model = Gemma3ForCausalLM.from_pretrained(save_dir, attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
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
        print_trainable_parameters(model)

        model = model.to(device)
        return model, tokenizer
    else:
        model_name = "google/gemma-3-1b-pt"
        token = ""
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
        model = model.to(device)
        print_trainable_parameters(model)
        exit()
        return model, tokenizer


class Gemma3ActivationLayer(nn.Module):
    def __init__(self, model_name="google/gemma-3-1b-pt"):
        super(Gemma3ActivationLayer, self).__init__()
        self.model, self.tokenizer = get_gemma()

    def forward(self, sentence: str) -> torch.Tensor:
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        outputs = self.model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        # last_hidden_state = outputs.hidden_states[-1]
        last_hidden_state = outputs.hidden_states[15]
        last_token_activations = last_hidden_state[:, -1, :]
        # return last_token_activations.float()
        return torch.sigmoid(0.2 * last_token_activations.float()) * 0.25


class QwenActivationLayer(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        super(QwenActivationLayer, self).__init__()
        self.model, self.tokenizer = get_qwen()

    def forward(self, sentence: str) -> torch.Tensor:
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        outputs = self.model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        last_token_activations = last_hidden_state[:, -1, :]
        raise
        return torch.sigmoid(0.2 * last_token_activations.float()) * 0.25


class MLP(nn.Module):  # gemma : 1152 ; gwen 2.5-3b = 2048
    def __init__(self, embedding_dim=1152, motion_dim=6, q_dim=6, hidden_dim=256):
        super().__init__()
        self.Qwen = Gemma3ActivationLayer()
        # self.Qwen = QwenActivationLayer()
        self.emb_enc = nn.Linear(embedding_dim, hidden_dim)
        self.motion_enc = nn.Linear(6, hidden_dim)
        self.q_enc = nn.Linear(6, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, motion_dim),
        )
        self.Qwen.to(device)
        # self.net = ParametrizedTransformerAggregator(hidden_dim)

    def forward(self, sentence, start_motion, q_start, target_placement):
        sentence = [clean_floats_in_sentence(s) for s in sentence]
        embedding = self.Qwen(sentence)
        embedding = embedding.to(start_motion.device)
        x = torch.cat(
            [
                self.emb_enc(embedding),
                self.motion_enc(start_motion),
                # self.q_enc(q_start),
            ],
            dim=1,
        )
        out = self.net(x)
        out = torch_normalizer.apply(out, normalizer, 0.7, 0.2)
        out = out.to("cpu")
        A_np = np.zeros((x.size(0) * seq_len, eq_dim, 6)).astype(np.float64)
        b_np = np.zeros((x.size(0), seq_len, 1)).astype(np.float64)
        A_np = torch.from_numpy(A_np)
        A_np = A_np.reshape(-1, 1, 6).requires_grad_(True)
        b_np = torch.from_numpy(b_np)
        b_np = b_np.reshape(-1, 1).requires_grad_(True)
        print("max out", torch.max(torch.norm(out, dim=1)).item())
        return (
            QPkkt.apply(
                q_start.detach().cpu().numpy(),
                out.unsqueeze(1).repeat(1, seq_len, 1),
                A_np * 0,
                b_np * 0,
                rmodel,
                workspace,
                x.size(0),
                seq_len,
                eq_dim,
                target_placement,
                dt,
                40,
            ),
            out,
            target_placement,
            q_start,
        )


with open("train_dataset_safe.pkl", "rb") as f:
    train_data = pickle.load(f)

with open("test_dataset_safe.pkl", "rb") as f:
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
model = MLP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(
    [
        {"params": model.Qwen.parameters(), "lr": 1e-4},
        {"params": model.net.parameters(), "lr": 1e-4},
        {"params": model.emb_enc.parameters(), "lr": 1e-4},
        {"params": model.motion_enc.parameters(), "lr": 1e-4},
        {"params": model.q_enc.parameters(), "lr": 1e-4},
    ],
    weight_decay=1e-5,
    lr=1e-4,
)


def lr_lambda(step):
    if step < 300:
        return 1.0  # car 1e-4 * 10 = 1e-3
    return 1.0  # ensuite lr normal


from torch.optim.lr_scheduler import LambdaLR

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

num_epochs = 1000
save_dir = "debug_batches"
os.makedirs(save_dir, exist_ok=True)
print("training v2")
num_epochs = 1000
running_loss = 0.0
alpha = 0.025
last_loss = 100000
recent_batches = deque(maxlen=5)
b = None
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for step, batch in tqdm(enumerate(train_loader)):
        # Tensorisation
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

        # Envoi sur device
        start_motion = start_motion.to(device)
        q_start = q_start.to(device)
        end_motion = end_motion.to(device)

        # Forward
        optimizer.zero_grad()
        output, out, target_placement, q_start = model(
            embedding, start_motion.float(), q_start.float(), end_placement
        )
        loss = output.mean()
        print(loss.item())

        # Update running loss (EMA)
        if running_loss == 0.0:
            running_loss = loss.item()
        else:
            running_loss = alpha * running_loss + (1 - alpha) * loss.item()

        # Sauvegarde batch courant dans la deque (pour potentielle dump)
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

        # Vérifie si la loss explose
        if loss.item() > 10 * running_loss or loss.item() > last_loss * 5:
            print(
                f"\n🔥 Pic de loss détecté : {loss.item():.6f} (10x la running loss {running_loss:.6f})"
            )
            print(f"--> Sauvegarde des {len(recent_batches)} derniers mini‑batches")

            # Sauvegarde chaque batch de la deque
            for i, bdata in enumerate(
                list(recent_batches)[-5:]
            ):  # les 5 derniers seulement
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
        # Backward + update
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item() * len(embedding)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {avg_loss:.6f}")
    model.eval()
    val_loss = 0.0
    # with torch.no_grad():
    for batch in test_loader:
        break
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
            [torch.tensor(motion.vector, dtype=torch.float32) for motion in end_motion]
        )
        start_motion = start_motion.to(device)
        q_start = q_start.to(device)
        end_motion = end_motion.to(device)
        end_placement = batch["end_SE3"]

        output = model(
            embedding,
            start_motion.float(),
            q_start.float(),
            end_placement,
        )
        loss = output.mean()
        loss.backward()
        print("### val loss", loss.item())
        optimizer.zero_grad()
        val_loss += loss.item() * len(embedding)

    # avg_val_loss = val_loss / len(test_loader.dataset)
    # print(f"Epoch {epoch+1}/{num_epochs} Validation Loss: {avg_val_loss:.6f}")
