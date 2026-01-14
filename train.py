import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from dataset_to_torch import TrajectoryDataset as MyDataset
from dataset_to_torch import TrajectoryDataset, custom_collate_fn
import sys
import coal
import torch.nn.functional as F
import os
from collections import deque
import platform
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import example_robot_data as erd
from autogradQP import QPkkt
from autonorm import torch_normalizer
import meshcat.geometry as g
from autobias import torch_SE3_Inductive_bias
from autoloss import torch_SE3_loss_2
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
)
from pathlib import Path
import pinocchio as pin
import tartempion
import platform


seq_len = 1000
dt = 1e-2
eq_dim = 1
SE3_loss_workspace = tartempion.SE3_loss_workspace()
SE3_loss_workspace.set_lambda(1e-3)
if platform.system() != "Linux":
    import viewer

    DEBUG = False
    batch_size = 2
else:
    DEBUG = False
    batch_size = 256
system = platform.system()
dtype = torch.float64
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if False and torch.mps.is_available()
    else "cpu"
)

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
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
        print("initial dtype", model.dtype)
        model = model.to(device).to(dtype)
        return model, tokenizer
    else:
        model_name = "google/gemma-3-1b-pt"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="eager",
        )
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
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model = model.to(device).to(dtype)
        print_trainable_parameters(model)
        return model, tokenizer


class Layer(nn.Module):
    def __init__(self, input_dim=1152 + 1, hidden_dim=100, output_dim=6, n_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


class Gemma3ActivationLayer(nn.Module):
    def __init__(self, model_name="google/gemma-3-1b-pt"):
        super(Gemma3ActivationLayer, self).__init__()
        self.model, self.tokenizer = get_gemma()
        self.layernorm = nn.LayerNorm(1152)
        self.layernorm2 = nn.LayerNorm(1152)
        self.last_token_activations = None
        self.motion_proj = nn.Linear(6, 1152)
        self.positions_proj = nn.Linear(12, 1152)
        self.token_emb = nn.Parameter(torch.randn(6, 1152))

    def forward(self, sentence: str, start_motion, trans, rot) -> torch.Tensor:
        if self.layernorm.weight.dtype != torch.float64:
            self.layernorm.to(torch.float64)
            self.layernorm2.to(torch.float64)
        B = trans.shape[0]
        pose = torch.cat([trans.reshape(B, 6, -1), rot.reshape(B, 6, -1)], dim=-1)
        pose = pose.to(
            self.positions_proj.weight.device, self.positions_proj.weight.dtype
        )
        emb = self.positions_proj(pose)
        emb = emb + self.token_emb[None, :, :]
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        emb = emb.to(inputs_embeds.dtype).to(inputs_embeds.device)
        inputs_embeds = torch.cat([emb, inputs_embeds], dim=1)
        emb_mask = torch.ones(
            (attention_mask.size(0), emb.size(1)), device=attention_mask.device
        )
        attention_mask = torch.cat([emb_mask, attention_mask], dim=1)
        if start_motion is not None:
            self.motion_proj = self.motion_proj.to(inputs_embeds.dtype)
            motion_proj_out = self.motion_proj(start_motion.to(inputs_embeds.dtype))
            motion_embed = motion_proj_out.view(-1, 1, 1152)
            inputs_embeds = torch.cat([motion_embed, inputs_embeds], dim=1)
            motion_mask = torch.ones(
                (attention_mask.size(0), 1), device=attention_mask.device
            )
            attention_mask = torch.cat([motion_mask, attention_mask], dim=1)
        else:
            raise
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
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
        self.llm = Gemma3ActivationLayer()
        self.emb_enc = nn.Linear(embedding_dim, hidden_dim)
        self.motion_enc = nn.Linear(6, hidden_dim)
        self.q_enc = nn.Linear(6, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, motion_dim),
        )
        self.R_proj = nn.Linear(embedding_dim, 6)
        self.t_proj = nn.Linear(embedding_dim, 3)

        self.llm.to(device)

    def forward(
        self,
        sentence,
        start_motion,
        q_start,
        target_placement,
        start_position,
        all_obj_trans: torch.Tensor,
        all_obj_rot: torch.Tensor,
    ):
        embedding_t, embedding_R = self.llm(
            sentence, start_motion, all_obj_trans, all_obj_rot
        )

        t = (
            self.t_proj(
                embedding_t.to(self.t_proj.weight.device, self.t_proj.weight.dtype)
            )
            / 100
        )

        data = self.R_proj(
            embedding_R.to(self.t_proj.weight.device, self.t_proj.weight.dtype)
        )
        a1 = data[:, :3]
        a2 = data[:, 3:]
        return torch_SE3_loss_2.apply(t, a1, a2, SE3_loss_workspace, start_position)

        R = mat_from_a1a2(a1, a2)

        out = logSE3(R, t)
        out = torch_normalizer.apply(out, normalizer, 1.1, 0.001)
        A_np = np.zeros((data.size(0) * seq_len, eq_dim, 6)).astype(np.float64)
        b_np = np.zeros((data.size(0), seq_len, 1)).astype(np.float64)
        A_np = torch.from_numpy(A_np)
        A_np = A_np.reshape(-1, 1, 6).requires_grad_(True)
        b_np = torch.from_numpy(b_np)
        b_np = b_np.reshape(-1, 1).requires_grad_(True)
        out = out.cpu()


if __name__ == "__main__":
    print("loading data")

    with open("train_qp_coll.pkl", "rb") as f:
        train_data = pickle.load(f)

    with open("test_qp_coll.pkl", "rb") as f:
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
    model.llm.to(torch.bfloat16)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        # weight_decay=1e-5,
        lr=1e-4,
    )

    q_reg = 1e-3
    speed = -2
    normalizer = tartempion.Normalizer()

    Inductive_bias_workspace = tartempion.SE3_Inductive_Bias()
    workspace = tartempion.QPworkspace()
    workspace.set_echo(True)
    workspace.set_q_reg(q_reg)
    workspace.set_lambda(speed)
    workspace.set_collisions_safety_margin(0.01)
    workspace.set_collisions_strength(50)
    workspace.set_L1(0.00)
    workspace.set_rot_w(1.0)
    workspace.view_geometries()
    workspace.add_coll_pair(1, 2)
    workspace.add_coll_pair(1, 4)
    workspace.add_coll_pair(0, 2)
    workspace.add_coll_pair(0, 3)
    workspace.add_coll_pair(0, 4)

    eff_pos = np.array([0, 0, 0.15])
    eff_rot = np.identity(3)

    theta = np.deg2rad(90)
    Ry = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    theta = np.deg2rad(180)
    Ry2 = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    arm_pos = np.array([-0.2, 0, 0.02])
    arm_rot = Ry

    plane_pos = np.array([0, 0, -5])
    plane_rot = np.identity(3)

    collate_fn = custom_collate_fn
    src_path = Path("model/src")
    files = [str(p) for p in src_path.rglob("*")]
    rmodel, gmodel, vmodel = pin.buildModelsFromUrdf(
        "model/mantis.urdf", package_dirs=files
    )
    rmodel.data = rmodel.createData()
    tool_id = 257
    init_pos = pin.neutral(rmodel)
    init_pos[len(init_pos) - 5] = -np.pi / 2
    init_pos[10] = -np.pi / 2
    rmodel = pin.buildReducedModel(rmodel, list(range(7, len(init_pos) + 1)), init_pos)
    rmodel.data = rmodel.createData()

    workspace.set_tool_id(tool_id)
    n_threads = 50
    os.environ["OMP_PROC_BIND"] = "spread"

    save_dir = "debug_batches"
    os.makedirs(save_dir, exist_ok=True)
    print("training v2")
    num_epochs = 1000
    running_loss = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for step, batch in tqdm(enumerate(train_loader)):
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

            workspace.pre_allocate(end_motion.size(0))
            local_batch_size = end_motion.size(0)

            eff_pos_batch = np.tile(eff_pos, (local_batch_size, 1))
            eff_rot_batch = np.tile(eff_rot, (local_batch_size, 1))

            workspace.set_all_coll_pos(0, eff_pos_batch, eff_rot_batch)

            arm_pos_batch = np.tile(arm_pos, (local_batch_size, 1))
            arm_rot_batch = np.tile(arm_rot, (local_batch_size, 1))
            workspace.set_all_coll_pos(1, arm_pos_batch, arm_rot_batch)

            plane_pos_batch = np.tile(plane_pos, (local_batch_size, 1))
            plane_rot_batch = np.tile(plane_rot, (local_batch_size, 1))
            workspace.set_all_coll_pos(2, plane_pos_batch, plane_rot_batch)

            which_obj = batch["obj_feature"]
            b = torch.arange(which_obj.size(0), device=which_obj.device)
            all_caps_pos = batch["obj_data_position"].view(local_batch_size, 6, 3)
            caps_pos = all_caps_pos[b, which_obj, :].detach().cpu().numpy()
            all_caps_rot = batch["obj_data_rot"].view(local_batch_size, 6, 3, 3)
            caps_rot = (
                all_caps_rot[b, which_obj, :]
                .view(local_batch_size * 3, 3)
                .detach()
                .cpu()
                .numpy()
            )
            cylinder_radius = batch["cylinder_radius"].detach().cpu().numpy()
            cylinder_length = batch["cylinder_length"].detach().cpu().numpy()

            workspace.set_all_coll_pos(3, caps_pos, caps_rot)
            workspace.set_capsule_size(
                np.array(cylinder_radius), np.array(cylinder_length)
            )

            ball_pos = (
                batch["ball_pos"].view(local_batch_size, 3).detach().cpu().numpy()
            )
            ball_rot = (
                batch["ball_rot"].view(local_batch_size * 3, 3).detach().cpu().numpy()
            )
            ball_size = batch["ball_size"].detach().cpu().numpy()
            workspace.set_all_coll_pos(4, ball_pos, ball_rot)
            workspace.set_ball_size(ball_size)

            if DEBUG:
                idx = 1
                for key, value in batch.items():
                    elt = value[idx]
                    print(
                        f"{key}: type={type(elt)}, shape={getattr(elt, 'shape', None)}"
                    )
                    print(elt)
                    print("-" * 40)
                custom_gmodel = pin.GeometryModel()
                eff_ball = coal.Sphere(0.1)
                arm = coal.Capsule(0.05, 0.5)
                plane = coal.Box(10, 10, 10)
                capsule = coal.Capsule(0.1, 0.1)
                ball = coal.Sphere(0.1)

                eff_T = workspace.get_coll_pos(0, idx)
                print(eff_T)
                eff_pos = eff_T.translation.copy()
                eff_rot = eff_T.rotation.copy()
                geom_end_eff = pin.GeometryObject(
                    "end_eff",
                    tool_id,
                    rmodel.frames[tool_id].parentJoint,
                    eff_ball,
                    pin.SE3(eff_rot, eff_pos),
                )

                eff_T = workspace.get_coll_pos(1, idx)
                print(eff_T)
                arm_pos = eff_T.translation.copy()
                arm_rot = eff_T.rotation.copy()
                geom_arm = pin.GeometryObject(
                    "arm",
                    209,
                    rmodel.frames[209].parentJoint,
                    arm,
                    pin.SE3(arm_rot, arm_pos),
                )

                eff_T = workspace.get_coll_pos(2, idx)
                print(eff_T)
                plane_pos = eff_T.translation.copy()
                plane_rot = eff_T.rotation.copy()
                geom_plane = pin.GeometryObject(
                    "plane",
                    0,
                    0,
                    plane,
                    pin.SE3(plane_rot, plane_pos),
                )

                eff_T = workspace.get_coll_pos(3, idx)
                print(eff_T)
                caps_pos = eff_T.translation.copy()
                caps_rot = eff_T.rotation.copy()
                geom_caps = pin.GeometryObject(
                    "capsule",
                    0,
                    0,
                    capsule,
                    pin.SE3(caps_rot, caps_pos),
                )

                eff_T = workspace.get_coll_pos(4, idx)
                print(eff_T)
                ball_pos = eff_T.translation.copy()
                ball_rot = eff_T.rotation.copy()
                geom_ball = pin.GeometryObject(
                    "ball",
                    0,
                    0,
                    ball,
                    pin.SE3(ball_rot, ball_pos),
                )

                color = np.random.uniform(0, 1, 4)
                color[3] = 1
                geom_end_eff.meshColor = color
                geom_arm.meshColor = color
                geom_plane.meshColor = color
                geom_plane.meshColor = np.array([1, 1, 1, 1])
                custom_gmodel.addGeometryObject(geom_end_eff)
                custom_gmodel.addGeometryObject(geom_arm)
                custom_gmodel.addGeometryObject(geom_plane)
                custom_gmodel.addGeometryObject(geom_caps)
                custom_gmodel.addGeometryObject(geom_ball)
                vmodel.addGeometryObject(geom_end_eff)
                vmodel.addGeometryObject(geom_arm)
                vmodel.addGeometryObject(geom_plane)
                vmodel.addGeometryObject(geom_caps)
                vmodel.addGeometryObject(geom_ball)
                gdata = custom_gmodel.createData()
                gdata.enable_contact = True

                viz = viewer.Viewer(rmodel, vmodel, vmodel)
                viz.viz.viewer["ball"].set_object(g.Sphere(0.1))
                viz.viz.viewer["ball"].set_transform(geom_ball.placement.homogeneous)
                viz.viz.viewer["target"].set_object(g.Sphere(0.1))
                viz.viz.viewer["target"].set_transform(
                    batch["end_SE3"][idx].homogeneous
                )
                viz.display(q_start[idx].detach().cpu().numpy())
                print(batch["end_SE3"][idx])
                print(batch["obj_feature"])
                print(pin.exp6(pin.Motion(batch["end_motion"][idx])))

            output = model(
                embedding,
                start_motion.float(),
                q_start.float(),
                end_motion,
                batch["end_SE3"],
                all_caps_pos,
                all_caps_rot,
            )
            loss = output.mean()

            loss.backward()
            print("mean", loss.item())
            print("median", torch.median(output))

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * len(embedding)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} Train Loss: {avg_loss:.6f}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
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

                workspace.pre_allocate(end_motion.size(0))
                local_batch_size = end_motion.size(0)

                eff_pos_batch = np.tile(eff_pos, (local_batch_size, 1))
                eff_rot_batch = np.tile(eff_rot, (local_batch_size, 1))

                workspace.set_all_coll_pos(0, eff_pos_batch, eff_rot_batch)

                arm_pos_batch = np.tile(arm_pos, (local_batch_size, 1))
                arm_rot_batch = np.tile(arm_rot, (local_batch_size, 1))
                workspace.set_all_coll_pos(1, arm_pos_batch, arm_rot_batch)

                plane_pos_batch = np.tile(plane_pos, (local_batch_size, 1))
                plane_rot_batch = np.tile(plane_rot, (local_batch_size, 1))
                workspace.set_all_coll_pos(2, plane_pos_batch, plane_rot_batch)

                which_obj = batch["obj_feature"]
                b = torch.arange(which_obj.size(0), device=which_obj.device)
                all_caps_pos = batch["obj_data_position"].view(local_batch_size, 6, 3)
                caps_pos = all_caps_pos[b, which_obj, :].detach().cpu().numpy()
                all_caps_rot = batch["obj_data_rot"].view(local_batch_size, 6, 3, 3)
                caps_rot = (
                    all_caps_rot[b, which_obj, :]
                    .view(local_batch_size * 3, 3)
                    .detach()
                    .cpu()
                    .numpy()
                )
                cylinder_radius = batch["cylinder_radius"].detach().cpu().numpy()
                cylinder_length = batch["cylinder_length"].detach().cpu().numpy()

                workspace.set_all_coll_pos(3, caps_pos, caps_rot)
                workspace.set_capsule_size(
                    np.array(cylinder_radius), np.array(cylinder_length)
                )

                ball_pos = (
                    batch["ball_pos"].view(local_batch_size, 3).detach().cpu().numpy()
                )
                ball_rot = (
                    batch["ball_rot"]
                    .view(local_batch_size * 3, 3)
                    .detach()
                    .cpu()
                    .numpy()
                )
                ball_size = batch["ball_size"].detach().cpu().numpy()
                workspace.set_all_coll_pos(4, ball_pos, ball_rot)
                workspace.set_ball_size(ball_size)

                output = model(
                    embedding,
                    start_motion.float(),
                    q_start.float(),
                    end_motion,
                    batch["end_SE3"],
                    all_caps_pos,
                    all_caps_rot,
                )
                loss = output.mean()
                print("val mean", loss.item())
                print("val median", torch.median(output))

                val_loss += loss.item() * len(embedding)

        avg_val_loss = val_loss / len(test_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} Validation Loss: {avg_val_loss:.6f}")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        }
        if epoch % 10 == 0:
            torch.save(
                checkpoint,
                f"checkpoint_epoch_{epoch}_loss_{avg_val_loss}_version_marche.pt",
            )
