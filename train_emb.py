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
import platform
import example_robot_data as erd
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model
import pinocchio as pin
import viewer
import meshcat.geometry as g
import time as tt
from autonorm import torch_normalizer

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
batch_size = 512
eq_dim = 1
n_threads = 20
os.environ["OMP_PROC_BIND"] = "spread"


A_np = np.zeros((batch_size * seq_len, eq_dim, 6)).astype(np.float64)
b_np = np.zeros((batch_size, seq_len, 1)).astype(np.float64)
A_np = torch.from_numpy(A_np)
A_np = A_np.reshape(-1, 1, 6).requires_grad_(True)
b_np = torch.from_numpy(b_np)
b_np = b_np.reshape(-1, 1).requires_grad_(True)
motion = torch.tensor(
    [0.0425, -0.0876, -0.2162, -0.1629, 0.2322, 0.1484],
    dtype=torch.float64,
    device="cpu",
)


class MLP(nn.Module):
    def __init__(self, embedding_dim=1536, motion_dim=6, q_dim=6, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, motion_dim),
        )
        self.motion_proj = nn.Linear(motion_dim, embedding_dim)

    def forward(self, embedding, start_motion, q_start, target_placement):
        x = torch.cat([embedding, self.motion_proj(start_motion)], dim=1)
        out = self.net(x) * 1
        out = torch_normalizer.apply(out, normalizer, 0.7, 0.2)

        A_np = np.zeros((x.size(0) * seq_len, eq_dim, 6)).astype(np.float64)
        b_np = np.zeros((x.size(0), seq_len, 1)).astype(np.float64)
        A_np = torch.from_numpy(A_np)
        A_np = A_np.reshape(-1, 1, 6).requires_grad_(True)
        b_np = torch.from_numpy(b_np)
        b_np = b_np.reshape(-1, 1).requires_grad_(True)
        # print("starting")
        custom_motion = motion.repeat(batch_size, 1)
        # workspace.allocate(rmodel, 0, 0, 0, 0, 0)
        # print("p", custom_motion)
        torch.save(
            (
                q_start.detach().cpu().numpy(),
                out.unsqueeze(1).repeat(1, seq_len, 1),
                A_np * 0,
                b_np * 0,
                rmodel,
                x.size(0),
                seq_len,
                eq_dim,
                target_placement,
                dt,
                n_threads,
            ),
            "args.pt",
        )
        temp = QPkkt.apply(
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
            n_threads,
        )
        print("done")

        return temp, out[0].detach().cpu().numpy()


with open("train_dataset_safe.pkl", "rb") as f:
    train_data = pickle.load(f)

with open("test_dataset_safe.pkl", "rb") as f:
    test_data = pickle.load(f)

train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

model = MLP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 100

plot = True

if plot:
    viz = viewer.Viewer(rmodel, gmodel, vmodel, True)
    viz.viz.viewer["start"].set_object(  # type: ignore
        g.Sphere(0.01),
        g.MeshLambertMaterial(
            color=0x0000FF, transparent=True, opacity=0.5
        ),  # vert transparent
    )
    viz.viz.viewer["start2"].set_object(  # type: ignore
        g.Sphere(0.01),
        g.MeshLambertMaterial(
            color=0x00FFFF, transparent=True, opacity=0.5
        ),  # vert transparent
    )
    viz2 = viewer.Viewer(rmodel, gmodel, vmodel, True)
    viz2.viz.viewer["start"].set_object(  # type: ignore
        g.Sphere(0.01),
        g.MeshLambertMaterial(
            color=0x0000FF, transparent=True, opacity=0.5
        ),  # vert transparent
    )
    viz2.viz.viewer["start2"].set_object(  # type: ignore
        g.Sphere(0.01),
        g.MeshLambertMaterial(
            color=0x00FFFF, transparent=True, opacity=0.5
        ),  # vert transparent
    )


b = None
num_batches = len(train_loader)
print("Nombre de batchs :", num_batches)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for iter, batch in tqdm(enumerate(train_loader)):
        # if iter >= 5:
        #     break
        if b is None:
            b = batch
        batch = b
        embedding = batch["embedding"]
        s = batch["sentence"]
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
        # for i in range(len(q_start)):
        #     q_start[i] = torch.tensor(
        #         [
        #             -0.01504673,
        #             1.0844696,
        #             -1.5193261,
        #             -1.5663941,
        #             -1.0836383,
        #             0.51387227,
        #         ],
        #         device="cpu",
        #         dtype=torch.float64,
        #     )

        # for i in range(len(end_motion)):
        #     end_motion[i] = torch.tensor(
        #         [0.271686, -0.198765, 0.68474, -2.00342, -2.08917, 0.0931319],
        #         dtype=torch.float64,
        #         device="cpu",
        #     )

        # for i in range(len(start_motion)):
        #     start_motion[i] = torch.tensor(
        #         [0.757385, 0.18007, 0.447527, -0.845822, -0.0029416, 0.55592],
        #         dtype=torch.float64,
        #         device="cpu",
        #     )
        embedding = embedding.to(device)
        start_motion = start_motion.to(device)
        q_start = q_start.to(device)
        end_motion = end_motion.to(device)
        end_placement = batch["end_SE3"]
        for i in range(len(end_placement)):
            # end_placement[i] = pin.exp6(end_motion[i].detach().cpu().numpy())
            if np.linalg.norm(end_placement[i].translation) > 0.7:
                print(end_placement[i])
                print(np.linalg.norm(end_placement[i].translation))
                raise
                end_placement[i].translation *= 0.6 / np.linalg.norm(
                    end_placement[i].translation
                )
            if np.linalg.norm(end_placement[i].translation) < 0.2:
                print(end_placement[i])
                print(np.linalg.norm(end_placement[i].translation))
                raise
                end_placement[i].translation *= 0.3 / np.linalg.norm(
                    end_placement[i].translation
                )

        optimizer.zero_grad()
        output, pred0 = model(
            embedding.squeeze(1).float(),
            start_motion.float(),
            q_start.float(),
            end_placement,
        )
        loss = output.mean()
        print("doing bckward")
        loss.backward()
        print(output)
        print(loss.item())
        print(np.linalg.norm(end_placement[0].translation))
        print(np.linalg.norm(end_placement[1].translation))
        exp = pin.exp6(pin.Motion(pred0))
        if plot:
            arr = np.array(workspace.get_q())
            for plot_time in range(arr.shape[1] - 1, arr.shape[1]):
                viz.display(arr[0, plot_time])
                # tt.sleep(dt)
            arr = arr[0]
            viz.viz.viewer["start"].set_transform(exp.homogeneous)
            viz.viz.viewer["start2"].set_transform(end_placement[0].homogeneous)
            viz.display(arr[-1])
            # print(output[0].item())
            # for plot_time in range(arr.shape[1] - 1, arr.shape[1]):
            #     viz.display(arr[plot_time])
            # tt.sleep(dt)
            arr = np.array(workspace.get_q())
            arr = arr[1]
            pin.framesForwardKinematics(rmodel, rmodel.data, arr[-1])
            viz2.viz.viewer["start"].set_transform(rmodel.data.oMf[tool_id].homogeneous)
            viz2.viz.viewer["start2"].set_transform(end_placement[1].homogeneous)
            # print(output[0].item())
            viz2.display(arr[-1])
            for _ in range(0):
                break
                for ttime in range(arr.shape[0]):
                    viz.display(arr[ttime])
                print("type to go next iter")
        # input()
        optimizer.step()

        total_loss += loss.item() * embedding.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {avg_loss:.6f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for iter, batch in enumerate(test_loader):
            if iter > 1:
                break
            embedding = batch["embedding"]
            s = batch["sentence"]
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
            embedding = embedding.to(device)
            start_motion = start_motion.to(device)
            q_start = q_start.to(device)
            end_motion = end_motion.to(device)
            end_placement = batch["end_SE3"]
            output, pred0 = model(
                embedding.squeeze(1).float(),
                start_motion.float(),
                q_start.float(),
                end_placement,
            )
            loss = output.mean()
            val_loss += loss.item() * embedding.size(0)

    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_loss = val_loss / batch_size

    print(f"Epoch {epoch+1}/{num_epochs} Validation Loss: {avg_val_loss:.6f}")
