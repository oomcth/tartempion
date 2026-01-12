import pinocchio as pin
import time
import viewer
from matplotlib.path import Path
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import tartempion
from tqdm import tqdm
import torch
import coal
import example_robot_data
from candlewick import Visualizer, VisualizerConfig, create_recorder_context

robot: pin.Model = example_robot_data.load("go2")
rmodel, gmodel, vmodel = robot.model, robot.collision_model, robot.visual_model

for i, frame in enumerate(rmodel.frames):
    print(frame.name, i)

FL_id = 10
FR_id = 24
RL_id = 38
RR_id = 52

data = rmodel.createData()


def plot_support_and_com(poly, com, inside=True):
    fig, ax = plt.subplots()
    polygon = plt.Polygon(
        poly, closed=True, color="lightgray", alpha=0.5, edgecolor="k"
    )
    ax.add_patch(polygon)
    ax.plot(
        com[0],
        com[1],
        "o",
        color="green" if inside else "red",
        markersize=10,
        label="Projection CoM",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Polygone de support et projection du CoM")
    ax.legend()
    plt.show()


def is_in_balance(rmodel, data, q, support_margin=0.0):
    pin.forwardKinematics(rmodel, data, q)
    pin.updateFramePlacements(rmodel, data)
    pin.centerOfMass(rmodel, data, q)
    com = data.com[0][:2]  # (x, y)

    # --- 2. Positions des trois appuis ---
    FR = data.oMf[FR_id].translation[:2]
    RL = data.oMf[RL_id].translation[:2]
    RR = data.oMf[RR_id].translation[:2]
    print(FR, RL, RR)
    input()
    pts = np.vstack([FR, RL, RR])

    # --- 3. Marge optionnelle sur le triangle ---
    if support_margin > 0:
        center = np.mean(pts, axis=0)
        direction = pts - center
        pts = center + (1.0 + support_margin) * direction

    # --- 4. Polygone de support (le triangle des trois appuis) ---
    hull = ConvexHull(pts)
    poly = pts[hull.vertices]  # sommets triés du triangle

    # --- 5. Test d’appartenance du CoM ---
    path = Path(poly)
    inside = path.contains_point(com)

    return inside, com, poly


print(rmodel)


data = rmodel.createData()
q0 = pin.neutral(rmodel)
q0[2] = 0.3258228

root_id = 1
rmodel, [gmodel, vmodel] = pin.buildReducedModel(
    rmodel, [gmodel, vmodel], [root_id], q0
)
q0 = pin.neutral(rmodel)
q0[1] = 0.7
q0[2] = -1.4
q0[4] = 0.7
q0[5] = -1.4
q0[7] = 0.7
q0[8] = -1.4
q0[10] = 0.7
q0[11] = -1.4
data = rmodel.createData()


def add_ball(pos: pin.SE3):
    geom = pin.GeometryObject(
        "sphere",
        0,
        0,
        coal.Sphere(0.01),
        pos,
    )
    geom.meshColor = np.ones(4)
    vmodel.addGeometryObject(geom)


for i, frames in enumerate(rmodel.frames):
    print(frames.name)


pin.framesForwardKinematics(rmodel, data, q0)
target = data.oMf[FR_id].copy()
target.translation = target.translation + np.array([0.2, 0, 0.5])
add_ball(target)
viz = viewer.Viewer(rmodel, gmodel, vmodel, candlewick=False)

batch_size = 1
q_reg = 1e-2
dt = 0.005
seq_len = 300

workspace = tartempion.QPworkspace()
workspace.set_echo(True)
workspace.set_allow_collisions(False)
workspace.pre_allocate(1)
workspace.set_all_ur5_config()
workspace.set_q_reg(q_reg)
workspace.set_lambda(-2)
workspace.set_collisions_safety_margin(0.02)
workspace.set_collisions_strength(50)
workspace.view_geometries()
workspace.set_L1(0.00)
workspace.set_rot_w(1e-10)
workspace.set_tool_id(FR_id)
workspace.set_equilibrium(True)
t2 = pin.log6(target).vector
t2 = np.repeat(t2[None, :], seq_len, axis=0)
workspace.set_equilibrium_second_target(t2)
workspace.set_equilibrium_tool_id(18)

workspace.allocate(rmodel, batch_size, seq_len, rmodel.nv, 0, 1)

tartempion.check_dub_dq(workspace, rmodel, q0, data)
tartempion.check_dGb_dq(workspace, rmodel, q0, data)


p_np = pin.log6(target).vector
p_np = np.vstack(
    [
        np.repeat(p_np[None, :], seq_len, axis=0),
    ],
)[None, :]

A_np = np.zeros((batch_size * seq_len, 1, 6)).astype(np.float64)
b_np = np.zeros((batch_size, seq_len, 1)).astype(np.float64)
states_init = q0[None, :]
targets = [target]

epoch = 1_000
bar = tqdm(range(epoch), leave=True)
for i in bar:
    loss: np.ndarray = tartempion.forward_pass(
        workspace,
        p_np,
        A_np,
        b_np,
        states_init,
        rmodel,
        1,
        targets,
        dt,
    )
    bar.set_description(f"loss={loss.sum():.6f}")
    arr = np.array(workspace.get_q())[0]
    viz.display(arr[-1])
    tartempion.backward_pass(
        workspace,
        rmodel,
        torch.tensor([[[0]]]).cpu().numpy(),
        1,
        1,
    )

    grad_1 = np.array(workspace.grad_p()).sum(0)
    grad_2 = np.array(workspace.grad_p2()).sum(0)
    # print(grad_1)
    # print(grad_2)
    lr = 1e-1
    p_np[0, 0] -= grad_1 * lr
    p_np = np.vstack(
        [
            np.repeat(p_np[0, 0][None, :], seq_len, axis=0),
        ],
    )[None, :]
    t2 -= grad_2 * lr
    workspace.set_equilibrium_second_target(t2)

# plt.plot(np.abs(grad_1[:, 0]))
# plt.plot(np.abs(grad_2[:, 0]))
# plt.yscale("log")
# plt.show()

print("done")

arr = np.array(workspace.get_q())[0]
in_balance, com_xy, polygon = is_in_balance(rmodel, data, q0)
print(in_balance)
plot_support_and_com(polygon, com_xy, in_balance)
for i in tqdm(range(len(arr))):
    viz.display(arr[i])
    time.sleep(dt)
for i in tqdm(range(len(arr))):
    viz.display(arr[i])
    time.sleep(dt)
for i in tqdm(range(len(arr))):
    viz.display(arr[i])
    time.sleep(dt)
for i in tqdm(range(len(arr))):
    viz.display(arr[i])
    time.sleep(dt)
for i in tqdm(range(len(arr))):
    viz.display(arr[i])
    time.sleep(dt)
in_balance, com_xy, polygon = is_in_balance(rmodel, data, arr[-1])
print(in_balance)
plot_support_and_com(polygon, com_xy, in_balance)
while True:
    viz.display(arr[0])
