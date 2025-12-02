import numpy as np
import torch
import pickle
from dataset_to_torch import TrajectoryDataset as MyDataset
import sys
import os
import platform
import example_robot_data as erd
from tqdm import tqdm
import time
import viewer
import matplotlib.pyplot as plt
import coal
import diffcoal
import pinocchio as pin
from scipy.spatial.transform import Rotation
from pathlib import Path


system = platform.system()
if not system == "Linux":
    import meshcat.geometry as g
paths = []
if system == "Linux":
    paths.append(
        "/lustre/fswork/projects/rech/tln/urh44lu/pinocchio-minimal-main/build/python"
    )
elif system == "Darwin":  # macOS
    paths.append("build/python")
else:
    raise RuntimeError(f"Système non supporté : {system}")
for p in paths:
    if os.path.exists(p):
        if p not in sys.path:
            sys.path.insert(0, p)
import tartempion


src_path = Path("model/src")
files = [str(p) for p in src_path.rglob("*")]
batch_size = 1
q_reg = 1e-3
bound = -1000
workspace = tartempion.QPworkspace()
workspace.set_q_reg(q_reg)
workspace.set_bound(bound)
workspace.set_lambda(-1)
workspace.set_collisions_safety_margin(0.01)
workspace.set_collisions_strength(50)
workspace.view_geometries()
workspace.add_coll_pair(1, 4)
workspace.add_coll_pair(0, 2)
workspace.add_coll_pair(0, 3)
workspace.add_coll_pair(0, 4)
workspace.set_L1(0.00)
workspace.set_rot_w(1.0)
robot = erd.load("ur5")
# rmodel, gmodel, vmodel = robot.model, robot.collision_model, robot.visual_model
rmodel, gmodel, vmodel = pin.buildModelsFromUrdf(
    "model/mantis.urdf", package_dirs=files
)

# tool_id = 21
tool_id = 257
init_pos = pin.neutral(rmodel)
init_pos[len(init_pos) - 5] = -np.pi / 2
init_pos[10] = -np.pi / 2
rmodel = pin.buildReducedModel(rmodel, list(range(7, len(init_pos) + 1)), init_pos)
rmodel.data = rmodel.createData()


workspace.set_tool_id(tool_id)
seq_len = 400
dt = 0.01
eq_dim = 1
n_threads = 50
os.environ["OMP_PROC_BIND"] = "spread"

A_np = np.zeros((batch_size * seq_len, eq_dim, 6)).astype(np.float64)
b_np = np.zeros((batch_size, seq_len, 1)).astype(np.float64)
p_np: np.ndarray


custom_gmodel = pin.GeometryModel()
eff_ball = coal.Sphere(0.1)
arm = coal.Capsule(0.05, 0.5)
plane = coal.Box(10, 10, 10)
cylinder_radius = 0.03
cylinder_length = 0.2
cylinder = coal.Capsule(cylinder_radius, cylinder_length)
ball_size = 0.14 / 2
ball = coal.Sphere(ball_size)

grasp_height = 0.02

eff_pos = np.zeros(3)
eff_rot = np.identity(3)
geom_end_eff = pin.GeometryObject(
    "end_eff",
    tool_id,
    rmodel.frames[tool_id].parentJoint,
    eff_ball,
    pin.SE3(eff_rot, eff_pos),
)
workspace.set_coll_pos(0, eff_pos, eff_rot)


theta = np.deg2rad(90)

# Rotation de 90° autour de X
Rx = np.array(
    [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
)

# Rotation de 90° autour de Y
Ry = np.array(
    [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
)

# Rotation de 90° autour de Z
Rz = np.array(
    [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
)
arm_pos = np.array([-0.2, 0, 0.02])
arm_rot = Ry
geom_arm = pin.GeometryObject(
    "arm",
    209,
    rmodel.frames[209].parentJoint,
    arm,
    pin.SE3(arm_rot, arm_pos),
)
workspace.set_coll_pos(1, arm_pos, arm_rot)

plane_pos = np.array([0, 0, -5])
plane_rot = np.identity(3)
geom_plane = pin.GeometryObject(
    "plane",
    0,
    0,
    plane,
    pin.SE3(plane_rot, plane_pos),
)
workspace.set_coll_pos(2, plane_pos, plane_rot)

cylinder_pos = np.array([0.5, 0.5, 0.25])
cylinder_rot = np.identity(3)
geom_cylinder = pin.GeometryObject(
    "cylinder",
    0,
    0,
    cylinder,
    pin.SE3(cylinder_rot, cylinder_pos),
)
workspace.set_coll_pos(3, cylinder_pos, cylinder_rot)
workspace.set_capsule_size(cylinder_radius, cylinder_length)

ball_pos = np.array([0.25, 0.25, ball_size])
ball_rot = np.identity(3)
geom_ball = pin.GeometryObject(
    "ball",
    0,
    0,
    ball,
    pin.SE3(ball_rot, ball_pos),
)
workspace.set_coll_pos(4, ball_pos, ball_rot)
workspace.set_ball_size(ball_size)


color = np.random.uniform(0, 1, 4)
color[3] = 1
geom_end_eff.meshColor = color
geom_arm.meshColor = color
geom_plane.meshColor = color
geom_cylinder.meshColor = color
geom_ball.meshColor = color
geom_plane.meshColor = np.array([1, 1, 1, 1])
custom_gmodel.addGeometryObject(geom_end_eff)
custom_gmodel.addGeometryObject(geom_arm)
custom_gmodel.addGeometryObject(geom_plane)
custom_gmodel.addGeometryObject(geom_cylinder)
custom_gmodel.addGeometryObject(geom_ball)
vmodel.addGeometryObject(geom_end_eff)
vmodel.addGeometryObject(geom_arm)
vmodel.addGeometryObject(geom_plane)
vmodel.addGeometryObject(geom_cylinder)
vmodel.addGeometryObject(geom_ball)
gdata = custom_gmodel.createData()
gdata.enable_contact = True


viz = viewer.Viewer(rmodel, gmodel, vmodel, True)
viz.viz.viewer["ideal"].set_object(
    g.Sphere(0.05),
    g.MeshLambertMaterial(color=0x00FFFF, transparent=True, opacity=0.5),
)
viz.viz.viewer["current"].set_object(
    g.Sphere(0.05),
    g.MeshLambertMaterial(color=0xFFFF00, transparent=True, opacity=0.5),
)
print(pin.neutral(rmodel))
viz.display(pin.neutral(rmodel))


est = 0
rest = 0


def finite_difference_forward_pass(func, q_initial, epsilon=1e-5):
    m, n = q_initial.shape
    f0 = func(q_initial[np.newaxis, :, :])
    f0 = np.asarray(f0)

    grad_shape = f0.shape + (m, n)
    grad = np.zeros(grad_shape)

    for i in tqdm(range(m)):
        if i % 1 == 0 or i >= m - 10:
            for j in range(n):
                q_plus = q_initial.copy()
                q_minus = q_initial.copy()
                q_plus[i, j] += epsilon
                q_minus[i, j] -= epsilon

                f_plus = np.asarray(func(q_plus[np.newaxis, :, :]))
                f_minus = np.asarray(func(q_minus[np.newaxis, :, :]))

                grad[:, i, j] = (f_plus - f_minus) / (2 * epsilon)

    return grad


def forward_kine(p):
    return tartempion.forward_pass(
        workspace,
        np.tile(p[:, :, :], (1, 1, 1)),
        A_np * 0,
        b_np * 0,
        states_init,
        rmodel,
        n_threads,
        targets,
        dt,
    )


def sample_p_start():
    while True:
        q = pin.randomConfiguration(rmodel)
        pin.framesForwardKinematics(rmodel, rmodel.data, q)
        T = rmodel.data.oMf[tool_id]
        if T.translation[2] > 0.2:
            return q


path = "/Users/mathisscheffler/Desktop/marche pas/debug_dump_1764250224.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

np.random.seed(1)
for l in tqdm(range(1000)):
    end_SE3 = pin.SE3.Random()
    q_start = sample_p_start()
    states_init = np.array([q_start])
    R = Rotation.random().as_matrix()
    v = np.random.randn(3) * 0.313
    Pexp = pin.SE3(R, v)
    p_np = np.array(pin.log6(Pexp).vector)
    p_np = data["p_np"][255, 0]
    p_np = np.array(
        pin.log6(pin.SE3(np.identity(3), np.array([0.25, 0.25, -0.25]))).vector
    )
    states_init[0] = data["q"][255]

    p_np = np.repeat(p_np[np.newaxis, :], repeats=batch_size, axis=0)
    p_np = np.repeat(p_np[:, np.newaxis, :], repeats=seq_len, axis=1)

    viz.display(states_init[0])
    targets = [end_SE3]
    targets = [data["target"][255]]

    print(targets)
    print(p_np.shape)
    print(states_init.shape)

    viz.viz.viewer["current"].set_transform(
        pin.exp6(pin.Motion(p_np[0, 0])).homogeneous
    )
    viz.viz.viewer["ideal"].set_transform(targets[0].homogeneous)

    articular_speed: np.ndarray = tartempion.forward_pass(
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

    arr = np.array(workspace.get_q())
    np.set_printoptions(precision=100)
    print("q", arr[0, -2])

    tartempion.backward_pass(
        workspace,
        rmodel,
        torch.tensor([[[0]]]).cpu().numpy(),
        1,
        1,
    )

    for i in tqdm(range(len(arr[0]))):
        if i % 1 == 0:
            viz.display(arr[0, i])
            # input()
            if arr[0, i, 0] == 0:
                break
                pass
            time.sleep(dt)
    p_grad = np.array(workspace.grad_p())
    grad = p_grad.sum(0)
    print("ana", grad)
    fd_grad = finite_difference_forward_pass(forward_kine, p_np[0, :, :], 1e-5)
    print("fd", fd_grad.sum(0).sum(0))
    print("err max", np.max(fd_grad - p_grad))
    plt.plot(p_grad[:, 0], color="blue")
    plt.plot(fd_grad[0, :, 0], color="red")
    plt.legend()
    plt.show()
