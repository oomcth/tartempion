import numpy as np
import torch
import pickle
from dataset_to_torch import TrajectoryDataset as MyDataset
import os
import example_robot_data as erd
from tqdm import tqdm
import viewer
import coal
import pinocchio as pin
from scipy.spatial.transform import Rotation
from pathlib import Path
import proxsuite
from typing import Optional, Union, Tuple
import meshcat.geometry as g
import tartempion
import random
import time
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Exemple : option --plot")

parser.add_argument(
    "--plot",
    nargs="?",
    const=True,
    type=int,
)
args = parser.parse_args()
plot_enabled = False
plot_n = 0
if args.plot is not None:
    plot_enabled = True
    if args.plot is not True:
        plot_n = args.plot

print(f"plot_enabled={plot_enabled}, plot_n={plot_n}")

np.random.seed(1)
pin.seed(1)


src_path = Path("model/src")
files = [str(p) for p in src_path.rglob("*")]
batch_size = 1
q_reg = 1e-4
seq_len = 100
dt = 0.01
bound = -1000
workspace = tartempion.QPworkspace()
workspace.set_echo(True)
workspace.set_allow_collisions(False)
workspace.pre_allocate(batch_size)
workspace.set_q_reg(q_reg)
workspace.set_bound(bound)
workspace.set_lambda(-2)
workspace.set_collisions_safety_margin(0.02)
workspace.set_collisions_strength(50)
workspace.view_geometries()
workspace.set_L1(0.00)
workspace.set_rot_w(1e-10)

workspace.add_coll_pair(0, 5)
# workspace.add_coll_pair(0, 8)
# workspace.add_coll_pair(0, 9)
# workspace.add_coll_pair(0, 10)
# workspace.add_coll_pair(0, 11)

# workspace.add_coll_pair(1, 5)
# workspace.add_coll_pair(1, 8)
# workspace.add_coll_pair(1, 9)
# workspace.add_coll_pair(1, 10)
# workspace.add_coll_pair(1, 11)

# workspace.add_coll_pair(2, 5)
# workspace.add_coll_pair(2, 8)
# workspace.add_coll_pair(2, 9)
# workspace.add_coll_pair(2, 10)
# workspace.add_coll_pair(2, 11)

# workspace.add_coll_pair(3, 5)
# workspace.add_coll_pair(3, 8)
# workspace.add_coll_pair(3, 9)
# workspace.add_coll_pair(3, 10)
# workspace.add_coll_pair(3, 11)

# workspace.add_coll_pair(4, 5)
# workspace.add_coll_pair(4, 8)
# workspace.add_coll_pair(4, 9)
# workspace.add_coll_pair(4, 10)
# workspace.add_coll_pair(4, 11)


robot = erd.load("ur5")
rmodel, gmodel, vmodel = pin.buildModelsFromUrdf(
    "model/mantis.urdf", package_dirs=files
)

tool_id = 257
init_pos = pin.neutral(rmodel)
init_pos[len(init_pos) - 5] = -np.pi / 2
init_pos[10] = -np.pi / 2
rmodel = pin.buildReducedModel(rmodel, list(range(7, len(init_pos) + 1)), init_pos)
rmodel.data = rmodel.createData()


workspace.set_tool_id(tool_id)
eq_dim = 1
n_threads = 50
os.environ["OMP_PROC_BIND"] = "spread"

A_np = np.zeros((batch_size * seq_len, eq_dim, 6)).astype(np.float64)
b_np = np.zeros((batch_size, seq_len, 1)).astype(np.float64)
p_np: np.ndarray


custom_gmodel = pin.GeometryModel()
eff_ball = coal.Sphere(0.1)
arm = coal.Cylinder(0.05, 0.5)
arm1 = coal.Sphere(0.08)
arm2 = coal.Sphere(0.10)
arm3 = coal.Sphere(0.08)
plane = coal.Box(10, 10, 10)
cylinder_radius = 0.3
cylinder_length = 10
cylinder = coal.Capsule(cylinder_radius, cylinder_length)
ball_radius = 0.1
ball = coal.Sphere(ball_radius)
b1 = (0.35, 0.55, 0.04)
box1 = coal.Box(*b1)
b2 = (0.35, 0.35, 0.04)
box2 = coal.Box(*b2)
b3 = (0.35, 0.35, 0.04)
box3 = coal.Box(*b3)
b4 = (0.35, 0.6, 0.04)
box4 = coal.Box(*b4)

grasp_height = 0.02

eff_pos = np.array([0, 0, 0.2])
eff_rot = np.identity(3)
geom_end_eff = pin.GeometryObject(
    "end_eff",
    tool_id,
    rmodel.frames[tool_id].parentJoint,
    eff_ball,
    pin.SE3(eff_rot, eff_pos),
)
workspace.set_coll_pos(0, 0, eff_pos, eff_rot)


def rotation_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


theta = np.deg2rad(90)
Ry = np.array(
    [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
)
theta = np.deg2rad(180)
Ry2 = np.array(
    [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
)
arm_pos = np.array([-0.2, 0, 0.02]) * 0
arm_rot = np.identity(3)
geom_arm = pin.GeometryObject(
    "arm",
    209,
    rmodel.frames[209].parentJoint,
    arm,
    pin.SE3(arm_rot, arm_pos),
)
workspace.set_coll_pos(1, 0, arm_pos, arm_rot)

arm_pos1 = np.array([-0.4, 0, 0.02])
arm_rot1 = np.identity(3)
geom_arm1 = pin.GeometryObject(
    "arm1",
    209,
    rmodel.frames[209].parentJoint,
    arm1,
    pin.SE3(arm_rot1, arm_pos1),
)
workspace.set_coll_pos(2, 0, arm_pos1, arm_rot1)

arm_pos2 = np.array([-0.2, 0, 0.02])
arm_rot2 = np.identity(3)
geom_arm2 = pin.GeometryObject(
    "arm2",
    209,
    rmodel.frames[209].parentJoint,
    arm2,
    pin.SE3(arm_rot2, arm_pos2),
)
workspace.set_coll_pos(3, 0, arm_pos2, arm_rot2)

arm_pos3 = np.array([-0.0, 0, 0.02])
arm_rot3 = np.identity(3)
geom_arm3 = pin.GeometryObject(
    "arm3",
    209,
    rmodel.frames[209].parentJoint,
    arm3,
    pin.SE3(arm_rot3, arm_pos3),
)
workspace.set_coll_pos(4, 0, arm_pos3, arm_rot3)

plane_pos = np.array([0, 0, -5])
plane_rot = np.identity(3)
geom_plane = pin.GeometryObject(
    "plane",
    0,
    0,
    plane,
    pin.SE3(plane_rot, plane_pos),
)
workspace.set_coll_pos(5, 0, plane_pos, plane_rot)


caps_pos = np.array([-0.5, 0.1, 4.4])
caps_rot = Ry
geom_caps = pin.GeometryObject(
    "caps",
    0,
    0,
    cylinder,
    pin.SE3(caps_rot, caps_pos),
)
workspace.set_coll_pos(6, 0, caps_pos, caps_rot)
workspace.set_capsule_size(np.array([cylinder_radius]), np.array([cylinder_length]))


ball_pos = np.array([0.1, 0.1, 10.3])
ball_rot = np.identity(3)
geom_ball = pin.GeometryObject(
    "ball",
    0,
    0,
    ball,
    pin.SE3(ball_rot, ball_pos),
)
workspace.set_coll_pos(7, 0, ball_pos, ball_rot)
workspace.set_ball_size(np.array([ball_radius]))


box_pos1 = np.array([0.3, 0.5, 0.35])
box_rot1 = np.identity(3)
geom_box1 = pin.GeometryObject(
    "box1",
    0,
    0,
    box1,
    pin.SE3(box_rot1, box_pos1),
)
workspace.set_coll_pos(8, 0, box_pos1, box_rot1)
workspace.set_box_size(np.array([b1[0]]), np.array([b1[1]]), np.array([b1[2]]), 1)

box_pos2 = np.array([0.3, 0.5 - b1[1] / 2, 0.35 / 2])
box_rot2 = rotation_x(np.deg2rad(90))
geom_box2 = pin.GeometryObject(
    "box2",
    0,
    0,
    box2,
    pin.SE3(box_rot2, box_pos2),
)
workspace.set_coll_pos(9, 0, box_pos2, box_rot2)
workspace.set_box_size(np.array([b2[0]]), np.array([b2[1]]), np.array([b2[2]]), 2)

box_pos3 = np.array([0.3, 0.5 + b1[1] / 2, 0.35 / 2])
box_rot3 = rotation_x(np.deg2rad(90))
geom_box3 = pin.GeometryObject(
    "box3",
    0,
    0,
    box3,
    pin.SE3(box_rot3, box_pos3),
)
workspace.set_coll_pos(10, 0, box_pos3, box_rot3)
workspace.set_box_size(np.array([b3[0]]), np.array([b3[1]]), np.array([b3[2]]), 3)

box_pos4 = np.array([0.3 + b1[0] / 2, box_pos1[1], box_pos1[2] / 2])
box_rot4 = Ry
geom_box4 = pin.GeometryObject(
    "box4",
    0,
    0,
    box4,
    pin.SE3(box_rot4, box_pos4),
)
workspace.set_coll_pos(11, 0, box_pos4, box_rot4)
workspace.set_box_size(np.array([b4[0]]), np.array([b4[1]]), np.array([b4[2]]), 4)

color = np.random.uniform(0, 1, 4)
color[3] = 1
geom_end_eff.meshColor = color
geom_arm.meshColor = color
geom_arm1.meshColor = color
geom_arm2.meshColor = color
geom_plane.meshColor = color
geom_plane.meshColor = np.array([1, 1, 1, 1])
custom_gmodel.addGeometryObject(geom_end_eff)
custom_gmodel.addGeometryObject(geom_arm)
custom_gmodel.addGeometryObject(geom_arm1)
custom_gmodel.addGeometryObject(geom_arm2)
custom_gmodel.addGeometryObject(geom_arm3)
custom_gmodel.addGeometryObject(geom_plane)
custom_gmodel.addGeometryObject(geom_caps)
custom_gmodel.addGeometryObject(geom_ball)
custom_gmodel.addGeometryObject(geom_box1)
custom_gmodel.addGeometryObject(geom_box2)
custom_gmodel.addGeometryObject(geom_box3)
custom_gmodel.addGeometryObject(geom_box4)
vmodel.addGeometryObject(geom_end_eff)
vmodel.addGeometryObject(geom_arm)
vmodel.addGeometryObject(geom_arm1)
vmodel.addGeometryObject(geom_arm2)
vmodel.addGeometryObject(geom_arm3)
vmodel.addGeometryObject(geom_plane)
vmodel.addGeometryObject(geom_caps)
vmodel.addGeometryObject(geom_ball)
vmodel.addGeometryObject(geom_box1)
vmodel.addGeometryObject(geom_box2)
vmodel.addGeometryObject(geom_box3)
vmodel.addGeometryObject(geom_box4)
gdata = custom_gmodel.createData()
gdata.enable_contact = True

workspace.allocate(rmodel, batch_size, seq_len, rmodel.nv, eq_dim, n_threads)
workspace.init_geometry(rmodel, batch_size)
gmodel2 = workspace.get_gmodel(0).copy()
obj_id = gmodel2.getGeometryId("plane")
geom_obj = gmodel2.geometryObjects[obj_id]
geom_obj.geometry = coal.Box(10.0, 10.0, geom_obj.geometry.halfSide[2] * 2)


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


viz = viewer.Viewer(rmodel, gmodel2, gmodel2, True)
viz.viz.viewer["ideal"].set_object(
    g.Sphere(0.05),
    g.MeshLambertMaterial(color=0x00FFFF, transparent=True, opacity=0.5),
)
viz.viz.viewer["current"].set_object(
    g.Sphere(0.05),
    g.MeshLambertMaterial(color=0xFFFF00, transparent=True, opacity=0.5),
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

    # for i in tqdm(range(len(arr[0]))):
    #     if i % 1 == 0:
    #         viz.display(arr[0, i])
    #         # input()
    #         if arr[0, i, 0] == 0:
    #             break
    #             pass
    #         # time.sleep(dt)
    p_grad = np.array(workspace.grad_p())
    grad = p_grad.sum(0)
    # plt.plot(p_grad[:, 0], color="blue")
    # plt.show()
    print("ana", grad)
    fd_grad = finite_difference_forward_pass(forward_kine, p_np[0, :, :], 1e-5)
    print("fd", fd_grad.sum(0).sum(0))
    print("err max", np.max(fd_grad - p_grad))
    fd_grad = np.squeeze(fd_grad)
    print("fd_grad squeezed shape:", fd_grad.shape)

    # Maintenant les deux ont shape (T, D)
    err = np.abs(p_grad - fd_grad)

    err_inf = np.max(err, axis=1)
    denom = np.maximum(np.max(np.abs(fd_grad), axis=1), 1e-14)
    rel_err_inf = err_inf / denom

    plt.figure(figsize=(8, 4))
    plt.plot(err_inf, label=r"$||p_{grad} - fd_{grad}||_\infty$", color="blue")
    plt.plot(rel_err_inf, label=r"Relative $\;||\cdot||_\infty$ error", color="orange")
    plt.yscale("log")
    plt.xlabel("time step")
    plt.ylabel("error (log scale)")
    plt.legend()
    plt.title("Gradient error evolution (∞-norm)")
    plt.tight_layout()
    plt.show()
