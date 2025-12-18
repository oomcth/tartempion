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
plot_n = -1
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
q_reg = 1e-2
dt = 0.005
seq_len = int(20 / dt)
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
workspace.add_coll_pair(0, 6)
workspace.add_coll_pair(0, 8)
workspace.add_coll_pair(0, 9)
workspace.add_coll_pair(0, 10)
workspace.add_coll_pair(0, 11)

workspace.add_coll_pair(1, 5)
workspace.add_coll_pair(1, 6)
workspace.add_coll_pair(1, 8)
workspace.add_coll_pair(1, 9)
workspace.add_coll_pair(1, 10)
workspace.add_coll_pair(1, 11)

workspace.add_coll_pair(2, 9)
workspace.add_coll_pair(3, 9)
workspace.add_coll_pair(4, 9)

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
arm = coal.Ellipsoid(0.25, 0.08, 0.08)
arm1 = coal.Sphere(0.08)
arm2 = coal.Sphere(0.10)
arm3 = coal.Sphere(0.08)
plane = coal.Box(10, 10, 10)
cylinder_radius = 0.1
cylinder_length = 1
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

eff_pos = np.array([0, 0, 0.15])
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
arm_pos = np.array([-0.2, 0, 0.02])
arm_rot = Ry
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


caps_pos = np.array([-0.5, -0.65, 0.4])
caps_rot = np.identity(3)
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
# vmodel.addGeometryObject(geom_end_eff)
# vmodel.addGeometryObject(geom_arm)
# vmodel.addGeometryObject(geom_arm1)
# vmodel.addGeometryObject(geom_arm2)
# vmodel.addGeometryObject(geom_arm3)
# vmodel.addGeometryObject(geom_plane)
# vmodel.addGeometryObject(geom_caps)
# vmodel.addGeometryObject(geom_ball)
# vmodel.addGeometryObject(geom_box1)
# vmodel.addGeometryObject(geom_box2)
# vmodel.addGeometryObject(geom_box3)
# vmodel.addGeometryObject(geom_box4)
gdata = custom_gmodel.createData()
gdata.enable_contact = True

workspace.allocate(rmodel, batch_size, seq_len, rmodel.nv, eq_dim, n_threads)
workspace.init_geometry(rmodel, batch_size)
gmodel2 = workspace.get_gmodel(0).copy()
obj_id = gmodel2.getGeometryId("plane")
geom_obj = gmodel2.geometryObjects[obj_id]
geom_obj.geometry = coal.Box(10.0, 10.0, geom_obj.geometry.halfSide[2] * 2)


for geom_obj in gmodel2.geometryObjects:
    copied_obj = geom_obj.copy()
    vmodel.addGeometryObject(copied_obj)
viz = viewer.Viewer(rmodel, gmodel2, vmodel, True)
viz.viz.viewer["ideal"].set_object(
    g.Sphere(0.01),
    g.MeshLambertMaterial(color=0x00FFFF, transparent=True, opacity=0.5),
)
viz.viz.viewer["T0"].set_object(
    g.Sphere(0.01),
    g.MeshLambertMaterial(color=0xFFFF00, transparent=True, opacity=0.5),
)
viz.viz.viewer["T1"].set_object(
    g.Sphere(0.01),
    g.MeshLambertMaterial(color=0xFF0000, transparent=True, opacity=0.5),
)

q_start = np.array(
    [-1.4835299, -1.6755161, -2.2165682, -1.5707963, 0.2094395, -0.5759587]
)


pin.framesForwardKinematics(rmodel, rmodel.data, q_start)
R_target = rmodel.data.oMf[tool_id].rotation
R = Ry
v = np.array([0.45, 0.35, 0.5])

end_SE3 = pin.SE3(R_target, v)
end_log = pin.log6(end_SE3).vector
states_init = np.array([q_start])
viz.viz.viewer["ideal"].set_transform(end_SE3.homogeneous)
viz.display(q_start)


p_0 = np.random.randn(6)
p_1 = np.random.randn(6)
# p_1 = pin.log6(end_SE3).vector
p_2 = np.random.randn(6)
p_3 = np.random.randn(6)

pos = rmodel.data.oMf[tool_id].copy()
pos.translation = pos.translation + np.array([-1, 0, 1])
p_0 = pin.log6(pos).vector
pos = end_SE3.copy()
pos.translation = pos.translation + np.array([-0.3, 0, +0.1])
pos.rotation = R_target
# p_1 = pin.log6(pos).vector
print(p_0.shape)

print(q_start)
print("Press ENTER to start")
input()

t = tqdm(range(100_000))
for iter in t:
    targets = [end_SE3]
    viz.viz.viewer["T0"].set_transform(pin.exp6(pin.Motion(p_0)).homogeneous)
    viz.viz.viewer["T1"].set_transform(pin.exp6(pin.Motion(p_1)).homogeneous)
    p_np = np.vstack(
        [
            np.repeat(p_0[None, :], seq_len // 2, axis=0),
            np.repeat(p_1[None, :], seq_len // 2, axis=0),
        ],
    )[None, :]
    print("forward")
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

    if (
        loss.mean() < 1e-6
        or (plot_enabled and plot_n < iter)
        or np.array(workspace.get_discarded())[0]
    ):
        print("press enter to see traj")
        input()
        arr = np.array(workspace.get_q())
        for i in tqdm(
            range(0, len(arr[0]), 1 if np.array(workspace.get_discarded())[0] else 10)
        ):
            pin.framesForwardKinematics(rmodel, rmodel.data, arr[0, i])
            print(rmodel.data.oMf[tool_id])
            viz.viz.viewer[str(i)].set_object(
                g.Sphere(0.005),
                g.MeshLambertMaterial(color=0xFFFFFF, transparent=False, opacity=1),
            )
            viz.viz.viewer[str(i)].set_transform(rmodel.data.oMf[tool_id].homogeneous)
            viz.display(arr[0, i])
            if np.array(workspace.get_discarded())[0]:
                input()
            else:
                time.sleep(dt)

    t.set_postfix(loss=float(loss.mean()))
    print("backward")
    tartempion.backward_pass(
        workspace,
        rmodel,
        torch.tensor([[[0]]]).cpu().numpy(),
        1,
        1,
    )

    arr = np.array(workspace.get_q())[0, -1]
    viz.display(arr)

    p_grad = np.array(workspace.grad_p())[None, :, :]

    # norm = np.linalg.norm(p_grad[:, : seq_len // 2].sum(1)[0])
    # print(norm)
    # if norm > 1:
    #     p_grad = p_grad[:, : seq_len // 2] / norm
    # norm = np.linalg.norm(p_grad[:, seq_len // 2 :].sum(1)[0])
    # print(norm)
    # if norm > 1:
    #     p_grad = p_grad[:, seq_len // 2 :] / norm
    lr = 1e-1
    p_0 -= lr * p_grad[:, : seq_len // 2].sum(1)[0]
    p_1 -= lr * p_grad[:, seq_len // 2 :].sum(1)[0]
    # print(p_grad[:, : seq_len // 2].sum(1)[0])
    # p_2 -= lr * p_grad[:, 200:300].sum(1)[0]
    # p_3 -= lr * p_grad[:, 300:].sum(1)[0]

print(p_np)
