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
from data_template import (
    golf_train,
    golf_test,
    banana_train,
    banana_test,
    fruit_cocktail_train,
    fruit_cocktail_test,
    carrot_train,
    carrot_test,
    peach_train,
    peach_test,
    cube_train,
    cube_test,
)

DISPLAY = False


np.random.seed(21)
pin.seed(21)


def is_position_reachable(
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    return_q: bool = True,
    init_pos: Optional[Union[np.ndarray, None]] = None,
) -> Union[Tuple[bool, None], Tuple[bool, np.ndarray]]:
    q_reg = 1e-3
    dt = 1e-2
    qp = proxsuite.proxqp.dense.QP(rmodel.nq, 0, 0)
    qp.settings.eps_abs = 1e-10
    qp.settings.primal_infeasibility_solving = False
    qp.settings.eps_rel = 0
    if init_pos is not None:
        q0 = init_pos
    else:
        q0 = np.random.randn(rmodel.nq)
    target_position = pin.SE3(target_rotation, target_position)
    max_iter = 10_000
    current_q = q0.copy()
    id = q_reg * np.identity(rmodel.nq)
    score = 1000
    for i in range(max_iter):
        if score < 1e-8:
            break
        pin.framesForwardKinematics(rmodel, rmodel.data, current_q)
        J = pin.computeFrameJacobian(rmodel, rmodel.data, current_q, tool_id, pin.LOCAL)
        err = pin.log6(rmodel.data.oMf[tool_id].actInv(target_position))
        p = J.T @ err.vector
        Q = J.T @ J + id
        qp.init(H=Q, g=p, A=None, b=None, C=None, l=None, u=None)
        qp.solve()
        current_q -= dt * qp.results.x
        score = np.sum(err.vector**2)
    if score < 1e-8:
        return True, current_q
    return False, None


src_path = Path("model/src")
files = [str(p) for p in src_path.rglob("*")]
batch_size = 1
q_reg = 1e-2
bound = -1000
workspace = tartempion.QPworkspace()
workspace.set_echo(False)
workspace.pre_allocate(batch_size)
workspace.set_q_reg(q_reg)
workspace.set_bound(bound)
workspace.set_lambda(-2)
workspace.set_collisions_safety_margin(0.01)
workspace.set_collisions_strength(50)
workspace.view_geometries()
# workspace.add_coll_pair(1, 4)
workspace.add_coll_pair(1, 3)
# workspace.add_coll_pair(0, 2)
workspace.add_coll_pair(0, 3)
# workspace.add_coll_pair(0, 4)
workspace.set_L1(0.00)
workspace.set_rot_w(1.0)
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
seq_len = 1000
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
cylinder_radius = 0.05
cylinder_length = 1
cylinder = coal.Capsule(cylinder_radius, cylinder_length)
ball_radius = 0.1
ball = coal.Sphere(ball_radius)

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

plane_pos = np.array([0, 0, -5])
plane_rot = np.identity(3)
geom_plane = pin.GeometryObject(
    "plane",
    0,
    0,
    plane,
    pin.SE3(plane_rot, plane_pos),
)
workspace.set_coll_pos(2, 0, plane_pos, plane_rot)


caps_pos = np.array([-0.5, 0.1, 0.55])
caps_rot = Ry
geom_caps = pin.GeometryObject(
    "caps",
    0,
    0,
    cylinder,
    pin.SE3(caps_rot, caps_pos),
)
workspace.set_coll_pos(3, 0, caps_pos, caps_rot)
workspace.set_capsule_size(np.array([cylinder_radius]), np.array([cylinder_length]))


ball_pos = np.array([0.1, 0.1, 0.3])
ball_rot = np.identity(3)
geom_ball = pin.GeometryObject(
    "ball",
    0,
    0,
    ball,
    pin.SE3(ball_rot, ball_pos),
)
workspace.set_coll_pos(4, 0, ball_pos, ball_rot)
workspace.set_ball_size(np.array([ball_radius]))


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


viz = viewer.Viewer(rmodel, gmodel, vmodel, True)
viz.viz.viewer["ideal"].set_object(
    g.Sphere(0.01),
    g.MeshLambertMaterial(color=0x00FFFF, transparent=True, opacity=0.5),
)
viz.viz.viewer["current"].set_object(
    g.Sphere(0.01),
    g.MeshLambertMaterial(color=0xFFFF00, transparent=True, opacity=0.5),
)


def sample_p_start():
    while True:
        q = pin.randomConfiguration(rmodel)
        pin.framesForwardKinematics(rmodel, rmodel.data, q)
        T = rmodel.data.oMf[tool_id]
        if T.translation[2] > 0.2:
            return q


q_start = sample_p_start()
q_start = np.array([2.0071, -0.7854, -0.6109, -2.0779, 1.0297, -2.9147])
viz.display(q_start)

pin.framesForwardKinematics(rmodel, rmodel.data, q_start)
R = rmodel.data.oMf[tool_id].rotation.copy()
v = np.array([-0.5, -0.5, 0.5])

end_SE3 = pin.SE3(R, v)
states_init = np.array([q_start])
Pexp = pin.SE3(R, v)
p_np = np.array(pin.log6(Pexp).vector)
p_np = np.repeat(p_np[np.newaxis, :], repeats=batch_size, axis=0)
p_np = np.repeat(p_np[:, np.newaxis, :], repeats=seq_len, axis=1)

np.random.seed(1)

print(q_start)
for iter in tqdm(range(100_000)):
    targets = [end_SE3]

    viz.viz.viewer["current"].set_transform(Pexp.homogeneous)
    viz.viz.viewer["ideal"].set_transform(targets[0].homogeneous)

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

    print(loss[0])

    tartempion.backward_pass(
        workspace,
        rmodel,
        torch.tensor([[[0]]]).cpu().numpy(),
        1,
        1,
    )

    arr = np.array(workspace.get_q())[0, -1]
    viz.display(arr)

    p_grad = np.array(workspace.grad_p())
    p_np -= 1e-1 * p_grad

print(p_np)
