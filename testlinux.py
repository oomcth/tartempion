import numpy as np
import pinocchio
from math import sqrt
import torch
import copy
import sys
from tqdm import tqdm
from autogradQP import QPkkt
import pinocchio as pin
from colorama import Fore, Style, init
import torch.nn as nn
from scipy.optimize import minimize
import example_robot_data as erd


sys.path.insert(0, "/Users/mscheffl/Desktop/pinocchio-minimal-main/build/python")
import tartempion  # type: ignore

init(autoreset=True)
q_reg = 1e-3
bound = -10
kinematic_workspace = tartempion.KinematicsWorkspace()
workspace = tartempion.QPworkspace()
workspace.set_q_reg(q_reg)
workspace.set_bound(bound)
workspace.set_lambda(-2)
workspace.set_L1(0.00)
workspace.set_rot_w(1.0)
batch_size = 1
seq_len = 101
rmodel = pinocchio.buildSampleModelManipulator()
gmodel = pinocchio.buildSampleGeometryModelManipulator(rmodel)
tool_id = 15
dt = 1e-2
num_threads = 8
# robot = erd.load("panda")
# rmodel, gmodel, vmodel = robot.model, robot.collision_model, robot.visual_model
# tool_id = 19

# robot = erd.load("z1")
# rmodel, gmodel, vmodel = robot.model, robot.collision_model, robot.visual_model
# tool_id = 19

robot = erd.load("ur5")
rmodel, gmodel, vmodel = robot.model, robot.collision_model, robot.visual_model
tool_id = 21

workspace.set_tool_id(tool_id)

assert rmodel.nq == rmodel.nv


rmodel.data = rmodel.createData()
np.random.seed(21)

eq_dim = 1
p_np = np.zeros((batch_size, 6)).astype(np.float64)
p_np[:, -1] = 1
A_np = np.zeros((batch_size * seq_len, eq_dim, 6)).astype(np.float64)
b_np = np.zeros((batch_size, seq_len, 1)).astype(np.float64)
states_init = np.random.randn(*(batch_size, rmodel.nq)).astype(np.float64)


target = [pin.SE3.Identity() for _ in range(batch_size)]
max_translation_norm = 0.5
reach_threshold = 1e-4

for i in range(len(target)):
    while True:
        candidate = pin.SE3.Random()

        # Condition 1 : translation trop grande → on resample
        if np.linalg.norm(candidate.translation) > max_translation_norm:
            continue

        # Condition 2 : atteignabilité via optimisation
        q0 = pin.neutral(rmodel)
        nq = rmodel.nq
        rdata = rmodel.createData()

        def cost(q_np):
            q = np.array(q_np)
            pin.forwardKinematics(rmodel, rdata, q)
            pin.updateFramePlacements(rmodel, rdata)
            M = rdata.oMf[tool_id]
            M_diff = candidate.actInv(M)
            err_translation = np.linalg.norm(M_diff.translation)
            err_rotation = np.linalg.norm(pin.log3(M_diff.rotation))
            return err_translation**2 + err_rotation**2

        res = minimize(cost, q0, method="BFGS", options={"disp": False})

        if res.success and cost(res.x) < reach_threshold:
            target[i] = candidate
            break
# target = [pin.SE3.Identity() for _ in range(batch_size)]

p_np = np.random.random(p_np.shape)
# p_np = np.array(
#     [[0.50731002, 0.36233377, 0.02999527, 0.89096775, 0.39998257, 0.4481885]]
# )


def forward3(target: np.ndarray, q_init, T_star: pin.SE3):
    pin.framesForwardKinematics(rmodel, rmodel.data, q_init)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    target = pin.Motion(target)
    target = pin.exp6(target)
    erreur = pin.log6(rmodel.data.oMf[tool_id].actInv(target.copy()))
    J = pin.computeFrameJacobian(rmodel, rmodel.data, q_init, tool_id)
    Q = J.T @ J + 1e-4 * np.identity(rmodel.nq)
    p = -4 * J.T @ erreur
    print(erreur)
    print(J.T)
    print(p)
    print(Q)
    dq = -np.linalg.inv(Q) @ p
    q = q_init + dt * dq
    print("qd1 python", dq)
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    erreur = pin.log6(rmodel.data.oMf[tool_id].actInv(target.copy()))
    print(erreur)
    J = pin.computeFrameJacobian(rmodel, rmodel.data, q, tool_id)
    Q = J.T @ J + 1e-4 * np.identity(rmodel.nq)
    p = -4 * J.T @ erreur
    dq = -np.linalg.inv(Q) @ p
    q = q + dt * dq
    print("dq python", dq)
    print("last q :", q)
    pin.framesForwardKinematics(rmodel, rmodel.data, q)
    pin.updateFramePlacement(rmodel, rmodel.data, tool_id)
    return (pin.log6(T_star.inverse() * rmodel.data.oMf[tool_id]).vector ** 2).sum()


def forward(p):
    if isinstance(A_np, np.ndarray):
        return tartempion.forward_pass(
            workspace,
            np.tile(p[:, np.newaxis, :], (1, seq_len, 1)),
            A_np * 0,
            b_np * 0,
            states_init,
            rmodel,
            num_threads,
            target,
            dt,
        )
    else:
        return tartempion.forward_pass(
            workspace,
            np.tile(p[:, np.newaxis, :], (1, seq_len, 1)),
            np.zeros((batch_size * seq_len, eq_dim, 6)).astype(np.float64) * 0,
            np.zeros((batch_size, seq_len, 1)).astype(np.float64) * 0,
            states_init,
            rmodel,
            num_threads,
            target,
            dt,
        )


def finite_diff_gradient_batch(forward, p, eps=1e-7):
    p = p.astype(np.float64)
    grad = np.zeros(p.shape + forward(p).shape[1:], dtype=float)
    for b in range(p.shape[0]):
        for i in range(p.shape[1]):
            dp_plus = p.copy()
            dp_minus = p.copy()
            dp_plus[b, i] += eps
            dp_minus[b, i] -= eps
            f_plus = forward(dp_plus)
            f_minus = forward(dp_minus)
            grad[b, i] = (f_plus[b] - f_minus[b]) / (2 * eps)
    return grad


n_test = 5
for i in range(n_test):
    p_np = np.random.random(p_np.shape)
    grad_output = np.zeros((batch_size, seq_len, 2 * 9 + 1))
    out = forward(p_np)
    fd = finite_diff_gradient_batch(forward, p_np)
    print("-" * 50)
    backward = np.array(
        tartempion.backward_pass(
            workspace,
            rmodel,
            grad_output,
            num_threads,
            grad_output.shape[0],
        )
    )
    p_grad = np.array(workspace.grad_p())

    out_torch = QPkkt.apply(
        states_init,
        torch.from_numpy(p_np).unsqueeze(1).repeat(1, seq_len, 1),
        torch.from_numpy(A_np) * 0,
        torch.from_numpy(b_np) * 0,
        rmodel,
        workspace,
        batch_size,
        seq_len,
        eq_dim,
        target,
        dt,
        8,
    )
    loss = out_torch.sum()
    print(loss.item())
    # loss.backward()

    out = forward(p_np)
    qf = np.array(workspace.last_q())[0]

print("ok")
