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
import proxsuite
from typing import Optional, Union, Tuple
import meshcat.geometry as g
import tartempion


np.random.seed(2)


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
q_reg = 1e-3
bound = -1000
workspace = tartempion.QPworkspace()
workspace.pre_allocate(batch_size)
workspace.set_q_reg(q_reg)
workspace.set_bound(bound)
workspace.set_lambda(-2)
workspace.set_collisions_safety_margin(0.0)
workspace.set_collisions_strength(50)
workspace.view_geometries()
# workspace.add_coll_pair(1, 4)
# workspace.add_coll_pair(1, 2)
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


def random_z_rotation():
    theta = np.random.uniform(0, 2 * np.pi)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


# tool_id = 21
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
cylinder_radius = 0.03
cylinder_length = 0.2
cylinder = coal.Capsule(cylinder_radius, cylinder_length)
ball_size = 0.14
ball = coal.Sphere(ball_size)

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

cylinder_pos = np.array([0.25, 0.5, cylinder_radius])
cylinder_rot = random_z_rotation() @ Ry
geom_cylinder = pin.GeometryObject(
    "cylinder",
    0,
    0,
    cylinder,
    pin.SE3(cylinder_rot, cylinder_pos),
)
workspace.set_coll_pos(3, 0, cylinder_pos, cylinder_rot)
workspace.set_capsule_size(np.array([cylinder_radius]), np.array([cylinder_length]))

ball_pos = np.array([0.25, 0.25, ball_size])
ball_rot = np.identity(3)
geom_ball = pin.GeometryObject(
    "ball",
    0,
    0,
    ball,
    pin.SE3(ball_rot, ball_pos),
)
workspace.set_coll_pos(4, 0, ball_pos, ball_rot)
workspace.set_ball_size(np.array([ball_size]))


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
# viz.viz.viewer["ideal"].set_object(
#     g.Sphere(0.05),
#     g.MeshLambertMaterial(color=0x00FFFF, transparent=True, opacity=0.5),
# )
# viz.viz.viewer["current"].set_object(
#     g.Sphere(0.05),
#     g.MeshLambertMaterial(color=0xFFFF00, transparent=True, opacity=0.5),
# )

_, init_pos = is_position_reachable(np.array([0.25, 0.0, 0.0]), Ry2, True)
viz.display(init_pos)
np.printoptions(precision=20)
print(init_pos)

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


for l in tqdm(range(1000)):
    end_SE3 = pin.SE3.Random()
    q_start = sample_p_start()
    states_init = init_pos[None, :]
    R = Ry2
    v = np.array([0.25, 0.5, cylinder_radius / 2])
    P_exp = pin.SE3(Ry2, v)
    p_np = np.array(pin.log6(P_exp).vector)

    p_np = np.repeat(p_np[np.newaxis, :], repeats=batch_size, axis=0)
    p_np = np.repeat(p_np[:, np.newaxis, :], repeats=seq_len, axis=1)

    viz.display(states_init[0])
    targets = [end_SE3]

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
    ros_play = False
    if ros_play:
        torch.save(arr[0], "traj.pt")
        import rclpy
        from rclpy.node import Node
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        class TrajectorySender(Node):
            def __init__(self, topic_name="/trajectory"):
                super().__init__("trajectory_sender")
                self.publisher = self.create_publisher(JointTrajectory, topic_name, 10)
                self.get_logger().info(
                    f"Trajectory sender node started, publishing to '{topic_name}'"
                )

                self.timer = self.create_timer(10.0, self.send_trajectory)

            def build_traj_sin_good(
                self, amplitude=0.5, frequency=0.2, duration=5.0, dt=0.1
            ):
                traj = JointTrajectory()
                traj.joint_names = [
                    "left_shoulder_pan_joint",
                    "left_shoulder_lift_joint",
                    "left_elbow_joint",
                    "left_wrist_1_joint",
                    "left_wrist_2_joint",
                    "left_wrist_3_joint",
                ]

                num_points = int(duration / dt)
                for i in range(num_points):
                    t = i * dt
                    point = JointTrajectoryPoint()

                    point.positions = np.array([0.0, -1.5708, 0, -1.5708, 0.0, 0.0])
                    point.positions[0] += amplitude * np.sin(2 * np.pi * frequency * t)
                    point.positions[2] += amplitude * np.sin(2 * np.pi * frequency * t)

                    point.time_from_start.sec = int(t)
                    point.time_from_start.nanosec = int((t - int(t)) * 1e9)

                    traj.points.append(point)
                return traj

            def build_traj(self, amplitude=1, frequency=0.1, duration=5.0, dt=0.1):
                dt = 1e-2
                traj = JointTrajectory()
                traj.joint_names = [
                    "left_shoulder_pan_joint",
                    "left_shoulder_lift_joint",
                    "left_elbow_joint",
                    "left_wrist_1_joint",
                    "left_wrist_2_joint",
                    "left_wrist_3_joint",
                ]

                num_points = int(duration / dt)
                for i in range(num_points):
                    t = i * dt
                    point = JointTrajectoryPoint()
                    point.positions = arr[0, i]
                    point.time_from_start.sec = int(t)
                    point.time_from_start.nanosec = int((t - int(t)) * 1e9)
                    traj.points.append(point)
                return traj

            def send_trajectory(self):
                traj = self.build_traj()
                self.publisher.publish(traj)
                self.get_logger().info("Trajectory published.")

        rclpy.init(args=None)
        node = TrajectorySender()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    np.set_printoptions(precision=100)
    print("q", arr[0, -2])

    tartempion.backward_pass(
        workspace,
        rmodel,
        torch.tensor([[[0]]]).cpu().numpy(),
        1,
        1,
    )

    print("enter to continue")
    input()
    print("continuing")

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
