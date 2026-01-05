import pinocchio as pin
import numpy as np
import example_robot_data as erd
import torch
from pathlib import Path
from candlewick import Visualizer, VisualizerConfig, create_recorder_context
import time
from trajopt import vmodel, tool_id, end_SE3
import coal
from scipy.spatial.transform import Rotation as R


src_path = Path("model/src")
files = [str(p) for p in src_path.rglob("*")]
traj = torch.load(
    "/Users/mathisscheffler/Desktop/pinocchio-minimal-main/traj.pt", weights_only=False
)
traj = torch.load(
    "/Users/mathisscheffler/Desktop/pinocchio-minimal-main/learned_traj_complex.pt",
    weights_only=False,
)
traj = torch.load(
    "/Users/mathisscheffler/Desktop/pinocchio-minimal-main/learned_traj.pt",
    weights_only=False,
)
robot = erd.load("ur5")
rmodel, gmodel, _ = pin.buildModelsFromUrdf("model/mantis.urdf", package_dirs=files)

init_pos = pin.neutral(rmodel)
init_pos[len(init_pos) - 5] = -np.pi / 2
init_pos[10] = -np.pi / 2
rmodel = pin.buildReducedModel(rmodel, list(range(7, len(init_pos) + 1)), init_pos)
rmodel.data = rmodel.createData()


def timing_from_energy_profile(
    q_list, model, E_cruise, s_acc_ratio=0.1, s_dec_ratio=0.1
):
    data = model.createData()
    N = len(q_list)
    s = np.zeros(N)
    for i in range(1, N):
        s[i] = s[i - 1] + np.linalg.norm(q_list[i] - q_list[i - 1])
    L = s[-1]
    s_acc, s_dec = s_acc_ratio * L, s_dec_ratio * L

    m_eq = []
    for i in range(N):
        q = q_list[i]
        M = pin.crba(model, data, q)
        data.M = M

        if i == 0:
            dq_ds = q_list[i + 1] - q_list[i]
        elif i == N - 1:
            dq_ds = q_list[i] - q_list[i - 1]
        else:
            dq_ds = 0.5 * (q_list[i + 1] - q_list[i - 1])
        dq_ds /= np.linalg.norm(dq_ds)
        m_eq.append(float(dq_ds.T @ M @ dq_ds))
    m_eq_mean = np.mean(m_eq)
    sdot_cruise = np.sqrt(2 * E_cruise / m_eq_mean)

    def sdot(si):
        if si < s_acc:
            return sdot_cruise * (si / s_acc)
        elif si < L - s_dec:
            return sdot_cruise
        else:
            return sdot_cruise * (1 - (si - (L - s_dec)) / s_dec)

    T = np.zeros(N)
    for i in range(1, N):
        s_mean = 0.5 * (s[i] + s[i - 1])
        T[i] = T[i - 1] + (s[i] - s[i - 1]) / sdot(s_mean)

    return T


T = timing_from_energy_profile(traj, rmodel, 2)

to_remove_names = {
    "end eff",
    "arm",
    "arm1",
    "arm2",
    "arm3",
    "arm4",
    "arm5",
    "plane",
    "cylinder",
    "ball",
}
vmodel: pin.GeometryModel
for name in to_remove_names:
    vmodel.removeGeometryObject(name)

transparent_boxes = {"box1", "box2", "box3", "box4"}
for geom in vmodel.geometryObjects:
    if geom.name in transparent_boxes:
        geom.meshColor = np.array([0.4, 0.2, 0.0, 0.4])

sphere_obj = coal.Sphere(0.01)
parent_frame_id = 0
model_joint_id = 0
placement = end_SE3
geom_end = pin.GeometryObject(
    name, parent_frame_id, model_joint_id, sphere_obj, placement
)
geom_end.meshColor = np.array([1, 1, 1, 1])
vmodel.addGeometryObject(geom_end)

config = VisualizerConfig()
config.width = 1280
config.height = 720
viz = Visualizer(config, rmodel, vmodel)
viz.addFrameViz(tool_id)

viz.display(traj[0])
while not viz.shouldExit:
    t_start = time.time()
    idx = 0

    while not viz.shouldExit and idx < len(traj) - 1:
        t = time.time() - t_start
        while idx < len(T) - 1 and T[idx] < t:
            idx += 1
        pin.forwardKinematics(rmodel, rmodel.data, traj[idx])
        print(rmodel.data.oMf[tool_id])
        print(traj[idx])
        viz.display(traj[idx])

        time.sleep(0.005)

    if viz.shouldExit:
        break

    end_time = time.time()
    print("Trajectoire terminée — pause 3 s")
    while not viz.shouldExit and time.time() - end_time < 3.0:
        viz.display(traj[-1])
        time.sleep(0.01)
