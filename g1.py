import pinocchio as pin
import time
import viewer
from matplotlib.path import Path
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import tartempion
from tqdm import tqdm


rmodel, gmodel, vmodel = pin.buildModelsFromUrdf(
    "g1_description/g1_29dof_mode_15.urdf",
    root_joint=pin.JointModelFreeFlyer(),
    package_dirs=["g1_description"],
)


q0 = pin.neutral(rmodel)
q0[2] = 0.8
q0[7 + 14] = 0.5


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


def is_in_balance(rmodel, data, q, support_margin=0.10):
    # --- 1. Cinématique ---
    pin.forwardKinematics(rmodel, data, q)
    pin.updateFramePlacements(rmodel, data)
    pin.centerOfMass(rmodel, data, q)
    com = data.com[0][:2]  # (x, y)

    # --- 2. Positions des chevilles ---
    fid_L = rmodel.getFrameId("left_ankle_roll_link")
    fid_R = rmodel.getFrameId("right_ankle_roll_link")

    pL = data.oMf[fid_L].translation
    pR = data.oMf[fid_R].translation

    def rect_around(p):
        r = support_margin
        return np.array(
            [
                [p[0] - r, p[1] - r],
                [p[0] - r, p[1] + r],
                [p[0] + r, p[1] + r],
                [p[0] + r, p[1] - r],
            ]
        )

    pts = np.vstack([rect_around(pL), rect_around(pR)])

    hull = ConvexHull(pts)
    poly = pts[hull.vertices]

    path = Path(poly)
    inside = path.contains_point(com)

    return inside, com, poly


print(rmodel)


keep_ids = list(range(17, 31))
all_ids = list(range(1, len(rmodel.joints)))
lock_ids = [jid for jid in all_ids if jid not in keep_ids]
rmodel, [gmodel, vmodel] = pin.buildReducedModel(rmodel, [gmodel, vmodel], lock_ids, q0)

data = rmodel.createData()
viz = viewer.Viewer(rmodel, gmodel, vmodel, candlewick=True)

q0 = np.array(
    [
        0.44791734,
        0.74484575,
        -1.95861577,
        0.68967171,
        1.79089002,
        -0.46196909,
        1.09168256,
        2.52210376,
        -1.25696321,
        -1.3010015,
        0.28589358,
        1.23480038,
        1.28480901,
        -0.82469511,
    ]
)


for i, frames in enumerate(rmodel.frames):
    print(frames.name)

fid_left_hand = rmodel.getFrameId("left_rubber_hand")
fid_right_hand = rmodel.getFrameId("right_rubber_hand")
pin.framesForwardKinematics(rmodel, data, q0)
target = data.oMf[fid_left_hand].copy()
target.translation = target.translation + np.array([1, 0, 0])

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
workspace.set_tool_id(fid_left_hand)
workspace.set_equilibrium(True)
t2 = pin.log6(target).vector
t2 = np.repeat(t2[None, :], seq_len, axis=0)
workspace.set_equilibrium_second_target(t2)
workspace.set_equilibrium_tool_id(fid_right_hand)


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
print("done")
arr = np.array(workspace.get_q())[0]
# in_balance, com_xy, polygon = is_in_balance(rmodel, data, q0)
# print(in_balance)
# plot_support_and_com(polygon, com_xy, in_balance)
for i in tqdm(range(len(arr))):
    viz.display(arr[i])
    time.sleep(dt)
# in_balance, com_xy, polygon = is_in_balance(rmodel, data, arr[-1])
# print(in_balance)
# plot_support_and_com(polygon, com_xy, in_balance)
