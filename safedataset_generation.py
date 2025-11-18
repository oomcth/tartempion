import numpy as np
import torch
import pinocchio as pin
import example_robot_data as erd

from scipy.spatial.transform import Rotation as R
from typing import Tuple, Union, Optional
import random
import pickle
import proxsuite
from tqdm import tqdm
from data_template import (
    forward_templates,
    backward_templates,
    left_templates,
    right_templates,
    up_templates,
    down_templates,
    no_movement_templates,
    vectors_world,
    roll_templates,
    pitch_templates,
    yaw_templates,
)


scale = 0.7
min_scale = 0.2

max_distance_meters = 0.4
unit_data = {
    "mm": {"factor": 0.001, "spellings": ["mm", "millimeter", "millimeters"]},
    "cm": {"factor": 0.01, "spellings": ["cm", "centimeter", "centimeters"]},
    "dm": {"factor": 0.1, "spellings": ["dm", "decimeter", "decimeters"]},
}
instruction_list = [
    right_templates,
    left_templates,
    forward_templates,
    backward_templates,
    up_templates,
    down_templates,
    no_movement_templates,
    roll_templates,
    pitch_templates,
    yaw_templates,
]


robot = erd.load("ur5")
rmodel, gmodel, vmodel = robot.model, robot.collision_model, robot.visual_model
rmodel.data = rmodel.createData()
tool_id = 21


frame = pin.LOCAL
example_per_train_item = 10
train_sample_size = 0.9


def sample_distance_and_unit():
    unit_type = random.choice(list(unit_data.keys()))
    unit_info = unit_data[unit_type]

    max_distance_in_unit = max_distance_meters / unit_info["factor"]
    if unit_type == "mm":
        distance = random.randint(1, int(max_distance_in_unit))
    elif unit_type == "cm":
        distance = random.randint(1, int(max_distance_in_unit))
    elif unit_type == "dm":
        distance = random.randint(1, int(max_distance_in_unit))
    else:
        raise

    unit_spelling = random.choice(unit_info["spellings"])

    return distance * unit_info["factor"], distance, unit_spelling


def rotation_matrix_x(degrees):
    rad = np.radians(degrees)
    return np.array(
        [[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]]
    )


def rotation_matrix_y(degrees):
    rad = np.radians(degrees)
    return np.array(
        [[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]]
    )


def rotation_matrix_z(degrees):
    rad = np.radians(degrees)
    return np.array(
        [[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]]
    )


def is_position_reachable(
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    return_q: bool = True,
    init_pos: Optional[Union[np.ndarray, None]] = None,
) -> Union[Tuple[bool, None], Tuple[bool, np.ndarray]]:
    scale = 0.7
    q_reg = 1e-3
    dt = 1e-2
    qp = proxsuite.proxqp.dense.QP(rmodel.nq, 0, 0)
    tool_id = 21
    qp.settings.eps_abs = 1e-10
    qp.settings.primal_infeasibility_solving = False
    qp.settings.eps_rel = 0
    if init_pos is not None:
        q0 = init_pos
    else:
        q0 = np.zeros(rmodel.nq)
    target_position = pin.SE3(target_rotation, target_position)
    if (
        np.linalg.norm(target_position.translation) > scale
        or np.linalg.norm(target_position.translation) < min_scale
    ):
        return False, None
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
    if (
        score < 1e-8
        and np.linalg.norm(rmodel.data.oMf[tool_id].translation) < scale
        and np.linalg.norm(rmodel.data.oMf[tool_id].translation) > min_scale
    ):
        return True, current_q
    return False, None



def sample_reachable_pose():
    while True:
        while True:
            x = np.random.randn()
            y = np.random.randn()
            z = np.random.randn()
            position = np.array([x, y, z])

            if (
                np.linalg.norm(position) < scale
                and np.linalg.norm(position) > min_scale
            ):
                break
        rotation_matrix = R.random().as_matrix()
        reachable, q = is_position_reachable(
            position.copy(), rotation_matrix.copy(), return_q=True
        )

        if reachable:
            return position, rotation_matrix, q


train_dataset = []
test_dataset = []


def all_triplets():
    for idx, template_list in enumerate(instruction_list):
        for i, template in enumerate(template_list):
            if idx > 6:
                for sample_id in range(example_per_train_item * 3):
                    yield idx, i, sample_id, template
            else:
                for sample_id in range(example_per_train_item):
                    yield idx, i, sample_id, template


total = sum(
    len(t) * example_per_train_item * (3 if idx > 6 else 1)
    for idx, t in enumerate(instruction_list)
)
for idx, i, sample_id, template in tqdm(all_triplets(), total=total, desc="Processing"):
    distance, distance_spelling, unit = sample_distance_and_unit()
    if idx == 6:
        position, rotation_matrix, q_start = sample_reachable_pose()
        sentence = template
        start_SE3 = end_SE3 = pin.SE3(rotation_matrix, position)
        start_motion = end_motion = pin.log6(start_SE3)
        embedding = torch.tensor([])
        assert np.linalg.norm(end_SE3.translation) < scale
        assert np.linalg.norm(end_SE3.translation) > min_scale
        assert np.linalg.norm(start_SE3.translation) < scale
        assert np.linalg.norm(start_SE3.translation) > min_scale
        sample = (
            sentence,
            start_SE3,
            start_motion,
            end_SE3,
            end_motion,
            unit,
            distance,
            distance_spelling,
            q_start,
            embedding,
        )
        if i > train_sample_size * len(instruction_list[idx]):
            test_dataset.append(sample)
        else:
            train_dataset.append(sample)
    elif idx > 6:
        if idx == 7:
            getRot = rotation_matrix_x
        elif idx == 8:
            getRot = rotation_matrix_y
        else:
            getRot = rotation_matrix_z
        while True:
            position, rotation_matrix, q_start = sample_reachable_pose()
            degree = np.random.randint(-180, 180)
            asked_rot = getRot(degree)
            if frame == pin.LOCAL:
                R_new = rotation_matrix @ asked_rot
            elif frame == pin.WORLD:
                R_new = asked_rot @ rotation_matrix
            else:
                raise
            sentence = template.format(degree)
            if is_position_reachable(
                (position).copy(),
                R_new.copy(),
                True,
                q_start,
            )[0]:
                start_SE3 = pin.SE3(rotation_matrix, position)
                start_motion = pin.log6(start_SE3)
                end_SE3 = pin.SE3(R_new, position)
                end_motion = pin.log6(end_SE3)
                assert np.linalg.norm(end_SE3.translation) < scale
                assert np.linalg.norm(end_SE3.translation) > min_scale
                assert np.linalg.norm(start_SE3.translation) < scale
                assert np.linalg.norm(start_SE3.translation) > min_scale
                embedding = torch.tensor([])
                sample = (
                    sentence,
                    start_SE3,
                    start_motion,
                    end_SE3,
                    end_motion,
                    unit,
                    distance,
                    distance_spelling,
                    q_start,
                    embedding,
                )
                if i > train_sample_size * len(instruction_list[idx]):
                    test_dataset.append(sample)
                else:
                    train_dataset.append(sample)
                break
    else:
        sentence = template.format(distance_spelling, unit)
        if frame == pin.WORLD:
            while True:
                position, rotation_matrix, q_start = sample_reachable_pose()
                if is_position_reachable(
                    (position + distance * vectors_world[idx][:3]).copy(),
                    rotation_matrix.copy(),
                    True,
                    q_start,
                )[0]:
                    start_SE3 = pin.SE3(rotation_matrix, position)
                    start_motion = pin.log6(start_SE3)
                    end_SE3 = pin.SE3(
                        rotation_matrix, position + distance * vectors_world[idx][:3]
                    )
                    end_motion = pin.log6(end_SE3)
                    assert np.linalg.norm(end_SE3.translation) < scale
                    assert np.linalg.norm(end_SE3.translation) > min_scale
                    assert np.linalg.norm(start_SE3.translation) < scale
                    assert np.linalg.norm(start_SE3.translation) > min_scale
                    embedding = torch.tensor([])
                    sample = (
                        sentence,
                        start_SE3,
                        start_motion,
                        end_SE3,
                        end_motion,
                        unit,
                        distance,
                        distance_spelling,
                        q_start,
                        embedding,
                    )
                    if i > train_sample_size * len(instruction_list[idx]):
                        test_dataset.append(sample)
                    else:
                        train_dataset.append(sample)
                    break
        else:
            while True:
                position, rotation_matrix, q_start = sample_reachable_pose()
                if is_position_reachable(
                    (
                        position + distance * rotation_matrix @ vectors_world[idx][:3]
                    ).copy(),
                    rotation_matrix.copy(),
                    True,
                    q_start,
                )[0]:
                    start_SE3 = pin.SE3(rotation_matrix, position)
                    start_motion = pin.log6(start_SE3)
                    end_SE3 = pin.SE3(
                        rotation_matrix,
                        position + distance * rotation_matrix @ vectors_world[idx][:3],
                    )
                    end_motion = pin.log6(end_SE3)
                    assert np.linalg.norm(end_SE3.translation) < scale
                    assert np.linalg.norm(end_SE3.translation) > min_scale
                    assert np.linalg.norm(start_SE3.translation) < scale
                    assert np.linalg.norm(start_SE3.translation) > min_scale
                    embedding = torch.tensor([])
                    sample = (
                        sentence,
                        start_SE3,
                        start_motion,
                        end_SE3,
                        end_motion,
                        unit,
                        distance,
                        distance_spelling,
                        q_start,
                        embedding,
                    )
                    if i > train_sample_size * len(instruction_list[idx]):
                        test_dataset.append(sample)
                    else:
                        train_dataset.append(sample)
                    break

with open("data_qp.pkl", "wb") as f:
    pickle.dump((train_dataset, test_dataset), f)
