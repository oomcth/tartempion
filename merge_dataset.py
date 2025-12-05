import numpy as np
import torch
import pickle


def augment_old_sample(sample10, ref_sample18):
    (
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
    ) = sample10

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        ball_pos_ref,
        ball_rot_ref,
        ball_size_ref,
        cylinder_radius_ref,
        cylinder_length_ref,
        obj_data_position_ref,
        obj_data_rot_ref,
        obj_feature_ref,
    ) = ref_sample18

    def rand_like(x):
        x = np.asarray(x)
        return np.random.randn(*x.shape)

    ball_pos = rand_like(ball_pos_ref) - np.array([20.0, 20.0, 20.0])
    ball_rot = rand_like(ball_rot_ref)
    ball_size = 0 * rand_like(ball_size_ref) ** 2
    cylinder_radius = 0 * rand_like(cylinder_radius_ref) ** 2
    cylinder_length = 0 * rand_like(cylinder_length_ref) ** 2
    obj_data_position = rand_like(obj_data_position_ref)
    obj_data_rot = rand_like(obj_data_rot_ref)
    obj_feature = -1

    return (
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
        ball_pos,
        ball_rot,
        ball_size,
        cylinder_radius,
        cylinder_length,
        obj_data_position,
        obj_data_rot,
        obj_feature,
    )


if __name__ == "__main__":
    with open("data_v1.pkl", "rb") as f:
        data_v1 = pickle.load(f)

    with open("data_v2.pkl", "rb") as f:
        data_v2 = pickle.load(f)

    ref_sample = data_v2[0]

    data_v1_aug = [augment_old_sample(s, ref_sample) for s in data_v1]

    merged_data = data_v1_aug + data_v2

    print(f"✅ Dataset fusionné : {len(merged_data)} échantillons")
    print(f"Chaque sample contient {len(merged_data[0])} champs")

    with open("data_merged.pkl", "wb") as f:
        pickle.dump(merged_data, f)

    print("💾 Fichier sauvegardé sous : data_merged.pkl")
