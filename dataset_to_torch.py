import numpy as np
import torch
import pickle
import pinocchio as pin
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random


class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def custom_collate_fn(batch):
    transposed = list(zip(*batch))

    return {
        "sentence": list(transposed[0]),
        "start_SE3": list(transposed[1]),
        "start_motion": list(transposed[2]),
        "end_SE3": list(transposed[3]),
        "end_motion": list(transposed[4]),
        "unit": list(transposed[5]),
        "distance": torch.as_tensor(transposed[6]),
        "distance_spelling": list(transposed[7]),
        "q_start": torch.stack([torch.as_tensor(q) for q in transposed[8]]),
        "embedding": torch.stack([torch.as_tensor(e) for e in transposed[9]]),
        "ball_pos": torch.stack([torch.as_tensor(x) for x in transposed[10]]),
        "ball_rot": torch.stack([torch.as_tensor(x) for x in transposed[11]]),
        "ball_size": torch.stack([torch.as_tensor(x) for x in transposed[12]]),
        "cylinder_radius": torch.stack([torch.as_tensor(x) for x in transposed[13]]),
        "cylinder_length": torch.stack([torch.as_tensor(x) for x in transposed[14]]),
        "obj_data_position": torch.stack([torch.as_tensor(x) for x in transposed[15]]),
        "obj_data_rot": torch.stack([torch.as_tensor(x) for x in transposed[16]]),
        "obj_feature": torch.stack([torch.as_tensor(x) for x in transposed[17]]),
    }


if __name__ == "__main__":
    with open("data_merged.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open("data_merged_test.pkl", "rb") as f:
        test_dataset = pickle.load(f)

    train_dataset_torch = TrajectoryDataset(train_dataset)
    test_dataset_torch = TrajectoryDataset(test_dataset)

    print(len(train_dataset_torch))
    print(len(test_dataset_torch))

    with open("train_qp_coll.pkl", "wb") as f:
        pickle.dump(train_dataset_torch, f)

    with open("test_qp_coll.pkl", "wb") as f:
        pickle.dump(test_dataset_torch, f)
