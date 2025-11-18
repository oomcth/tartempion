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
        "distance": torch.tensor(transposed[6]),
        "distance_spelling": list(transposed[7]),
        "q_start": torch.stack([torch.as_tensor(q) for q in transposed[8]]),
        "embedding": torch.stack([torch.as_tensor(e) for e in transposed[9]]),
    }


if __name__ == "__main__":

    with open("data_qp.pkl", "rb") as f:
        train_dataset, test_dataset = pickle.load(f)

    num_samples = 5

    indices = random.sample(range(len(train_dataset)), num_samples)

    norms = []

    for i, sample in enumerate(train_dataset):
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
        ) = sample

        norm_val = np.linalg.norm(end_SE3.translation)
        norms.append((norm_val, sentence, i))

    norms_sorted = sorted(norms, key=lambda x: x[0], reverse=True)

    print("\nTop 10 des plus grandes normes pour T_star :\n")
    for val, sent, idx in norms_sorted[:10]:
        print(f"{idx:4d}  norme = {val:.6f}   phrase = {sent}")

    for i in indices:
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
        ) = train_dataset[i]
        print(sentence)
        print(start_SE3)
        print(end_SE3)
        input()

    train_dataset_torch = TrajectoryDataset(train_dataset)
    test_dataset_torch = TrajectoryDataset(test_dataset)

    with open("train_qp_coll.pkl", "wb") as f:
        pickle.dump(train_dataset_torch, f)

    with open("test_qp_coll.pkl", "wb") as f:
        pickle.dump(test_dataset_torch, f)

