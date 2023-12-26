from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from utils import path_util
from utils.end2end_utils import communicator


class Sampler(ABC):
    def __init__(
            self,
            update_fraction: float = 0.2,
            update_size: int = None
    ):
        self.update_fraction = update_fraction
        self.update_size = update_size

    def get_update_size(self, data: np.ndarray):
        if self.update_size:
            return
        data_size = data.shape[0]
        self.update_size = int(data_size * self.update_fraction)
        return self.update_size

    @abstractmethod
    def sample(self, data: np.ndarray):
        pass


class PermuteSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        print("Permute - START")
        n_rows, n_cols = data.shape
        samples = np.empty(shape=(self.update_size, n_cols))

        for i in range(self.update_size):
            if i % 100 == 0:
                print("Permute - {}/{}".format(i, self.update_size))
            idxs = np.random.choice(range(n_rows), n_cols, replace=False)
            for j, idx in enumerate(idxs):
                samples[i, j] = data[idx, j]

        print("Permute - END")
        return samples.astype(np.float32)


class PermuteOptimizedSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        print("Permute - START")
        n_rows, n_cols = data.shape
        samples = np.zeros((self.update_size, n_cols))

        for col in range(n_cols):
            samples[:, col] = np.random.choice(data[:, col], self.update_size, replace=False)

        print("Permute - END")
        return samples.astype(np.float32)


class SingleSamplingSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        idx = np.random.randint(data.shape[0])
        sample_idx = [idx] * self.update_size
        sample = data[sample_idx]
        return sample


class SamplingSampler(Sampler):
    def sample(self, data: np.ndarray):
        self.get_update_size(data)
        sample_idx = np.random.choice(range(data.shape[0]), size=self.update_size, replace=True)
        sample_idx = np.sort(sample_idx)
        sample = data[sample_idx]
        return sample


def create_sampler(
        sampler_type: str,
        update_fraction: float = 0.2,
        update_size: int = None
) -> Sampler:
    if sampler_type == "sample":
        return SamplingSampler(update_fraction=update_fraction, update_size=update_size)
    elif sampler_type == "permute":
        return PermuteSampler(update_fraction=update_fraction, update_size=update_size)
    elif sampler_type == "permute-opt":
        return PermuteOptimizedSampler(update_fraction=update_fraction, update_size=update_size)
    elif sampler_type == "single":
        return SingleSamplingSampler(update_fraction=update_fraction, update_size=update_size)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


class DataUpdater:
    def __init__(self, data: np.ndarray, sampler: Sampler):
        self.sampler = sampler
        self.raw_data = data
        self.sampled_data = None
        self.updated_data = None

    def get_sampled_data(self):
        return self.sampled_data

    def get_updated_data(self):
        return self.updated_data

    def update_data(self):
        # 使用 sampler 采样数据
        self.sampled_data = self.sampler.sample(self.raw_data)
        # 合并原始数据和采样数据
        self.updated_data = np.vstack((self.raw_data, self.sampled_data))

    def store_updated_data_to_file(self, output_path: Path):
        # 将self.data保存到output_path
        np.save(output_path, self.updated_data)

    @staticmethod
    def update_dataset_from_file_to_file(
            data_update_method: str,
            update_fraction: float,
            raw_dataset_path: Path,
            updated_dataset_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        # 从原路径读取当前数据集
        raw_data = np.load(raw_dataset_path).astype(np.float32)  # 原数据

        # 更新数据
        updater = DataUpdater(
            data=raw_data,
            sampler=create_sampler(
                sampler_type=data_update_method,
                update_fraction=update_fraction
            )
        )  # 创建DataUpdater
        updater.update_data()  # 执行数据更新
        sampled_data = updater.get_sampled_data()  # 新增的数据

        # 将更新后的数据保存到新路径
        updater.store_updated_data_to_file(output_path=updated_dataset_path)

        # 计算并保存landmarks
        original_data_end = len(raw_data)
        updated_data_end = original_data_end + len(sampled_data)
        landmarks = [original_data_end, updated_data_end]
        communicator.SplitIndicesCommunicator().set(landmarks)  # 保存landmarks到文件

        return raw_data, sampled_data


if __name__ == "__main__":
    # 初始化
    dataset_name = "bjaq"
    assert dataset_name in ["census", "forest", "bjaq", "power"]
    raw_file_path = f"./data/{dataset_name}/{dataset_name}.npy"
    # raw_file_path = f"./data/{dataset_name}/end2end-{dataset_name}.npy"
    new_file_path = f"./data/{dataset_name}/end2end-{dataset_name}.npy"
    abs_raw_file_path = path_util.get_absolute_path(raw_file_path)
    abs_new_file_path = path_util.get_absolute_path(new_file_path)

    # 读取原始npy数据
    raw_data = np.load(abs_raw_file_path).astype(np.float32)
    print(raw_data.shape)

    # 采样器
    samplers = [
        PermuteOptimizedSampler(update_fraction=0.2),
        # SingleSamplingSampler(update_fraction=0.2),
        # SamplingSampler(update_fraction=0.2)
    ]

    # 更新器
    for sampler in samplers:
        print("New sampler")
        updater = DataUpdater(raw_data, sampler=sampler)
        updater.update_data()
        updated_data = updater.get_updated_data()
        print(updated_data.shape)
        updater.store_updated_data_to_file(abs_new_file_path)
