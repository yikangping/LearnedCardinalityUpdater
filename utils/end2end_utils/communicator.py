from abc import ABC, abstractmethod
from pathlib import Path

from utils import path_util

CURRENT_MODEL_PATH_TXT = './end2end/communicate/model_path.txt'
CURRENT_DATASET_PATH_TXT = './end2end/communicate/dataset_path.txt'
CURRENT_IS_DRIFT_TXT = './end2end/communicate/is_drift.txt'
CURRENT_SPLIT_INDICES_TXT = './end2end/communicate/split_indices.txt'


class FileCommunicator:
    def __init__(self, file_path: str):
        self.abs_file_path = path_util.get_absolute_path(file_path)

    def get(self):
        with open(self.abs_file_path, 'r') as file:
            content = file.read()
        return content

    def set(self, content: str):
        with open(self.abs_file_path, 'w') as file:
            file.write(content)


class PathCommunicator(FileCommunicator):
    def __init__(self, file_path: str, prompt: str):
        super().__init__(file_path)
        self.prompt = prompt  # 新增属性来区分类型

    def get(self) -> Path:
        """
        读取txt获取当前路径
        """
        with open(self.abs_file_path, 'r') as file:
            content = file.read()
        print(f"CUR-{self.prompt}-PATH={content}")
        abs_path = path_util.get_absolute_path(content)
        return abs_path

    def set(self, new_path: str):
        """
        将new_path写入txt用于记录当前路径
        """
        with open(self.abs_file_path, 'w') as file:
            file.write(new_path)
        print(f"NEW-{self.prompt}-PATH={new_path}")


class ModelPathCommunicator(PathCommunicator):
    def __init__(self, txt_path: str = CURRENT_MODEL_PATH_TXT):
        super().__init__(file_path=txt_path, prompt="MODEL")


class DatasetPathCommunicator(PathCommunicator):
    def __init__(self, txt_path: str = CURRENT_DATASET_PATH_TXT):
        super().__init__(file_path=txt_path, prompt="DATASET")


class DriftCommunicator(FileCommunicator):
    def __init__(self, file_path: str = CURRENT_IS_DRIFT_TXT):
        super().__init__(file_path)

    def get(self) -> bool:
        """
        读取txt，若为"true"则返回True，否则返回False
        """
        with open(self.abs_file_path, 'r') as file:
            content = file.read()
        return content == 'true'

    def set(self, is_drift: bool):
        """
        将"true"或"false"写入txt
        """
        with open(self.abs_file_path, 'w') as file:
            content = 'true' if is_drift else 'false'
            file.write(content)


class CommaSplitArrayCommunicator(FileCommunicator):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    def get(self) -> list:
        """
        读取txt文件，将字符串分割为数组
        """
        with open(self.abs_file_path, 'r') as file:
            content = file.read()
        return content.split(',') if content else []

    def set(self, array: list):
        """
        将数组转换为字符串并写入txt文件
        """
        array_str = ','.join(map(str, array))
        with open(self.abs_file_path, 'w') as file:
            file.write(array_str)


class SplitIndicesCommunicator(CommaSplitArrayCommunicator):
    def __init__(self, txt_path: str = CURRENT_SPLIT_INDICES_TXT):
        super().__init__(txt_path)


if __name__ == '__main__':
    # 示例用法
    model_path_communicator = ModelPathCommunicator()
    dataset_path_communicator = DatasetPathCommunicator()

    original_model_path = model_path_communicator.get()
    model_path_communicator.set('./models/new-model-path.pt')
    new_model_path = model_path_communicator.get()

    original_dataset_path = dataset_path_communicator.get()
    dataset_path_communicator.set('./datasets/new-dataset-path.pt')
    new_dataset_path = dataset_path_communicator.get()

    print(original_model_path)
    print(new_model_path)
    print(original_dataset_path)
    print(new_dataset_path)
