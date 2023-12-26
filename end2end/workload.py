import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

from utils import path_util
from utils.script_runner import PythonScriptRunner


class BaseWorkload(ABC):
    """
    The base class of all workloads.
    """
    def get_type(self):
        return self.__class__.__name__

    @abstractmethod
    def execute_workload(self):
        pass


class PythonScriptWorkload(BaseWorkload):
    """
    A workload that runs a python script.
    """
    def __init__(self, python_script_path: Path, args: Dict[str, str]):
        self.script_path = python_script_path
        self.args = args

    def execute_workload(self):
        PythonScriptRunner(script_path=self.script_path, args=self.args).run_script()


class QueryWorkload(PythonScriptWorkload):
    """
    A workload that queries the model.
    """
    def __init__(self, script_path: Path, args: Dict[str, str]):
        args['eval_type'] = 'estimate'
        super().__init__(script_path, args)


class DataUpdateWorkload(PythonScriptWorkload):
    """
    A workload that updates the data & check drift.
    """
    def __init__(self, script_path: Path, args: Dict[str, str]):
        args['eval_type'] = 'drift'
        super().__init__(script_path, args)


class WorkloadGenerator:
    """
    A workload generator that randomly generates workloads from a list of workloads,
    with the possibility to assign weights to each workload.
    """
    def __init__(self, workloads: Dict[BaseWorkload, int], random_seed: int):
        self.workloads = workloads
        random.seed(random_seed)  # Set the random seed
        self.choices = self._create_weighted_choices()

    def _create_weighted_choices(self):
        """
        Create a list of workloads where each workload appears a number of times
        corresponding to its weight.
        """
        choices = []
        for workload, weight in self.workloads.items():
            choices.extend([workload] * weight)
        return choices

    def generate(self) -> BaseWorkload:
        """
        Randomly selects a workload based on the defined weights.
        """
        return random.choice(self.choices)


if __name__ == "__main__":
    relative_script_path = path_util.get_absolute_path('./Naru/eval_model.py')
    workload_args = {
        'dataset': 'census',
        'drift_test': 'ddup',
        'model_update': 'adapt',
        'data_update': 'permute-ddup',
        'model': 'naru',
    }
    # Initialize workloads
    allowed_workloads = {
        QueryWorkload(args=workload_args, script_path=relative_script_path): 15,
        DataUpdateWorkload(args=workload_args, script_path=relative_script_path): 5
    }
    generator = WorkloadGenerator(workloads=allowed_workloads, random_seed=42)
    generated_workloads = [generator.generate() for _ in range(20)]

    for i, cur_workload in enumerate(generated_workloads):
        if isinstance(cur_workload, DataUpdateWorkload):
            print('is DataUpdateWorkload')