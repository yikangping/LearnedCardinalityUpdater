from itertools import product
from pathlib import Path

from utils import path_util
from utils.end2end_utils.script_runner import PythonScriptRunner


def run_multi_experiments(script_path: Path, args_dict: dict):
    # Generate all combinations of arguments
    all_combinations = product(*args_dict.values())

    # Iterate over each combination and run the script
    for combination in all_combinations:
        arg_dict = dict(zip(args_dict.keys(), combination))

        # 根据model_update添加drift_test
        if arg_dict['model_update'] == 'update':
            arg_dict['drift_test'] = 'ddup'
        elif arg_dict['model_update'] == 'adapt':
            arg_dict['drift_test'] = 'js'

        print("Going to run 1 experiment with arguments: ", arg_dict)
        runner = PythonScriptRunner(script_path=script_path, args=arg_dict)
        runner.run_script()


if __name__ == "__main__":
    # 实验脚本路径
    experiment_script_path = path_util.get_absolute_path('./end2end/experiment.py')

    # 实验参数组合
    experiment_args = {
        'dataset': ['forest'],
        'model_update': ['update', 'adapt'],
        'data_update': ['single', 'permute-opt', 'sample'],
        'model': ['naru'],
        'num_workload': [50]
    }

    # 运行多组实验
    run_multi_experiments(script_path=experiment_script_path, args_dict=experiment_args)
