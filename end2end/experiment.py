import argparse
import glob
import os
import shutil
import sys
from pathlib import Path
from typing import List

from end2end.workload import QueryWorkload, DataUpdateWorkload, WorkloadGenerator, BaseWorkload
from utils import path_util
from utils.arg_util import add_common_arguments, ArgType
from utils.end2end_utils import communicator
from utils.end2end_utils.script_runner import PythonScriptRunner


def parse_args():
    parser = argparse.ArgumentParser(description='端到端实验参数解析')

    # 添加通用参数
    common_args: List[ArgType] = [
        ArgType.DATASET,
        ArgType.DEBUG
    ]
    add_common_arguments(parser, arg_types=common_args)

    parser.add_argument(
        '--drift_test',
        type=str,
        choices=['js', 'ddup'],
        required=True,
        help='漂移测试方法：js (JS-divergence), ddup'
    )

    parser.add_argument(
        '--model_update',
        type=str, choices=['update', 'adapt', 'finetune'],
        required=True,
        help='模型更新方法：update (drift_test=ddup), adapt (drift_test=js), finetune (baseline)'
    )

    parser.add_argument(
        '--data_update', type=str,
        choices=['permute-ddup', 'permute', 'sample', 'single'],
        required=True,
        help='数据更新方法：permute (DDUp), sample (FACE), permute (FACE), single (our)'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['naru', 'face'],
        required=True,
        help='模型选择：naru, face'
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='随机种子'
    )

    parser.add_argument(
        '--num_workload',
        type=int,
        default=5,
        help='工作负载数量'
    )

    parsed_args = parser.parse_args()

    return parsed_args


def validate_argument(args):
    # 如果 model_update 是 update，drift_test 必须是 ddup
    if args.model_update == 'update' and args.drift_test != 'ddup':
        sys.exit("参数错误：当 model_update 为 'update' 时，drift_test 必须为 'ddup'。")

    # 如果 model_update 是 adapt，drift_test 必须是 js
    if args.model_update == 'adapt' and args.drift_test != 'js':
        sys.exit("参数错误：当 model_update 为 'adapt' 时，drift_test 必须为 'js'。")

    # random_seed 必须大于等于0
    if args.random_seed < 0:
        sys.exit("参数错误：random_seed 必须大于等于0。")

    # num_workload 必须大于等于1
    if args.num_workload < 1:
        sys.exit("参数错误：num_workload 必须大于等于1。")


def create_workloads(
        args,
        workload_script_path: Path,
        output_file_path: Path = None
) -> List[BaseWorkload]:
    workload_args = {
        # 'data_update': args.data_update,
        'dataset': args.dataset,
        # 'drift_test': args.drift_test,
        'end2end': None,
        # 'model': args.model,
        # 'model_update': args.model_update,
    }
    # 定义查询负载
    query_workload = QueryWorkload(
        args=workload_args,
        script_path=workload_script_path,
        output_file_path=output_file_path
    )
    # 定义数据更新负载
    date_update_workload = DataUpdateWorkload(
        args=workload_args,
        script_path=workload_script_path,
        output_file_path=output_file_path
    )
    # 设置负载权重，根据实际需求修改
    dict_from_workload_to_weight = {
        query_workload: 10,
        date_update_workload: 10,
    }
    # 定义负载生成器
    workload_generator = WorkloadGenerator(workloads=dict_from_workload_to_weight, random_seed=args.random_seed)
    # 生成args.num_workload个工作负载
    generated_workloads = [workload_generator.generate() for _ in range(args.num_workload)]

    return generated_workloads


def run_workloads(
        args,
        workloads: List[BaseWorkload],
        model_update_script_path: Path,
        output_file_path: Path
):
    with open(output_file_path, 'a') as output_file:
        # 顺序运行所有工作负载
        for i, workload in enumerate(workloads):
            start_message = f"Start workload {i+1}/{len(workloads)}, type: {workload.get_type()}\n"
            print(start_message)
            output_file.write(start_message)
            output_file.flush()

            # 运行当前工作负载
            workload.execute_workload()

            # 若为DataUpdateWorkload，则需要检测漂移；若漂移，则更新模型(incremental_train.py)
            if isinstance(workload, DataUpdateWorkload):
                is_drift = communicator.DriftCommunicator().get()
                if is_drift:
                    PythonScriptRunner(
                        script_path=model_update_script_path,
                        args=args,
                        output_file_path=output_file_path
                    ).run_script()

            end_message = f"\nFinish workload {i+1}/{len(workloads)}\n\n\n"
            print(end_message)
            output_file.write(end_message)
            output_file.flush()


def main():
    # 提取参数
    args = parse_args()
    validate_argument(args)

    # 定义文件路径
    workload_script_path = path_util.get_absolute_path('./Naru/eval_model.py')  # 工作负载
    model_update_script_path = path_util.get_absolute_path('./Naru/incremental_train.py')  # 更新模型
    output_file_path = path_util.get_absolute_path('./end2end/experiment-records/record1.txt')  # 实验记录

    # 获取end2end模型路径
    dataset_name = args.dataset
    init_model_reg = f'./models/origin-{dataset_name}*.pt'  # 初始模型
    abs_model_reg: Path = path_util.get_absolute_path(init_model_reg)
    model_paths = glob.glob(str(abs_model_reg))  # 正则匹配结果
    if not model_paths:
        print("No matching model paths found.")
        return
    model_path = model_paths[0]  # 取第1个匹配结果
    print(f"First matching model path: {model_path}")
    model_filename = os.path.basename(model_path)
    end2end_model_path = f'./models/{model_filename}'
    print(f"Model path: {end2end_model_path}")

    # 获取end2end数据集路径
    raw_dataset_path = f'./data/{dataset_name}/{dataset_name}.npy'  # 原始数据集路径
    end2end_dataset_path = f'./data/{dataset_name}/end2end/{dataset_name}.npy'  # end2end数据集路径
    abs_raw_dataset_path = path_util.get_absolute_path(raw_dataset_path)
    abs_end2end_dataset_path = path_util.get_absolute_path(end2end_dataset_path)
    # 将原始数据集复制到end2end文件夹下，如果已存在则覆盖
    shutil.copy2(src=abs_raw_dataset_path, dst=abs_end2end_dataset_path)
    print(f"Copied {abs_raw_dataset_path} to {abs_end2end_dataset_path}")

    with open(output_file_path, 'a') as output_file:
        communicator.ModelPathCommunicator().set(end2end_model_path)  # 设置模型路径
        communicator.DatasetPathCommunicator().set(end2end_dataset_path)  # 设置数据集路径

        # 打印实验参数
        msg = f"Input arguments = {args}\n\n"
        print(msg)
        output_file.write(msg)
        output_file.flush()

        # >>> 创建工作负载 <<<
        generated_workloads: List[BaseWorkload] = create_workloads(
            args=args,
            workload_script_path=workload_script_path,
            output_file_path=output_file_path
        )

        # >>> 运行工作负载 <<<
        run_workloads(
            args=args,
            workloads=generated_workloads,
            model_update_script_path=model_update_script_path,
            output_file_path=output_file_path
        )

    print('Experiment Finished')


if __name__ == "__main__":
    main()
