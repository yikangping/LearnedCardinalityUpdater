import argparse
import sys
from pathlib import Path
from typing import List

from utils.end2end_utils import communicator
from end2end.workload import QueryWorkload, DataUpdateWorkload, WorkloadGenerator, BaseWorkload
from utils import path_util
from utils.end2end_utils.print_util import redirect_stdout_to_file


def parse_args():
    parser = argparse.ArgumentParser(description='端到端实验参数解析')

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['bjaq', 'census', 'forest', 'power'],
        required=True,
        help='选择数据集：bjaq, census, forest, power'
    )

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
        default=20,
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


def create_workloads(args, workload_script_path: Path, output_file_path: Path) -> List[BaseWorkload]:
    workload_args = {
        'dataset': args.dataset,
        'drift_test': args.drift_test,
        'model_update': args.model_update,
        'data_update': args.data_update,
        'model': args.model,
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
    # 设置负载权重
    dict_from_workload_to_weight = {
        query_workload: 15,
        date_update_workload: 5,
    }
    # 定义负载生成器
    workload_generator = WorkloadGenerator(workloads=dict_from_workload_to_weight, random_seed=args.random_seed)
    # 生成args.num_workload个工作负载
    generated_workloads = [workload_generator.generate() for _ in range(args.num_workload)]

    return generated_workloads


def run_workloads(workloads: List[BaseWorkload]):
    # 顺序运行所有工作负载
    for i, workload in enumerate(workloads):
        print(f"Start workload {i+1}/{len(workloads)}, type: {workload.get_type()}")

        # 运行当前工作负载
        workload.execute_workload()

        # 若为DataUpdateWorkload，则需要检测漂移；若漂移，则更新模型(incremental_train.py)
        if isinstance(workload, DataUpdateWorkload):
            is_drift = communicator.DriftCommunicator().get()
            if is_drift:
                # TODO: 更新模型(incremental_train.py)
                pass

        print(f"Finish workload {i+1}/{len(workloads)}")


def main():
    # 提取参数
    args = parse_args()
    validate_argument(args)

    # 定义文件路径
    workload_script_path = path_util.get_absolute_path('./Naru/eval_model.py')  # 工作负载
    output_file_path = path_util.get_absolute_path('./end2end/experiment-records/record1.txt')  # 实验记录
    init_model_path = './models/origin-census-22.5MB-model26.689-data14.989-300epochs-seed0.pt'  # 初始模型

    # 使用上下文管理器重定向输出
    with redirect_stdout_to_file(output_file_path, mode='w'):
        communicator.ModelPathCommunicator().set(init_model_path)  # 设置模型路径
        print("Input arguments =", args)  # 打印参数

        # >>> 创建工作负载 <<<
        generated_workloads: List[BaseWorkload] = create_workloads(
            args=args,
            workload_script_path=workload_script_path,
            output_file_path=output_file_path
        )

        # >>> 运行工作负载 <<<
        run_workloads(workloads=generated_workloads)


if __name__ == "__main__":
    main()
