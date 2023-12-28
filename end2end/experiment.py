import argparse
import glob
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from end2end.workload import QueryWorkload, DataUpdateWorkload, WorkloadGenerator, BaseWorkload
from utils import path_util, log_util
from utils.arg_util import add_common_arguments, ArgType
from utils.end2end_utils import communicator, log_parser
from utils.end2end_utils.script_runner import PythonScriptRunner


def parse_args():
    parser = argparse.ArgumentParser(description='端到端实验参数解析')

    # 添加通用参数
    common_args: List[ArgType] = [
        ArgType.DATA_UPDATE,
        ArgType.DATASET,
        ArgType.DEBUG,
        ArgType.DRIFT_TEST,
        ArgType.MODEL_UPDATE
    ]
    add_common_arguments(parser, arg_types=common_args)

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
    query_workload_args = {
        'dataset': args.dataset,
        'end2end': None,
    }
    data_update_workload_args = {
        'data_update': args.data_update,
        'dataset': args.dataset,
        'drift_test': args.drift_test,
        'end2end': None
    }
    # 定义查询负载
    query_workload = QueryWorkload(
        args=query_workload_args,
        script_path=workload_script_path,
        output_file_path=output_file_path
    )
    # 定义数据更新负载
    data_update_workload = DataUpdateWorkload(
        args=data_update_workload_args,
        script_path=workload_script_path,
        output_file_path=output_file_path
    )
    # 设置负载权重，根据实际需求修改
    weight_query = 15
    weight_data_update = 10
    dict_from_workload_to_weight = {
        query_workload: weight_query,
        data_update_workload: weight_data_update,
    }
    msg = f"Workload weights: query={weight_query}, data-update={weight_data_update}\n"
    log_util.append_to_file(output_file_path, msg)

    # 定义负载生成器
    workload_generator = WorkloadGenerator(workloads=dict_from_workload_to_weight, random_seed=args.random_seed)

    # 生成args.num_workload个工作负载
    # 第1个工作负载为查询负载，其余为随机选取
    generated_workloads = [query_workload] + [workload_generator.generate() for _ in range(args.num_workload - 1)]

    # 打印生成的工作负载
    workloads_description = [workload.get_type() for workload in generated_workloads]
    msg = f"Workloads: {workloads_description}\n"
    log_util.append_to_file(output_file_path, msg)

    return generated_workloads


def run_workloads(
        args,
        workloads: List[BaseWorkload],
        model_update_script_path: Path,
        output_file_path: Path
) -> int:
    drift_count = 0  # 漂移次数

    # 顺序运行所有工作负载
    for i, workload in enumerate(workloads):
        start_message = (f"WORKLOAD-START | "
                         f"Type: {workload.get_type()} | "
                         f"Progress: {i + 1}/{len(workloads)}\n")
        drift_message = f"\nDRIFT-DETECTED after {i + 1}-th workload\n"

        # 打印工作负载开始信息
        log_util.append_to_file(output_file_path, start_message)

        # 运行当前工作负载
        workload_start_time = time.time()
        workload.execute_workload()
        workload_time = time.time() - workload_start_time

        # 若为DataUpdateWorkload，则需要检测漂移；若漂移，则更新模型(incremental_train.py)
        model_update_time = 0
        if isinstance(workload, DataUpdateWorkload):
            is_drift = communicator.DriftCommunicator().get()
            if is_drift:
                drift_count += 1
                # 打印漂移检测信息
                log_util.append_to_file(output_file_path, drift_message)

                # 运行模型更新脚本
                incremental_train_args = {
                    'dataset': args.dataset,
                    'end2end': None,
                    'model_update': args.model_update
                }
                model_update_start_time = time.time()
                PythonScriptRunner(
                    script_path=model_update_script_path,
                    args=incremental_train_args,
                    output_file_path=output_file_path
                ).run_script()
                model_update_time = time.time() - model_update_start_time

        # 记录工作负载时间和模型更新时间
        end_message = (f"\nWORKLOAD-FINISHED | "
                       f"Type: {workload.get_type()} | "
                       f"Progress: {i + 1}/{len(workloads)} | "
                       f"Workload-time: {workload_time:.6f} | "  # 将工作负载时间精确到小数点后六位
                       f"Model-update-time: {model_update_time:.6f}\n\n\n")  # 将模型更新时间精确到小数点后六位

        # 打印工作负载结束信息
        log_util.append_to_file(output_file_path, end_message)

    return drift_count


def main():
    start_time = time.time()

    # 提取参数
    args = parse_args()
    validate_argument(args)

    # 定义文件路径
    workload_script_path = path_util.get_absolute_path('./Naru/eval_model.py')  # 生成工作负载的脚本
    model_update_script_path = path_util.get_absolute_path('./Naru/incremental_train.py')  # 更新模型的脚本
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%y%m%d-%H%M")  # 格式化日期和时间为 'yyMMdd-HHmm' 格式
    output_file_name = (f'{args.dataset}+'  # 数据集
                        f'{args.model_update}+'  # 模型更新方式(adapt/update)
                        f'{args.data_update}+'  # 数据更新方式(single/permute/sample)
                        f'wl{args.num_workload}+'  # 工作负载数量
                        f't{formatted_datetime}'   # 实验时间
                        f'.txt')
    output_file_path = path_util.get_absolute_path(f"./end2end/experiment-records/{output_file_name}")  # 实验记录文件路径
    print('Output file path:', output_file_path)

    # 打印实验参数
    log_util.append_to_file(output_file_path, f"Input arguments = {args}\n")

    # 获取end2end模型路径
    dataset_name = args.dataset
    init_model_reg = f'./models/origin-{dataset_name}*.pt'  # 初始模型
    abs_model_reg: Path = path_util.get_absolute_path(init_model_reg)
    model_paths = glob.glob(str(abs_model_reg))  # 正则匹配结果
    if not model_paths:
        print("No matching model paths found.")
        return
    model_path = model_paths[0]  # 取第1个匹配结果
    src_model_filename = os.path.basename(model_path)
    src_model_path = f'./models/{src_model_filename}'
    print(f"Source Model path: {src_model_path}")
    # 将原始模型复制到end2end文件夹下，如果已存在则覆盖
    end2end_model_path = f'./models/end2end/{src_model_filename}'  # end2end模型路径
    abs_src_model_path = path_util.get_absolute_path(src_model_path)
    abs_end2end_model_path = path_util.get_absolute_path(end2end_model_path)
    shutil.copy2(src=abs_src_model_path, dst=abs_end2end_model_path)

    # 获取end2end数据集路径
    src_dataset_path = f'./data/{dataset_name}/{dataset_name}.npy'  # 原始数据集路径
    end2end_dataset_path = f'./data/{dataset_name}/end2end/{dataset_name}.npy'  # end2end数据集路径
    abs_src_dataset_path = path_util.get_absolute_path(src_dataset_path)
    abs_end2end_dataset_path = path_util.get_absolute_path(end2end_dataset_path)
    # 将原始数据集复制到end2end文件夹下，如果已存在则覆盖
    shutil.copy2(src=abs_src_dataset_path, dst=abs_end2end_dataset_path)

    # 记录日志
    log_util.append_to_file(output_file_path, f"MODEL-PATH={abs_end2end_model_path}\n")
    log_util.append_to_file(output_file_path, f"DATASET-PATH={abs_end2end_dataset_path}\n")

    # 设置communicator
    communicator.ModelPathCommunicator().set(end2end_model_path)  # 设置end2end模型路径
    communicator.DatasetPathCommunicator().set(end2end_dataset_path)  # 设置end2end数据集路径
    communicator.RandomSeedCommunicator().set(0)  # 设置初始随机种子为0

    # >>> 创建工作负载 <<<
    generated_workloads: List[BaseWorkload] = create_workloads(
        args=args,
        workload_script_path=workload_script_path,
        output_file_path=output_file_path
    )

    # 记录日志
    log_util.append_to_file(output_file_path, "\n\n\n")

    # >>> 运行工作负载 <<<
    drift_count = run_workloads(
        args=args,
        workloads=generated_workloads,
        model_update_script_path=model_update_script_path,
        output_file_path=output_file_path
    )

    # 打印实验总结
    experiment_summary = (f"\n\n\n"
                          f"Experiment Summary: "
                          f"#drift={drift_count} | "
                          f"total-time={time.time() - start_time:.6f}"
                          f"\n")
    log_util.append_to_file(output_file_path, experiment_summary)

    # 整理实验记录
    log_parser.parse_experiment_records()


if __name__ == "__main__":
    main()
