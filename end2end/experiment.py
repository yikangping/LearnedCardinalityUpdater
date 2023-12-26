import argparse
import sys

from utils import path_util
import workload_util


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


class EndToEndExperiment:
    pass


def main():
    # >>> 提取参数 <<<
    args = parse_args()
    validate_argument(args)
    print("Input arguments =", args)

    # >>> 创建工作负载 <<<
    workload_script_path = path_util.get_absolute_path('./Naru/eval_model.py')
    workload_args = {
        'dataset': args.dataset,
        'drift_test': args.drift_test,
        'model_update': args.model_update,
        'data_update': args.data_update,
        'model': args.model,
    }
    allowed_workloads = {
        workload_util.QueryWorkload(args=workload_args, script_path=workload_script_path): 15,
        workload_util.DataUpdateWorkload(args=workload_args, script_path=workload_script_path): 5,
    }
    workload_generator = workload_util.WorkloadGenerator(workloads=allowed_workloads, random_seed=args.random_seed)
    # 生成args.num_workload个工作负载
    generated_workloads = [workload_generator.generate() for _ in range(args.num_workload)]

    # >>> 运行工作负载 <<<
    # 顺序运行所有工作负载
    for i, workload in enumerate(generated_workloads):
        print(f"Start workload {i+1}/{args.num_workload}, which is a {workload.get_workload_name()}")

        # 运行当前工作负载
        # workload.execute_workload()

        # 若为DataUpdateWorkload，则需要检测漂移；若漂移，则更新模型
        if isinstance(workload, workload_util.DataUpdateWorkload):
            pass

        # print(f"Finish workload {i+1}/{args.num_workload}, which is a {workload.get_workload_name()}")


if __name__ == "__main__":
    main()
