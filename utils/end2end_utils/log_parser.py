import ast
import re
from pathlib import Path
from typing import List

import numpy as np

from utils import path_util, log_util


def parse_lines_with_keywords(src_path: Path, dst_path: Path, start_words: List[str]):
    with src_path.open('r') as src_file, dst_path.open('w') as dst_file:
        for line in src_file:
            if any(line.startswith(word) for word in start_words):
                dst_file.write(line)
                dst_file.write("\n")


def parse_experiment_records(
        src_dir: str = "./end2end/experiment-records",
        dst_dir: str = "./end2end/parsed-records",
        start_words: List[str] = None,
        var_names: List[str] = None,
        list_names: List[str] = None
):
    """
    整理src_dir下的所有txt实验记录，输出到dst_dir下

    使用：
        若需要重新生成：直接删除dst_dir，再次运行本函数
    """
    # 赋默认值
    if start_words is None:
        start_words = [
            "Input arguments",
            "Experiment Summary",
            "Mean JS divergence",
            "WORKLOAD-FINISHED",
            "ReportEsts",
        ]
    if var_names is None:
        var_names = ["Model-update-time"]
    if list_names is None:
        list_names = ['ReportEsts']

    src_dir_path = path_util.get_absolute_path(src_dir)
    dst_dir_path = path_util.get_absolute_path(dst_dir)

    # 确保目标文件夹存在
    dst_dir_path.mkdir(parents=True, exist_ok=True)

    # 处理源文件夹下的每个文件
    for src_file_path in src_dir_path.glob('*.txt'):
        dst_file_path = dst_dir_path / src_file_path.name

        # 跳过已处理的文件
        if dst_file_path.exists():
            continue

        # 提取关键行并写入
        parse_lines_with_keywords(src_file_path, dst_file_path, start_words)

        log_util.append_to_file(dst_file_path, "\n\n\n")

        # 统计求和变量
        for var_name in var_names:
            var_sum = sum_float_var_in_log(dst_file_path, var_name=var_name)
            log_util.append_to_file(dst_file_path, f"Sum of {var_name} = {var_sum:.4f}\n")

        # 统计数组变量
        for list_name in list_names:
            concat_list, match_cnt = concat_list_var_in_log(dst_file_path, list_name=list_name)
            # log_util.append_to_file(dst_file_path, f"Concatenated {list_name} = {concat_list}")
            if list_name == "ReportEsts":
                def generate_report_est_str(arr: list) -> str:
                    arr_max = np.max(arr)
                    quant99 = np.quantile(arr, 0.99)
                    quant95 = np.quantile(arr, 0.95)
                    arr_median = np.quantile(arr, 0.5)
                    arr_mean = np.mean(arr)
                    msg = (f"max: {arr_max:.4f}\t"
                           f"99th: {quant99:.4f}\t"
                           f"95th: {quant95:.4f}\t"
                           f"median: {arr_median:.4f}\t"
                           f"mean: {arr_mean:.4f}\n")
                    return msg

                tuple_len = int(len(concat_list) / match_cnt)
                first_query_errs = concat_list[:tuple_len]
                first_query_est_msg = "The 1st    ReportEsts -> " + generate_report_est_str(first_query_errs)
                log_util.append_to_file(dst_file_path, content=first_query_est_msg)

                after_query_errs = concat_list[tuple_len:]
                after_query_est_msg = "2nd to end ReportEsts -> " + generate_report_est_str(after_query_errs)
                log_util.append_to_file(dst_file_path, content=after_query_est_msg)


def sum_float_var_in_log(file_path: Path, var_name: str) -> float:
    total_sum = 0.0
    with file_path.open('r') as file:
        for line in file:
            if var_name in line:
                # Extract the value after the variable name and sum it up
                try:
                    value_str = line.split(var_name + ':')[1].strip()
                    total_sum += float(value_str)
                except (IndexError, ValueError):
                    # Handle cases where the line format is unexpected or the value is not a float
                    print(f"Warning: Could not parse line: {line.strip()}")
    return total_sum


def concat_list_var_in_log(file_path: Path, list_name: str):
    concatenated_list = []

    # Regular expression to find the desired list name and its contents
    pattern = rf"{re.escape(list_name)}: \[([^\]]*)\]"

    match_cnt = 0

    with file_path.open('r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Extract the list and convert it to a Python list
                list_str = '[' + match.group(1) + ']'
                current_list = ast.literal_eval(list_str)

                # Concatenate to the main list
                concatenated_list.extend(current_list)
                match_cnt += 1

    return concatenated_list, match_cnt


if __name__ == "__main__":
    parse_experiment_records()
