import argparse
from enum import Enum, auto
from typing import List


class ArgType(Enum):
    DATASET = auto()
    DEBUG = auto()
    END2END = auto()
    EVALUATION_TYPE = auto()


DICT_FROM_ARG_TO_ALLOWED_ARG_VALS = {
    ArgType.DATASET: ["census", "forest", "bjaq", "power"],
    ArgType.EVALUATION_TYPE: ["estimate", "drift"]
}


def add_common_arguments(parser: argparse.ArgumentParser, arg_types: List[ArgType]):
    """
    根据提供的枚举值列表向解析器添加参数。
    """
    for arg_type in arg_types:
        if arg_type == ArgType.DATASET:
            parser.add_argument(
                '--dataset',
                type=str,
                choices=['bjaq', 'census', 'forest', 'power'],
                required=True,
                help='选择数据集：bjaq, census, forest, power'
            )

        if arg_type == ArgType.DEBUG:
            parser.add_argument(
                '--debug',
                action='store_true',  # 当指定 --debug 时，值为 True；否则为 False
                default=False,
                help='启用调试模式'
            )

        if arg_type == ArgType.END2END:
            parser.add_argument(
                '--end2end',
                action='store_true',  # 当指定 --debug 时，值为 True；否则为 False
                default=False,
                help='启用端到端实验'
            )

        if arg_type == ArgType.EVALUATION_TYPE:
            parser.add_argument(
                '--eval_type',
                type=str,
                choices=['estimate', 'drift'],
                required=True,
                help='选择评估类型：estimate, drift'
            )



def validate_argument(arg_type: ArgType, arg_val: str):
    """
    Validates if the provided argument is allowed.

    Args:
        arg_type (ArgType): The type of the argument to be validated.
        arg_val (str): The value of the argument to be validated.

    Raises:
        ValueError: If the argument value is not in the allowed list.
    """
    if arg_val not in DICT_FROM_ARG_TO_ALLOWED_ARG_VALS.get(arg_type, []):
        raise ValueError(f"Validate Argument: UNKNOWN {arg_type.name}=\"{arg_val}\"")
    print(f"Validate Argument: {arg_type.name}=\"{arg_val}\" is valid.")


if __name__ == "__main__":
    # validate_argument(ArgType.DATASET, "census")
    # validate_argument(ArgType.EVALUATION_TYPE, "estimate")
    pass
