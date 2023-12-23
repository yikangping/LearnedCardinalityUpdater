from enum import Enum, auto


class ArgType(Enum):
    DATASET = auto()
    EVALUATION_TYPE = auto()


DICT_FROM_ARG_TO_ALLOWED_ARG_VALS = {
    ArgType.DATASET: ["census", "forest", "bjaq", "power"],
    ArgType.EVALUATION_TYPE: ["estimate", "drift"]
}


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
