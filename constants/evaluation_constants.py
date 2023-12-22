ALLOWED_EVALUATION_TYPES = ["estimate", "drift"]


def validate_eval_type(eval_type: str):
    """
    Validates if the provided argument is allowed.

    Args:
        eval_type (str): The argument to be validated.

    Raises:
        ValueError: If the dataset is not in the allowed list.
    """
    if eval_type not in ALLOWED_EVALUATION_TYPES:
        raise ValueError(f"Unknown eval_type \"{eval_type}\"")
    print(f"eval_type name \"{eval_type}\" is valid.")


if __name__ == "__main__":
    validate_eval_type("estimate")
    validate_eval_type("drift")
    validate_eval_type("null")
