ALLOWED_DATASETS = ["census", "forest", "bjaq", "power"]


def validate_dataset(dataset: str):
    """
    Validates if the provided dataset is allowed.

    Args:
        dataset (str): The dataset to be validated.

    Raises:
        ValueError: If the dataset is not in the allowed list.
    """
    if dataset not in ALLOWED_DATASETS:
        raise ValueError(f"Unknown dataset name \"{dataset}\"")
    print(f"Dataset name \"{dataset}\" is valid.")


if __name__ == "__main__":
    validate_dataset("census")
    validate_dataset("null")
