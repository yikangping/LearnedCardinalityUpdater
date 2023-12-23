from data import datasets
from constants.dataset_constants import validate_dataset


def load_dataset(dataset: str):
    """
    Loads the dataset.

    Args:
        dataset (str): The dataset to be loaded.

    Returns:
        The loaded dataset
    """
    validate_dataset(dataset)  # Validate dataset name

    if dataset == "census":
        table = datasets.LoadCensus()
    elif dataset == "forest":
        table = datasets.LoadForest()
    elif dataset == "bjaq":
        table = datasets.LoadBJAQ()
    elif dataset == "power":
        table = datasets.LoadPower()
    else:
        raise ValueError(f"Unknown dataset name \"{dataset}\"")

    return table


def load_permuted_dataset(dataset: str, permute: bool = False):
    """
    Loads the permuted dataset.

    Args:
        dataset (str): The dataset to be loaded.
        permute (bool, optional): Whether to permute the dataset. Defaults to False.

    Returns:
        tuple: The loaded dataset and the split indices.
    """
    validate_dataset(dataset)  # Validate dataset name

    if dataset == "census":
        table, split_indices = datasets.LoadPermutedCensus(permute=permute)
    elif dataset == "forest":
        table, split_indices = datasets.LoadPermutedForest(permute=permute)
    elif dataset == "bjaq":
        table, split_indices = datasets.LoadPermutedBJAQ(permute=permute)
    elif dataset == "power":
        table, split_indices = datasets.LoadPermutedPower(permute=permute)
    else:
        raise ValueError(f"Unknown dataset name \"{dataset}\"")
    
    return table, split_indices


def load_partly_permuted_dataset(dataset: str, num_of_sorted_cols: int):
    """
    Loads the partly permuted dataset.

    Args:
        dataset (str): The dataset to be loaded.
        num_of_sorted_cols (int): The number of sorted columns.

    Returns:
        The loaded dataset
    """
    validate_dataset(dataset)  # Validate dataset name

    if dataset == "census":
        table = datasets.LoadPartlyPermutedCensus(num_of_sorted_cols=num_of_sorted_cols)
    else:
        raise ValueError(f"Unknown dataset name \"{dataset}\"")

    return table
