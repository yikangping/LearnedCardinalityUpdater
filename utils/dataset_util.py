from data import datasets
from utils import arg_util


class DatasetLoader:
    @staticmethod
    def load_dataset(dataset: str):
        """
        Loads the dataset.

        Args:
            dataset (str): The dataset to be loaded.

        Returns:
            The loaded dataset
        """
        arg_util.validate_argument(arg_util.ArgType.DATASET, dataset)

        if dataset == "census":
            table = datasets.CsvDatasetLoader.load_csv_dataset(dataset_name="census")
        elif dataset == "forest":
            table = datasets.CsvDatasetLoader.load_csv_dataset(dataset_name="forest")
        elif dataset == "bjaq":
            table = datasets.NpyDatasetLoader.load_npy_dataset(dataset_name="BJAQ")
        elif dataset == "power":
            table = datasets.NpyDatasetLoader.load_npy_dataset(dataset_name="power")
        else:
            raise ValueError(f"Unknown dataset name \"{dataset}\"")

        return table

    @staticmethod
    def load_permuted_dataset(dataset: str, permute: bool = False):
        """
        Loads the permuted dataset.

        Args:
            dataset (str): The dataset to be loaded.
            permute (bool, optional): Whether to permute the dataset. Defaults to False.

        Returns:
            tuple: The loaded dataset and the split indices.
        """
        arg_util.validate_argument(arg_util.ArgType.DATASET, dataset)

        if dataset == "census":
            table, split_indices = datasets.CsvDatasetLoader.load_permuted_csv_dataset(dataset_name="census", permute=permute)
        elif dataset == "forest":
            table, split_indices = datasets.CsvDatasetLoader.load_permuted_csv_dataset(dataset_name="forest", permute=permute)
        elif dataset == "bjaq":
            table, split_indices = datasets.NpyDatasetLoader.load_permuted_npy_dataset(dataset_name="BJAQ", permute=permute)
        elif dataset == "power":
            table, split_indices = datasets.NpyDatasetLoader.load_permuted_npy_dataset(dataset_name="power", permute=permute)
        else:
            raise ValueError(f"Unknown dataset name \"{dataset}\"")

        return table, split_indices

    @staticmethod
    def load_partly_permuted_dataset(dataset: str, num_of_sorted_cols: int):
        """
        Loads the partly permuted dataset.

        Args:
            dataset (str): The dataset to be loaded.
            num_of_sorted_cols (int): The number of sorted columns.

        Returns:
            The loaded dataset
        """
        arg_util.validate_argument(arg_util.ArgType.DATASET, dataset)

        if dataset == "census":
            table = datasets.CsvDatasetLoader.LoadPartlyPermutedCensus(num_of_sorted_cols=num_of_sorted_cols)
        else:
            raise ValueError(f"Unknown dataset name \"{dataset}\"")

        return table


class DatasetConverter:
    @staticmethod
    def convert_from_csv_to_npy(dataset: str):
        """
        Converts the dataset.

        Args:
            dataset (str): The dataset to be converted.

        Returns:
            The converted dataset
        """
        arg_util.validate_argument(arg_util.ArgType.DATASET, dataset)

        if dataset == "census":
            table = datasets.convert_csv_dataset(dataset_name="census", output_format=output_format)
        elif dataset == "forest":
            table = datasets.convert_csv_dataset(dataset_name="forest", output_format=output_format)
        else:
            raise ValueError(f"Unknown dataset name \"{dataset}\"")

        return table
