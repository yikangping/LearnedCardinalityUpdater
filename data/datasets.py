"""Dataset registrations."""

import numpy as np
import pandas as pd

import Naru.common as common
from utils.path_util import get_absolute_path


DICT_FROM_DATASET_TO_COLS = {
    "census": [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ],
    "forest": [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
}


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def load_csv_dataset(
        dataset_name: str,
        batch_num=None,
        finetune=False
):
    cols = DICT_FROM_DATASET_TO_COLS.get(dataset_name, [])
    # 读取数据
    csv_file = f"./data/{dataset_name}/{dataset_name}.csv"
    csv_file = get_absolute_path(csv_file)
    df = pd.read_csv(csv_file, usecols=cols, sep=",")
    # 处理数据
    df = _clean_df(df)

    if batch_num != None:
        landmarks = int(len(df) * 10 / 12) + np.linspace(
            0, int((len(df) * 10 / 12) * 0.2), 6, dtype=np.int
        )
        df = df.iloc[: landmarks[batch_num]]

    if finetune:
        landmarks = int(len(df) * 10 / 12) + np.linspace(
            0, int((len(df) * 10 / 12) * 0.2), 6, dtype=np.int
        )
        return common.CsvTable(dataset_name, df, cols), landmarks

    # landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
    # df = df.iloc[:landmarks[5]]

    print(f"load_csv_dataset - {dataset_name}", df.shape)
    return common.CsvTable(dataset_name, df, cols)


def _handle_permuted_dataset(
        permute: bool,
        df: pd.DataFrame,
        permuted_csv_file_path,
        dataset_name: str
):
    if permute:
        columns_to_sort = df.columns

        sorted_columns = pd.concat(
            [
                df[col].sort_values(ignore_index=True).reset_index(drop=True)
                for col in columns_to_sort
            ],
            axis=1,
            ignore_index=True,
        )
        sorted_columns.columns = df.columns
        update_sample = sorted_columns.sample(frac=0.2)
        data = pd.concat([df, update_sample])
        landmarks = len(df) + np.linspace(0, len(update_sample), 2, dtype=np.int32)
        save_path = get_absolute_path(permuted_csv_file_path)
        data.to_csv(save_path, sep=",", index=None)
        print(f"{str(save_path)} Saved")
    else:
        # update_sample = df.sample(frac=0.2)
        data = df
        lenth = int(len(data) * 5 / 6)
        landmarks = lenth + np.linspace(0, len(data) - lenth, 2, dtype=np.int32)

    print(
        "data size: {}, total size: {}, split index: {}".format(
            len(df), len(data), landmarks
        )
    )
    del df

    if permute:
        del sorted_columns
        del update_sample

    # print("census data size: {}".format(data.shape))
    return common.CsvTable(dataset_name, data, cols=data.columns), landmarks


def load_permuted_csv_dataset(dataset_name: str, permute=True):
    raw_csv_path = f"./data/{dataset_name}/{dataset_name}.csv"
    permuted_csv_file_path = f"./data/{dataset_name}/permuted_dataset.csv"
    if permute:
        csv_file = raw_csv_path
    else:
        csv_file = permuted_csv_file_path
    csv_file = get_absolute_path(csv_file)
    cols = DICT_FROM_DATASET_TO_COLS.get(dataset_name, [])

    # 读取数据
    df = pd.read_csv(csv_file, usecols=cols, sep=",")
    print(df.shape)

    # 处理数据
    df = _clean_df(df)

    return _handle_permuted_dataset(
        permute, df, permuted_csv_file_path, dataset_name
    )


def LoadPartlyPermutedCensus(filename="census.csv", num_of_sorted_cols=1):
    csv_file = "../data/census/{}".format(filename)
    csv_file = get_absolute_path(csv_file)
    cols = DICT_FROM_DATASET_TO_COLS.get("census", [])
    assert num_of_sorted_cols < len(cols)

    df = pd.read_csv(csv_file, usecols=cols, sep=",")
    print(df.shape)
    # 处理数据
    df = _clean_df(df)

    if num_of_sorted_cols == 0:
        update_sample = df.sample(frac=0.2)
        landmarks = len(df) + np.linspace(0, len(update_sample), 2, dtype=np.int)

        data = pd.concat([df, update_sample])
        del df
        del update_sample
        data.to_csv("permuted_dataset.csv", sep=",", index=None)
        return common.CsvTable("census", df, cols=df.columns)

    columns_to_sort = [df.columns[i] for i in range(num_of_sorted_cols)]
    columns_not_sort = [
        df.columns[i] for i in range(num_of_sorted_cols, len(df.columns))
    ]

    sorted_columns = pd.concat(
        (
                [
                    df[col].sort_values(ignore_index=True).reset_index(drop=True)
                    for col in columns_to_sort
                ]
                + [df[col] for col in columns_not_sort]
        ),
        axis=1,
        ignore_index=True,
    )
    sorted_columns.columns = df.columns

    update_sample = sorted_columns.sample(frac=0.2)
    landmarks = len(df) + np.linspace(0, len(update_sample), 2, dtype=np.int)

    data = pd.concat([df, update_sample])
    del df
    del update_sample

    data.to_csv("permuted_dataset.csv", sep=",", index=None)
    return common.CsvTable("census", data, cols=data.columns)


def _load_npy_as_df(abs_file_path):
    data = np.load(abs_file_path)

    # Calculate the number of columns in the data
    num_columns = data.shape[1]

    # Generate column names based on the number of columns
    column_names = [f'col-{i + 1}' for i in range(num_columns)]

    # Create the DataFrame with the generated column names
    df = pd.DataFrame(data, columns=column_names)
    return df, column_names


def load_npy_dataset(dataset_name: str):
    # 读取数据
    file_path = f"./FACE/data/{dataset_name}.npy"
    abs_file_path = get_absolute_path(file_path)
    df, cols = _load_npy_as_df(abs_file_path)

    # 处理数据
    df = _clean_df(df)

    print("_load_npy_dataset - df.shape =", df.shape)

    return common.CsvTable(dataset_name, df, cols)


def load_permuted_npy_dataset(dataset_name: str, permute=True):
    # 读取数据
    npy_file_path = f"./FACE/data/{dataset_name}.npy"
    permuted_csv_file_path = f"./data/{dataset_name}/permuted_dataset.csv"
    if permute:
        # 读取原始npy文件
        abs_file_path = get_absolute_path(npy_file_path)
        df, cols = _load_npy_as_df(abs_file_path)
    else:
        # 读取permute后的csv文件
        abs_file_path = get_absolute_path(permuted_csv_file_path)
        df = pd.read_csv(abs_file_path, sep=",")

    # 处理数据
    df = _clean_df(df)

    return _handle_permuted_dataset(
        permute, df, permuted_csv_file_path, dataset_name
    )


if __name__ == "__main__":
    pass
