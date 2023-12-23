"""Dataset registrations."""

import numpy as np
import pandas as pd

import Naru.common as common
from utils.path_util import get_absolute_path


def LoadCensus(filename="census.csv", batch_num=None, finetune=False):
    csv_file = "./data/census/{}".format(filename)
    csv_file = get_absolute_path(csv_file)
    cols = [
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
    ]

    df = pd.read_csv(csv_file, usecols=cols, sep=",")
    # df = df[cols]
    df = df.dropna(axis=1, how="all")

    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)
    if batch_num != None:
        landmarks = int(len(df) * 10 / 12) + np.linspace(
            0, int((len(df) * 10 / 12) * 0.2), 6, dtype=np.int
        )
        df = df.iloc[: landmarks[batch_num]]

    if finetune:
        landmarks = int(len(df) * 10 / 12) + np.linspace(
            0, int((len(df) * 10 / 12) * 0.2), 6, dtype=np.int
        )
        return common.CsvTable("census", df, cols), landmarks

    # landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
    # df = df.iloc[:landmarks[5]]

    print("Naru/datasets.py - LoadCensus", df.shape)
    return common.CsvTable("census", df, cols)


def LoadPermutedCensus(filename="census.csv", permute=True):
    if permute:
        csv_file = "./data/census/{}".format(filename)
    else:
        csv_file = "./data/census/permuted_dataset.csv"
    csv_file = get_absolute_path(csv_file)
    cols = [
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
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).

    df = pd.read_csv(csv_file, usecols=cols, sep=",")
    # df = pd.read_csv(csv_file, header=None, sep=",")
    # df.columns = cols
    # print(df.shape)
    df = df.dropna(axis=1, how="all")

    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)

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
        save_path = get_absolute_path("./data/census/permuted_dataset.csv")
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
    return common.CsvTable("census", data, cols=data.columns), landmarks


def LoadPartlyPermutedCensus(filename="census.csv", num_of_sorted_cols=1):
    csv_file = "../data/census/{}".format(filename)
    csv_file = get_absolute_path(csv_file)
    cols = [
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
    ]
    assert num_of_sorted_cols < len(cols)

    df = pd.read_csv(csv_file, usecols=cols, sep=",")
    print(df.shape)
    df = df.dropna(axis=1, how="all")

    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)

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


def LoadForest(filename="forest.csv", batch_num=None, finetune=False):
    csv_file = "./data/forest/{}".format(filename)
    # csv_file = './permuted_dataset.csv'
    csv_file = get_absolute_path(csv_file)
    cols = [
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

    df = pd.read_csv(csv_file, usecols=cols, sep=",")
    # df = df[cols]
    df = df.dropna(axis=1, how="all")

    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)
    if batch_num != None:
        landmarks = int(len(df) * 10 / 12) + np.linspace(
            0, int((len(df) * 10 / 12) * 0.2), 6, dtype=np.int
        )
        df = df.iloc[: landmarks[batch_num]]

    if finetune:
        landmarks = int(len(df) * 10 / 12) + np.linspace(
            0, int((len(df) * 10 / 12) * 0.2), 6, dtype=np.int
        )
        return common.CsvTable("Forest", df, cols), landmarks

    # landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
    # df = df.iloc[:landmarks[5]]
    print(df.shape)

    return common.CsvTable("Forest", df, cols)


def LoadPermutedForest(filename="forest.csv", permute=True):
    if permute:
        csv_file = "./data/forest/{}".format(filename)
    else:
        csv_file = "./data/forest/permuted_dataset.csv"
    csv_file = get_absolute_path(csv_file)
    cols = [
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

    df = pd.read_csv(csv_file, usecols=cols, sep=",")
    print(df.shape)
    df = df.dropna(axis=1, how="all")

    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)

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
        data.to_csv("../data/forest/permuted_dataset.csv", sep=",", index=None)
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
    return common.CsvTable("census", data, cols=data.columns), landmarks


def _load_npy_as_df(abs_file_path):
    data = np.load(abs_file_path)

    # Calculate the number of columns in the data
    num_columns = data.shape[1]

    # Generate column names based on the number of columns
    column_names = [f'col-{i + 1}' for i in range(num_columns)]

    # Create the DataFrame with the generated column names
    df = pd.DataFrame(data, columns=column_names)
    return df, column_names


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all")
    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def _load_npy_dataset(file_name: str, table_name: str):
    # 读取数据
    file_path = "./FACE/data/{}".format(file_name)
    abs_file_path = get_absolute_path(file_path)
    df, cols = _load_npy_as_df(abs_file_path)

    # 处理数据
    df = clean_df(df)

    print("_load_npy_dataset - df.shape =", df.shape)

    return common.CsvTable(table_name, df, cols)


def _load_permuted_npy_dataset(dataset_name, permute=True):
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
    df = clean_df(df)

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

    return common.CsvTable(dataset_name, data, cols=data.columns), landmarks


def LoadBJAQ():
    return _load_npy_dataset(file_name="BJAQ.npy", table_name="bjaq")


def LoadPower():
    return _load_npy_dataset(file_name="power.npy", table_name="power")


def LoadPermutedBJAQ(permute=True):
    return _load_permuted_npy_dataset(dataset_name="BJAQ", permute=permute)


def LoadPermutedPower(permute=True):
    return _load_permuted_npy_dataset(dataset_name="power", permute=permute)


if __name__ == "__main__":
    # LoadPermutedCensus(permute=True)
    # LoadPartlyPermutedCensus()
    pass
