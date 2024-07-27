from pathlib import Path

import pandas as pd

data_dir = Path().cwd().joinpath("data")
print(data_dir)


def create_dataset(
    start_year,
    end_year,
    age_span,
    target_feature_list,
):
    data_start = pd.read_csv(data_dir.joinpath(f"players_{start_year}.csv"))
    data_start_selected = (
        data_start.query("age >= @age_span[0] & age <= @age_span[1]")
        .reset_index()
        .drop("index", axis=1)
        .copy()
    )
    data_start_selected.columns = [
        col + f"-{start_year}" for col in data_start_selected.columns
    ]
    data_end = pd.read_csv(data_dir.joinpath(f"players_{end_year}.csv"))
    span = end_year - start_year
    age_span_end = (age_span[0] + span, age_span[1] + span)
    data_end_selected = (
        data_end.query("age >= @age_span_end[0] & age <= @age_span_end[1]")
        .reset_index()
        .drop("index", axis=1)
        .copy()
    )
    data_end_selected.columns = [
        col + f"-{end_year}" for col in data_end_selected.columns
    ]
    data_merged = (
        pd.merge(
            data_start_selected,
            data_end_selected,
            how="left",
            left_on=f"sofifa_id-{start_year}",
            right_on=f"sofifa_id-{end_year}",
        )
        .dropna(subset=[f"overall-{end_year}"])
        .reset_index()
        .drop("index", axis=1)
    )
    # data_merged = data_merged
    selected_col_list = [col + f"-{start_year}" for col in target_feature_list]
    selected_col_list.append(f"sofifa_id-{start_year}")
    df_X = data_merged[selected_col_list]
    df_X = df_X.drop(columns=f"overall-{start_year}")
    df_Y = data_merged[f"overall-{end_year}"]
    df_X.columns = [col.split("-")[0] for col in df_X.columns]
    return df_X, df_Y


def prep_dataset(df_X, df_Y):
    df_X_train_dropped = df_X.drop(
        ["goalkeeping_speed", "mentality_composure"], axis=1
    ).dropna(how="any")
    df_X_train_dropped_index = df_X_train_dropped.index
    df_X_train_dropped = df_X_train_dropped.reset_index().drop("index", axis=1)
    df_Y_train_dropped = (
        df_Y.iloc[df_X_train_dropped_index].reset_index().drop("index", axis=1)
    ).squeeze()
    return df_X_train_dropped, df_Y_train_dropped


def create_master_table(csv_files, columns_list):
    """
    csv_files = list(data_dir.glob("*.csv"))
    test = create_master_table(
        csv_files, columns_list=["sofifa_id", "short_name", "player_positions"]
    )
    """
    for i, csv_file in enumerate(csv_files):
        df_tmp = pd.read_csv(csv_file)[columns_list]
        if i == 0:
            df_base = df_tmp
        else:
            df_base = pd.concat([df_base, df_tmp])
    df_base = df_base.drop_duplicates()
    return df_base
