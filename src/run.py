import const
from experiment import do_experiments
from preprocessing import create_dataset, prep_dataset

if __name__ == "__main__":
    df_X_train, df_Y_train = create_dataset(
        start_year=15,
        end_year=19,
        age_span=(15, 22),
        target_feature_list=const.TARGET_COL_LIST,
    )
    df_X_train_dropped, df_Y_train_dropped = prep_dataset(df_X_train, df_Y_train)
    do_experiments(
        df_X_train_dropped, df_Y_train_dropped, tracking_uri="sqlite:///mlflow.db"
    )
