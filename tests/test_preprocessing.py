from src import const
from src.preprocessing import create_dataset, prep_dataset


def test_predict_function():
    df_X_test, df_Y_test = create_dataset(
        start_year=18,
        end_year=22,
        age_span=(15, 22),
        target_feature_list=const.TARGET_COL_LIST,
    )
    assert df_X_test.shape[1] == 47
    assert df_X_test.shape[0] == df_Y_test.shape[0]
    df_X_test_dropped, df_Y_test_dropped = prep_dataset(df_X_test, df_Y_test)
    assert df_X_test_dropped.shape[1] == 45
    assert df_X_test_dropped.shape[0] == df_Y_test_dropped.shape[0]
