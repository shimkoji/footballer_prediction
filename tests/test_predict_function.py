import numpy as np
import pandas as pd

from src import const
from src.predict import predict
from src.preprocessing import create_dataset, prep_dataset


def test_predict_function():
    df_X_test, df_Y_test = create_dataset(
        start_year=18,
        end_year=22,
        age_span=(15, 22),
        target_feature_list=const.TARGET_COL_LIST,
    )

    df_X_test_dropped, df_Y_test_dropped = prep_dataset(df_X_test, df_Y_test)
    sample_player_feature = pd.DataFrame(df_X_test_dropped.iloc[0]).T

    df_Y_pred = predict(sample_player_feature)
    print(df_Y_pred)
    assert isinstance(df_Y_pred, np.float32) is True
