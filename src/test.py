import pandas as pd
import requests

import const
from preprocessing import create_dataset, prep_dataset

df_X_test, df_Y_test = create_dataset(
    start_year=18,
    end_year=22,
    age_span=(15, 22),
    target_feature_list=const.TARGET_COL_LIST,
)

df_X_test_dropped, df_Y_test_dropped = prep_dataset(df_X_test, df_Y_test)


sample_player_feature = pd.DataFrame(df_X_test_dropped.iloc[0]).T.to_json()
url = "http://127.0.0.1:9696/predict"
response = requests.post(url, json=sample_player_feature)
print(response.json())
