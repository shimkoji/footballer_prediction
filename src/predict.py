import pickle

import pandas as pd
from flask import Flask, jsonify, request

from src.const import TARGET_COL_LIST
from src.preprocessing import create_dataset, prep_dataset

# TRACKING_URI = "http://127.0.0.1:5000"
TRACKING_URI = "sqlite:///mlflow.db"

with open("./model/best_model/model.pickle", "rb") as f:
    best_model = pickle.load(f)


def predict(df_X_test):
    df_Y_pred = best_model.predict(df_X_test)
    return df_Y_pred[0]


app = Flask("predict_player_skills")


@app.route("/predict", methods=["POST"])
def predict_future_score():
    print("hi")
    input_json = request.get_json()
    input_df = pd.read_json(input_json)

    df_Y_pred = predict(input_df).tolist()
    print(df_Y_pred)
    result = {"pred_skills": df_Y_pred}
    print(jsonify(result))
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
