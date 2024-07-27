import pickle
import shutil
from pathlib import Path

import lightgbm as lgb
import mlflow
import xgboost as xgb
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_SOURCE_NAME, MLFLOW_USER
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from utils import MlflowWriter

data_dir = Path().cwd().joinpath("data")
model_dir = Path().cwd().joinpath("model")
RANDOM_SEED = 0

xgb_params = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "n_estimators": [100, 200, 300],
    # "min_child_weight": [1, 3, 5],
    # "subsample": [0.7, 0.8, 0.9],
}
lgbm_params = {
    "num_leaves": [5, 10, 20],
    "learning_rate": [0.01, 0.1, 0.3],
    "colsample_bytree": [0.5],
    # "subsample": [0.7, 0.8, 0.9],
    "n_estimators": [100, 200, 300],
}
rf_params = {
    "bootstrap": [True, False],
    "max_depth": [10, 20],
    # "max_features": ["auto", "sqrt"],
    "min_samples_leaf": [1, 2, 4],
    # "min_samples_split": [2, 5, 10],
    "n_estimators": [100, 200],
}


class Experiment:
    def __init__(
        self,
        model_name,
        param_grid,
        df_X_train,
        df_Y_train,
        tracking_uri,
    ) -> None:
        self.model_name = model_name
        if self.model_name == "xgboost":
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror", random_state=RANDOM_SEED
            )

        elif self.model_name == "lightgbm":
            self.model = lgb.LGBMRegressor(
                objective="regression",
                metric="mse",
                random_state=RANDOM_SEED,
                boosting_type="gbdt",
            )

        elif self.model_name == "randomforest":
            self.model = RandomForestRegressor(random_state=RANDOM_SEED)
        else:
            raise ValueError("model_name should be xgboost, lightgbm, or randomforest")
        self.param_grid = param_grid
        self.df_X_train = df_X_train
        self.df_Y_train = df_Y_train
        self.tracking_uri = tracking_uri

    def run(self):
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=4,
        )
        self.grid_search.fit(self.df_X_train, self.df_Y_train)

    def log_mlflow(self, model_dir):
        EXPERIMENT_NAME = f"footballer_pred_{self.model_name}"
        EXPERIMENT_PATH = model_dir.joinpath(EXPERIMENT_NAME)
        if not EXPERIMENT_PATH.exists():
            EXPERIMENT_PATH.mkdir(parents=True, exist_ok=True)
        writer = MlflowWriter(EXPERIMENT_NAME, self.tracking_uri)

        for i, params in enumerate(self.grid_search.cv_results_["params"]):
            tags = {
                "trial": 0,
                MLFLOW_RUN_NAME: f"{self.model_name}_grid_search_{i}",
                MLFLOW_USER: "shimkoji",
                MLFLOW_SOURCE_NAME: "default",
            }
            writer.create_new_run(tags)
            writer.log_params(params)
            writer.log_metric(
                "mean_test_score", self.grid_search.cv_results_["mean_test_score"][i]
            )
            writer.log_metric(
                "std_test_score", self.grid_search.cv_results_["std_test_score"][i]
            )
            model = xgb.XGBRegressor(**params)
            model.fit(self.df_X_train, self.df_Y_train)
            with open(EXPERIMENT_PATH.joinpath("model.pickle"), "wb") as f:
                pickle.dump(model, f)
            writer.log_artifact(local_path=EXPERIMENT_PATH.joinpath("model.pickle"))
            writer.set_terminated()
        shutil.rmtree(EXPERIMENT_PATH)


def do_experiments(
    df_X_train_dropped, df_Y_train_dropped, tracking_uri="sqlite:///mlflow.db"
):

    experiment_v1 = Experiment(
        model_name="xgboost",
        param_grid=xgb_params,
        df_X_train=df_X_train_dropped,
        df_Y_train=df_Y_train_dropped,
        tracking_uri=tracking_uri,
    )
    experiment_v1.run()
    experiment_v1.log_mlflow(model_dir=model_dir)

    experiment_v2 = Experiment(
        model_name="lightgbm",
        param_grid=lgbm_params,
        df_X_train=df_X_train_dropped,
        df_Y_train=df_Y_train_dropped,
        tracking_uri=tracking_uri,
    )
    experiment_v2.run()
    experiment_v2.log_mlflow(model_dir=model_dir)

    experiment_v3 = Experiment(
        model_name="randomforest",
        param_grid=rf_params,
        df_X_train=df_X_train_dropped,
        df_Y_train=df_Y_train_dropped,
        tracking_uri=tracking_uri,
    )
    experiment_v3.run()
    experiment_v3.log_mlflow(model_dir=model_dir)
    model_name_list = ["xgboost", "lightgbm", "randomforest"]
    experiment_names = [
        f"footballer_pred_{model_name}" for model_name in model_name_list
    ]
    mlflow.set_tracking_uri(tracking_uri)
    best_run = mlflow.search_runs(
        experiment_names=experiment_names,
        max_results=1,
        order_by=["metrics.mean_test_score DESC"],
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    # result = client.create_model_version(
    #     name="footballer_prediction",
    #     source=best_run["artifact_uri"][0],
    #     run_id=best_run["run_id"][0],
    # )
    run_id = best_run["run_id"][0]
    client.download_artifacts(run_id, "./model.pickle", dst_path="./model/best_model/")
