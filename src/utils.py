import logging

from mlflow.tracking import MlflowClient


class MlflowWriter:
    def __init__(self, experiment_name, tracking_uri):
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except Exception as e:
            print(e)
            self.experiment_id = self.client.get_experiment_by_name(
                experiment_name
            ).experiment_id

        self.experiment = self.client.get_experiment(self.experiment_id)
        self.logger_mlflow = logging.getLogger("mlflow")
        self.logger_mlflow.setLevel(logging.INFO)
        self.logger_mlflow.info("New experiment started")
        self.logger_mlflow.info(f"Name: {self.experiment.name}")
        self.logger_mlflow.info(f"Experiment_id: {self.experiment.experiment_id}")
        self.logger_mlflow.info(
            f"Artifact Location: {self.experiment.artifact_location}"
        )

    def log_params(self, item_dict):
        for key, value in item_dict.items():
            self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_metric_step(self, key, value, step):
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, local_path):
        self.client.log_artifact(
            self.run_id,
            local_path,
        )

    def log_dict(self, dictionary, file):
        self.client.log_dict(self.run_id, dictionary, file)

    def log_figure(self, figure, file):
        self.client.log_figure(self.run_id, figure, file)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)

    def create_registered_model(self, model_name):
        self.client.create_registered_model(model_name)

    def create_model_version(self, name, source):
        self.client.create_model_version(name=name, source=source, run_id=self.run_id)

    def create_new_run(self, tags=None):
        self.run = self.client.create_run(self.experiment_id, tags=tags)
        self.run_id = self.run.info.run_id
        self.logger_mlflow.info(f"New run started: {tags['mlflow.runName']}")
