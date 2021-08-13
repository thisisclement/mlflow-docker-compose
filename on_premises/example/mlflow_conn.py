import os
import mlflow

class MLflowConnect():
    """
    Simple wrapper to connect to MLflow server with the desired environment variables. 
    """
    def __init__(self, uri='http://0.0.0.0:5000', username='mlflow_user', password='mlflow_pwd'):
        os.environ['MLFLOW_TRACKING_URI']      = uri
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password

    def load_model(self, run_id):
        self.logged_model = f"runs:/{run_id}/model"
        return mlflow.pytorch.load_model(self.logged_model)
