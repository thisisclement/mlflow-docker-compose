from mlflow_conn import MLflowConnect
import torch

conn = MLflowConnect()
loaded_model = conn.load_model("c967115edb5942399c8f4996edf2c290")
data = torch.tensor("this is me")
loaded_model(data)