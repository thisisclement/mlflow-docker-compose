from sentence_transformers import SentenceTransformer
from sys import version_info
import sentence_transformers
import cloudpickle
import mlflow
from experiment_recorder import * 

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

sbert_path = "/tmp/model/sbert_paraphrase-MiniLM-L12-v2.pth"

# sbert = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L12-v2")
# sbert.save(sbert_path)

artifacts = {
    "sbert_miniLM-L12-v2": sbert_path
}


class SbertModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # load your artifacts
        self.sbert_model = SentenceTransformer(context.artifacts["sbert_miniLM-L12-v2"])
        

    def predict(self, context, model_input):
        embeddings = self.sbert_model.encode(model_input)
        return embeddings



# Create a Conda environment for the new MLflow Model that contains all necessary dependencies.
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'sentence_transformers=={}'.format(sentence_transformers.__version__),
          'cloudpickle=={}'.format(cloudpickle.__version__),
        ],
      },
    ],
    'name': 'sbert_env'
}

print(conda_env)

mlflow_pyfunc_model_path = "sbert_miniLM-L12-v2_mlflow_pyfunc"

mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path, python_model=SbertModel(), artifacts=artifacts,
        conda_env=conda_env)

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

print(loaded_model.predict(["this is me", "the greatest show on earth!"]))

#Log model in experiment run
exp = ExperimentRecorder('Default')
mlflow.pyfunc.log_model(artifact_path="model", python_model= SbertModel(), artifacts=artifacts, conda_env=conda_env)
exp.end_run()