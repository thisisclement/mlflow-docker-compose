from transformers import AutoTokenizer, AutoModel
import torch
import mlflow
import os

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 
'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L12-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings_np = [i.detach().cpu().numpy() for i in sentence_embeddings]

print("Sentence embeddings:")
print(sentence_embeddings_np[0])

os.environ['MLFLOW_TRACKING_URI']      = "http://0.0.0.0:5000"
os.environ['MLFLOW_TRACKING_USERNAME'] = "mlflow_user"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "mlflow_pass"
mlflow.set_experiment("sentbert")

with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, "model")


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L12-v2')
embeddings = model.encode(sentences)
print("Sentence Transformer embeddings 1:", embeddings[0])
print("Automodel Transformer embeddings 1:", sentence_embeddings_np[0])

compare = True
for i, j in zip(embeddings, sentence_embeddings_np):
    if i.all() != j.all():
        compare = False

if compare == True:
    print("Both arrays are the same!")
else:
    raise("well well, something is not quite right here.")
