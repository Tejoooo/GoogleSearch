
from sentence_transformers import SentenceTransformer

# Load from local directory
model_path = "/workspaces/GoogleSearch/all-MiniLM-L6-v2"
model = SentenceTransformer(model_path)

# Test it
embeddings = model.encode(["Hello world", "How are you?"])
print(embeddings.shape)
