from sentence_transformers import  SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_model():
    return _model
