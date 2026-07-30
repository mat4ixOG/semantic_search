from sentence_transformers import  SentenceTransformer
import config

MODEL_NAME = config.EMBEDDING_MODEL

_model = SentenceTransformer(MODEL_NAME)

def get_model():
    return _model
