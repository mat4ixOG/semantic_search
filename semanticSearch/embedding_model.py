from sentence_transformers import  SentenceTransformer
import config

# Loaded on first use, not on import. Importing a module must not cost the
# RAM of a model the caller may never touch.
_model = None


def get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)

    return _model
