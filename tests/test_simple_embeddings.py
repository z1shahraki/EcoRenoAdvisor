import numpy as np
from sentence_transformers import SentenceTransformer


def test_low_voc_sentence_is_most_similar(sample_sentences):
    """Ensure cosine similarity ranks the low VOC paint sentence highest."""
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    sentences = sample_sentences
    query = "What is a good low VOC option for kids bedroom walls?"

    sentence_embeddings = model.encode(sentences, normalize_embeddings=True)
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    similarities = sentence_embeddings @ query_embedding
    top_match_idx = int(np.argmax(similarities))

    assert (
        sentences[top_match_idx] == sentences[0]
    ), "Expected the low VOC paint sentence to be the closest match"

