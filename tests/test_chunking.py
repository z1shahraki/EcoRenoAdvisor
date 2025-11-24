from ingestion.chunk_and_embed import chunk_text


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(50))
    chunk_size = 10
    overlap = 2

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    assert chunks, "Expected chunk_text to return chunks for non-empty text"
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk.split()) <= chunk_size for chunk in chunks)

    for first, second in zip(chunks, chunks[1:]):
        overlap_words = set(first.split()[-overlap:]) & set(second.split()[:overlap])
        assert overlap_words, "Consecutive chunks should share overlapping words"

