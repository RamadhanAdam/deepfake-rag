from rag import RAGPipeline

rag = RAGPipeline()


def test_load():
    """Test that RAG loads without errors."""
    rag.load()
    assert rag.index is not None
    assert rag.chunks is not None
    assert rag.embedding_model is not None
    assert rag.groq_client is not None
    print("PASS: RAG loaded successfully")


def test_retrieve():
    """Test that retrieval returns correct number of chunks."""
    results, distances = rag.retrieve("GAN artifacts deepfake", k=3)
    assert len(results) == 3
    assert all("text" in r for r in results)
    assert all("source" in r for r in results)
    print("PASS: Retrieval works correctly")


def test_explain_deepfake():
    """Test full pipeline with a deepfake prediction."""
    explanation = rag.explain("DEEPFAKE", 0.94)
    assert len(explanation) > 50
    print("PASS: Deepfake explanation generated")
    print(f"\nExplanation:\n{explanation}")


def test_explain_real():
    """Test full pipeline with a real image prediction."""
    explanation = rag.explain("REAL", 0.87)
    assert len(explanation) > 50
    print("PASS: Real image explanation generated")
    print(f"\nExplanation:\n{explanation}")


if __name__ == "__main__":
    print("Running RAG tests...\n")
    test_load()
    test_retrieve()
    test_explain_deepfake()
    test_explain_real()
    print("\nAll tests passed.")