import json
import faiss  # type: ignore[reportMissingImports]
from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
from groq import Groq  # type: ignore[reportMissingImports]
from dotenv import load_dotenv
import os
from typing import Any, cast

load_dotenv()


class RAGPipeline:
    def __init__(self):
        self.index: Any | None = None
        self.chunks: list[dict[str, Any]] | None = None
        self.embedding_model: Any | None = None
        self.groq_client: Any | None = None

    def load(self):
        """Load chunks, embed them, build FAISS index, initialize Groq client."""
        with open("knowledge_base/chunks.json", "r") as f:
            chunks = cast(list[dict[str, Any]], json.load(f))

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        if os.path.exists("knowledge_base/faiss.index"):
            index = faiss.read_index("knowledge_base/faiss.index")
        else:
            embeddings = embedding_model.encode(
                [chunk["text"] for chunk in chunks]
            ).astype("float32")
            d = embeddings.shape[1]
            index = faiss.IndexFlatL2(d)
            cast(Any, index).add(embeddings)

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")

        self.chunks = chunks
        self.embedding_model = embedding_model
        self.index = index
        self.groq_client = Groq(api_key=groq_api_key)

    def retrieve(self, query, k=5):
        """Embed query and retrieve top-K most similar chunks from FAISS and get the actual chunk data from chunks.json along with their distances."""
        if self.embedding_model is None or self.index is None or self.chunks is None:
            raise RuntimeError("RAGPipeline.load() must be called before retrieve().")

        query_vector = self.embedding_model.encode([query]).astype("float32")
        distances, labels = self.index.search(query_vector, k)
        retrieved = [self.chunks[idx] for idx in labels[0]]
        return retrieved, distances[0]

    def build_prompt(self, prediction, confidence, retrieved_chunks):
        context = ""
        for chunk in retrieved_chunks:
            context += f"\n\n[Source: {chunk['source']} {chunk['year']}]\n{chunk['text']}"

        if prediction == "REAL":
            question = f"The detection model predicted this image is REAL with {confidence:.0%} confidence. Based on the research context, explain what characteristics make an image likely to be authentic and why this prediction may be reliable or unreliable."
        else:
            question = f"The detection model predicted this image is a DEEPFAKE with {confidence:.0%} confidence. Based on the research context, explain what artifacts or characteristics are typically present in deepfakes and why this prediction may be reliable or unreliable."

        prompt = f"""You are a forensic deepfake detection expert.

    CONTEXT (relevant research):
    {context}

    QUESTION:
    {question}
    Ground your explanation in the context above. If the context does not contain enough information, say so.

    EXPLANATION:"""
        return prompt

    def generate_explanation(self, prompt):
        """Send prompt to Groq LLM and return the generated explanation."""
        if self.groq_client is None:
            raise RuntimeError("RAGPipeline.load() must be called before generate_explanation().")

        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def explain(self, prediction, confidence):
        """Full RAG pipeline: retrieve -> prompt -> generate."""
        query = f"deepfake detection artifacts visual forensics {prediction.lower()} image characteristics"
        retrieved, distances = self.retrieve(query, k=5)
        prompt = self.build_prompt(prediction, confidence, retrieved)
        explanation = self.generate_explanation(prompt)
        return explanation


if __name__ == "__main__":
    rag = RAGPipeline()
    rag.load()
    explanation = rag.explain("DEEPFAKE", 0.94) # my test example
    print(explanation)
