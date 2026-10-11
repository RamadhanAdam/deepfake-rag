import json
import faiss  # type: ignore[reportMissingImports]
from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
from groq import Groq, NotFoundError  # type: ignore[reportMissingImports]
from dotenv import load_dotenv
import os
from typing import Any, cast

load_dotenv()

# Groq retires models from time to time (llama-3.3-70b-versatile went in 2026), so the model is chosen at
# startup from what the key can actually use: GROQ_MODEL if set, else the first available preferred model,
# else any available chat model.
PREFERRED_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
]
NOT_CHAT = ("whisper", "guard", "tts", "playai", "orpheus", "compound", "distil")


def pick_model(client: Any) -> str:
    """Name of a Groq chat model this key can use."""
    if os.getenv("GROQ_MODEL"):
        return os.environ["GROQ_MODEL"]
    available = [m.id for m in client.models.list().data]
    for name in PREFERRED_MODELS:
        if name in available:
            return name
    chat = sorted(m for m in available if not any(word in m for word in NOT_CHAT))
    if not chat:
        raise RuntimeError(
            f"No chat model available to this Groq key (found: {available})."
        )
    return chat[0]


class RAGPipeline:
    def __init__(self):
        self.index: Any | None = None
        self.chunks: list[dict[str, Any]] | None = None
        self.embedding_model: Any | None = None
        self.groq_client: Any | None = None
        self.model_name: str | None = None

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
        self.model_name = pick_model(self.groq_client)
        print(f"RAG: explanations will use the Groq model {self.model_name}")

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
            context += (
                f"\n\n[Source: {chunk['source']} {chunk['year']}]\n{chunk['text']}"
            )

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
            raise RuntimeError(
                "RAGPipeline.load() must be called before generate_explanation()."
            )

        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        except NotFoundError:
            # the model was retired while the server was running: choose again and retry once
            self.model_name = pick_model(self.groq_client)
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
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
    explanation = rag.explain("DEEPFAKE", 0.94)  # my test example
    print(explanation)
