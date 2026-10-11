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
        """Messages for the LLM: a short, plain answer to "is this photo AI-generated?", with a little context."""
        context = ""
        for chunk in retrieved_chunks:
            context += (
                f"\n\n[Source: {chunk['source']} {chunk['year']}]\n{chunk['text']}"
            )

        if prediction == "REAL":
            verdict = "a real photograph, not AI-generated"
            caution = "images from newer generators (such as diffusion models), or heavily edited or compressed photos, can still fool it"
        else:
            verdict = "AI-generated (a GAN-made face)"
            caution = "heavy compression, filters or unusual lighting can make a real photo look fake to it"
        unsure = (
            " Say plainly that the detector is not sure." if confidence < 0.75 else ""
        )

        system = (
            "You explain the result of an AI-image detector to a person who uploaded a photo and wants to know "
            "whether it is AI-generated. You have not seen the photo: you only know the detector's verdict and "
            "confidence, plus a few passages from deepfake-forensics papers. Never say you examined the image, and "
            "never claim any particular check was run on it. Write for a general reader."
        )
        user = f"""The detector, an Xception network trained on real photos and GAN-generated faces, says this photo is {verdict}, with {confidence:.1%} confidence.

Research passages:{context}

Write one short paragraph of 3 or 4 plain sentences, under 90 words, with no headings, lists, tables or bold text.
Start by saying what the verdict means in everyday words.{unsure}
Then give one or two things that detectors like this look for in AI-generated faces, taken from the passages, naming the source in brackets the way it appears in a passage header, for example (FaceForensics++).
End with one honest caution: {caution}.
Only cite sources named in the [Source: ...] headers above."""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate_explanation(self, messages):
        """Send the messages to the Groq LLM and return the explanation."""
        if self.groq_client is None:
            raise RuntimeError(
                "RAGPipeline.load() must be called before generate_explanation()."
            )

        def ask():
            options = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 800,
            }
            if self.model_name.startswith("openai/gpt-oss"):
                options["reasoning_effort"] = (
                    "low"  # keep the reasoning short so the answer fits
                )
            return self.groq_client.chat.completions.create(**options)

        try:
            response = ask()
        except NotFoundError:
            # the model was retired while the server was running: choose again and retry once
            self.model_name = pick_model(self.groq_client)
            response = ask()
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("The LLM returned an empty explanation.")
        return text

    def explain(self, prediction, confidence):
        """Full RAG pipeline: retrieve -> prompt -> generate."""
        if prediction == "REAL":
            query = "how detectors tell real photographs from GAN-generated faces"
        else:
            query = "visual artifacts that reveal GAN-generated deepfake faces"
        retrieved, distances = self.retrieve(query, k=5)
        messages = self.build_prompt(prediction, confidence, retrieved)
        return self.generate_explanation(messages)


if __name__ == "__main__":
    rag = RAGPipeline()
    rag.load()
    explanation = rag.explain("DEEPFAKE", 0.94)  # my test example
    print(explanation)
