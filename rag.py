import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


class RAGPipeline:
    def __init__(self):
        self.index = None
        self.chunks = None
        self.embedding_model = None
        self.groq_client = None

    def load(self):
        """Load chunks, embed them, build FAISS index, initialize Groq client."""
        with open("knowledge_base/chunks.json", "r") as f:
            self.chunks = json.load(f)

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        if os.path.exists("knowledge_base/faiss.index"):
            self.index = faiss.read_index("knowledge_base/faiss.index")
        else:
            embeddings = self.embedding_model.encode(
                [chunk["text"] for chunk in self.chunks]
            ).astype("float32")
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(embeddings)

        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def retrieve(self, query, k=5):
        """Embed query and retrieve top-K most similar chunks from FAISS."""
        query_vector = self.embedding_model.encode([query]).astype("float32")
        D, I = self.index.search(query_vector, k)
        retrieved = [self.chunks[idx] for idx in I[0]]
        return retrieved, D[0]

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