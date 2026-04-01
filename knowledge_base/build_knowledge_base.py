import fitz
import re
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PAPERS = [
    {"source": "FaceForensics++", "year": 2019, "url": "https://arxiv.org/abs/1901.08971"},
    {"source": "DeepFakes and Beyond Survey", "year": 2020, "url": "https://arxiv.org/abs/2001.00179"},
    {"source": "Xception", "year": 2017, "url": "https://arxiv.org/abs/1610.02357"},
    {"source": "Watch Your Up-Convolution", "year": 2020, "url": "https://arxiv.org/abs/2004.10448"},
    {"source": "Deepfake Generation and Detection Survey", "year": 2024, "url": "https://arxiv.org/abs/2403.17881"},
    {"source": "FreqNet Frequency-Aware Detection", "year": 2024, "url": "https://arxiv.org/abs/2403.07240"},
    {"source": "Deepfake Detection Reliability Survey", "year": 2022, "url": "https://arxiv.org/abs/2211.10881"},
    {"source": "Tug of War Deepfake Detection", "year": 2024, "url": "https://arxiv.org/abs/2407.06174"},
    {"source": "Deepfake Detection Generative AI Era", "year": 2024, "url": "https://arxiv.org/abs/2411.19537"},
    {"source": "RAG Lewis et al", "year": 2020, "url": "https://arxiv.org/abs/2005.11401"},
]


def extract_text(pdf_path):
    """Extract all text from PDF page by page."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    return pages_text


def chunk_text(pages_text, source, year, url):
    """Chunk text into paragraphs of 200-2000 chars, skipping references/figures/tables."""
    chunks = []

    for page_text in pages_text:
        lines = page_text.split("\n")
        buffer = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue
            buffer += " " + line

            if len(buffer) >= 300:
                para = re.sub(r"\s+", " ", buffer).strip()
                if 200 <= len(para) <= 2000:
                    if not any(para.startswith(x) for x in ["References", "Acknowledgment", "Figure", "Table"]):
                        if not re.match(r"^\d+$", para):
                            if not re.match(r"^\[\d+\]", para):
                                chunks.append({
                                    "text": para,
                                    "source": source,
                                    "year": year,
                                    "url": url
                                })
                buffer = ""

    return chunks

def build_knowledge_base():
    all_chunks = []
    chunk_id = 0

    for paper in PAPERS:
        pdf_path = f"knowledge_base/pdfs/{paper['source'].replace(' ', '_')}.pdf"

        if not os.path.exists(pdf_path):
            print(f"Missing PDF: {pdf_path} — skipping")
            continue

        print(f"\nProcessing: {paper['source']}")
        pages_text = extract_text(pdf_path)
        chunks = chunk_text(pages_text, paper["source"], paper["year"], paper["url"])

        for chunk in chunks:
            chunk["id"] = chunk_id
            chunk_id += 1

        all_chunks.extend(chunks)
        print(f"Got {len(chunks)} chunks")

    with open("knowledge_base/chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Saved to chunks.json")

    # Building and saving FAISS index
    print("\nBuilding FAISS index...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "knowledge_base/faiss.index")
    print("FAISS index built and saved to knowledge_base/faiss.index")


if __name__ == "__main__":
    build_knowledge_base()