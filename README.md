---
title: Deepfake RAG
emoji: 🔍
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

# Explainable Deepfake Detection with RAG

A deepfake detection system combining an **Xception CNN** classifier with a **Retrieval-Augmented Generation (RAG)** pipeline to produce grounded forensic explanations for every prediction — built from scratch without LangChain.

---

## Live Demo

**Frontend**: https://deepfake-rag.vercel.app  
**Backend API**: https://ramadhanzome-deepfake-rag.hf.space

---

## Architecture

![architecture](utilities/architecture.jpg)

---

## Dataset

The Xception classifier was trained on the **140k Real and Fake Faces** dataset:

- **Source**: [Kaggle — xhlulu/140k-real-and-fake-faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Fake images**: GAN-generated faces
- **Split**: 50k train / 10k valid / 10k test per class (100k train, 20k valid, 20k test total)
- **Input size**: Resized to 299×299, normalized to mean=0.5, std=0.5
- **Label mapping**: `fake=0, real=1` (alphabetical, via `ImageFolder`)
- **Training hardware**: Kaggle T4 x2 GPU with mixed precision (AMP)

---

## How Deepfakes Work

A deepfake is a face image or video generated or manipulated by a Generative Adversarial Network (GAN). Two networks compete:

- **Generator** — creates fake faces
- **Discriminator** — tries to detect fakes

They train together until the generator fools the discriminator. The result looks real to the human eye but leaves behind subtle artifacts from the generation process.

---

## GAN Artifacts This System Detects

1. **Eye inconsistencies** — Reflections, gaze direction, and blinking patterns are often unnatural or asymmetric between left and right eye.
2. **Blending boundaries** — Color and texture inconsistencies appear around the jaw, hairline, and neck where the fake face meets the original.
3. **High frequency noise** — Unnatural frequency patterns invisible to the human eye but detectable via FFT analysis.
4. **Upsampling artifacts** — Checkerboard patterns from GAN upsampling at pixel level.
5. **Facial asymmetry** — GAN faces are often unnaturally symmetric or asymmetric in ways real faces are not.
6. **Temporal inconsistencies** — In videos, deepfake faces flicker or transition unnaturally between frames.

---

## Knowledge Base

RAG retrieves from a knowledge base built from 10 peer-reviewed papers:

| Paper | Year |
|-------|------|
| FaceForensics++ | 2019 |
| DeepFakes and Beyond Survey | 2020 |
| Xception | 2017 |
| Watch Your Up-Convolution | 2020 |
| Deepfake Generation and Detection Survey | 2024 |
| FreqNet Frequency-Aware Detection | 2024 |
| Deepfake Detection Reliability Survey | 2022 |
| Tug of War Deepfake Detection | 2024 |
| Deepfake Detection in Generative AI Era | 2024 |
| RAG — Lewis et al. | 2020 |

The knowledge base can be updated at any time by adding new papers and rebuilding the FAISS index — no retraining required.

---

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| Validation accuracy | ~97% | 20,000 held-out face images |
| Inference speed | <2 seconds | CPU only, no GPU at inference |
| Confidence on clear fakes | 100% | Images with strong artefacts |
| Knowledge base | 2,665 chunks | From 10 peer-reviewed papers |
| RAG retrieval | Top 5 per query | FAISS cosine similarity |

---
## Project Structure
```
deepfake_rag/
├── api.py                       # FastAPI backend
├── predict.py                   # Xception CNN inference
├── rag.py                       # RAG pipeline
├── train.py                     # model training script
├── main.py                      # CLI entry point
├── requirements.txt
├── Dockerfile
├── frontend/
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       ├── index.css
│       └── main.jsx
├── knowledge_base/
│   ├── download_papers.py       # download research papers
│   ├── build_knowledge_base.py  # chunk, embed, build FAISS index
│   └── chunks.json              # processed knowledge base
├── models/
│   └── xception.py              # Xception architecture
├── tests/
│   └── test.py
├── utilities/                   # architecture diagram + training curves
└── xception-deepfake-detector.ipynb  # training notebook
```

---

## Installation
```bash
git clone https://github.com/RamadhanAdam/deepfake-rag
cd deepfake-rag

conda create -n deepfake_rag python=3.10
conda activate deepfake_rag

pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_key_here
```

---

## Usage

**Run API locally:**
```bash
uvicorn api:app --reload
```

**Build knowledge base:**
```bash
cd knowledge_base
python build_knowledge_base.py
```

**Run tests:**
```bash
python tests/test.py
```

---

## Limitations
- **GAN-focused**: Trained on GAN fakes only — diffusion model outputs (Stable Diffusion, Midjourney) are not yet handled
- **Images only**: No video/temporal analysis yet
- **Cold start**: First request can take ~50s on free tier while model weights load
- **Static knowledge base**: FAISS index requires manual updates when new papers are published

---

## Deployment

| Component | Platform | Details |
|-----------|----------|---------|
| Backend | HuggingFace Spaces | Docker, CPU Basic, 16GB RAM |
| Model weights | HuggingFace Hub | `RamadhanZome/deepfake-xception` |
| Frontend | Vercel | Auto-deploys on push to main |
| CI/CD | GitHub Actions | Builds and pushes Docker image to GHCR |