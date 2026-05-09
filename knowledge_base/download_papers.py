from pathlib import Path

import requests  # type: ignore[reportMissingImports]

PDF_DIR = Path(__file__).resolve().parent / "pdfs"

PAPERS = [
    {"url": "https://arxiv.org/pdf/1901.08971", "source": "FaceForensics++", "year": 2019},
    {"url": "https://arxiv.org/pdf/2001.00179", "source": "DeepFakes and Beyond Survey", "year": 2020},
    {"url": "https://arxiv.org/pdf/1610.02357", "source": "Xception", "year": 2017},
    {"url": "https://arxiv.org/pdf/2004.10448", "source": "Watch Your Up-Convolution", "year": 2020},
    {"url": "https://arxiv.org/pdf/2403.17881", "source": "Deepfake Generation and Detection Survey", "year": 2024},
    {"url": "https://arxiv.org/pdf/2403.07240", "source": "FreqNet Frequency-Aware Detection", "year": 2024},
    {"url": "https://arxiv.org/pdf/2211.10881", "source": "Deepfake Detection Reliability Survey", "year": 2022},
    {"url": "https://arxiv.org/pdf/2407.06174", "source": "Tug of War Deepfake Detection", "year": 2024},
    {"url": "https://arxiv.org/pdf/2411.19537", "source": "Deepfake Detection Generative AI Era", "year": 2024},
    {"url": "https://arxiv.org/pdf/2005.11401", "source": "RAG Lewis et al", "year": 2020},
]


def download_pdf(url: str, save_path: Path) -> None:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded: {save_path}")


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for paper in PAPERS:
        pdf_path = PDF_DIR / f"{paper['source'].replace(' ', '_')}.pdf"
        if pdf_path.exists():
            print(f"Already exists, skipping: {pdf_path}")
            continue
        download_pdf(paper["url"], pdf_path)


if __name__ == "__main__":
    main()
