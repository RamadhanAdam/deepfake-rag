"""
predict.py
Load the trained Xception model and classifies new images as real or fake. Usage:
Returns the label and confidence score for each image in the specified folder.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models.xception import Xception

# MODEL_PATH = 'models/best_xception.pth'
MODEL_PATH = 'RamadhanZome/deepfake-xception'
WEIGHTS_FILE = 'best_xception.pth'
LABELS = ['fake', 'real'] # matches ImageFolder alphabetical order

# Preprocessing - must match training transforms exactly
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

# def load_model(model_path: str = MODEL_PATH) -> nn.Module:
#     """Load trained Xception weights onto CPU (safe default for inference)"""
#     model = Xception(num_classes=2)
#     model.load_state_dict(torch.load(model_path, map_location='cpu'))
#     model.eval() # setting to eval mode for inference
#     return model

def load_model()-> nn.Module:
    """Load weights from HuggingFace Hub and load model."""
    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(repo_id=MODEL_PATH, filename=WEIGHTS_FILE)
    model = Xception(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

def predict(image_path: str, model: torch.nn.Module) -> dict:
    """
    Run inference on a single image.
    Returns: { 'label': 'real'|'fake', 'confidence': float (0-100) }
    """
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # add batch dim -> (1, 3, 299, 299)

    with torch.no_grad():
        output = model(input_tensor) # raw logits
        probs = torch.softmax(output, dim = 1) # convert to probabilities
        confidence, predicted_idx = torch.max(probs, dim=1)

    return {
        'label':      LABELS[predicted_idx.item()],
        'confidence': round(confidence.item() * 100, 2)
    }

# Quick test : run directly - python3 predict.py path/to/image.jpg
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 predict.py path/to/image.jpg")
        sys.exit(1)

    model = load_model()
    result = predict(sys.argv[1], model)
    print(f"Prediction: {result['label']} (Confidence: {result['confidence']}%)")