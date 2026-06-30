"""Fill this in. That's the whole interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
"""

import sys
import os
import warnings
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import joblib
from PIL import Image

warnings.filterwarnings("ignore")

weights = models.MobileNet_V3_Small_Weights.DEFAULT
base_model = models.mobilenet_v3_small(weights=weights)

base_model.classifier = torch.nn.Identity()
base_model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_path: str) -> float:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_svm_model.pkl")
    try:
        clf = joblib.load(model_path)
    except FileNotFoundError:
        print("Error: dl_svm_model.pkl not found.", file=sys.stderr)
        return 0.0

    try:
        img = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) 
    except Exception as e:
        print(f"Error processing {image_path}: {e}", file=sys.stderr)
        return 0.0

    with torch.no_grad():
        embedding = base_model(input_batch).squeeze().numpy()

    prob_screen = clf.predict_proba([embedding])[0][1]
    
    return float(prob_screen)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    print(predict(sys.argv[1]))
