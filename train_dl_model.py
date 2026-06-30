import os
import glob
import time
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def get_feature_extractor():
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_embedding(img_path, model, preprocess):
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) 

    with torch.no_grad():
        output = model(input_batch)
    return output.squeeze().numpy()

def main():
    model = get_feature_extractor()
    preprocess = get_transforms()
    
    X, y = [], []
    valid_exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.heic", "*.HEIC"]
    
    real_paths = []
    for ext in valid_exts:
        real_paths.extend(glob.glob(os.path.join("photo set", ext)))
        
    for img_path in real_paths:
        try:
            X.append(extract_embedding(img_path, model, preprocess))
            y.append(0)
        except Exception:
            pass

    screen_paths = []
    for ext in valid_exts:
        screen_paths.extend(glob.glob(os.path.join("photosetss", ext)))
        
    for img_path in screen_paths:
        try:
            X.append(extract_embedding(img_path, model, preprocess))
            y.append(1)
        except Exception:
            pass

    X = np.array(X)
    y = np.array(y)
    
    print(f"Extracted {X.shape[1]}-dim embeddings for {len(X)} images.")
    
    clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    
    start_time = time.time()
    _ = extract_embedding(real_paths[0], model, preprocess)
    latency_ms = (time.time() - start_time) * 1000
    print(f"Feature Extraction Latency: {latency_ms:.2f} ms")
    
    joblib.dump(clf, "dl_svm_model.pkl")
    print("Model saved as dl_svm_model.pkl")

if __name__ == "__main__":
    main()
