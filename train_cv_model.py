import os
import cv2
import numpy as np
import glob
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import time

def extract_features(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read {image_path}")
    
    img_resized = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    
    r = 30
    mask = np.ones((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), r, 0, -1)
    
    high_freq_energy = np.sum(magnitude_spectrum * mask)
    total_energy = np.sum(magnitude_spectrum)
    high_freq_ratio = high_freq_energy / (total_energy + 1e-6)
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (h * w)
    
    return [high_freq_ratio, laplacian_var, edge_density]

def main():
    X = []
    y = []
    
    valid_exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.heic", "*.HEIC"]
    
    real_paths = []
    for ext in valid_exts:
        real_paths.extend(glob.glob(os.path.join("photo set", ext)))
        
    for img_path in real_paths:
        try:
            X.append(extract_features(img_path))
            y.append(0)
        except Exception:
            pass

    screen_paths = []
    for ext in valid_exts:
        screen_paths.extend(glob.glob(os.path.join("photosetss", ext)))
        
    for img_path in screen_paths:
        try:
            X.append(extract_features(img_path))
            y.append(1)
        except Exception:
            pass

    X = np.array(X)
    y = np.array(y)
    
    print(f"Extracted features for {len(X)} images.")
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nValidation Accuracy: {acc * 100:.2f}%")
    
    joblib.dump(clf, "cv_model.pkl")
    print("Model saved as cv_model.pkl")

if __name__ == "__main__":
    main()
